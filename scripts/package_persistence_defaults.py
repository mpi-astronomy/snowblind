"""Write the persistence time of every measured detector into the package, one file per detector.

Reads the ``persistence_<detector>.ecsv`` tables that ``scripts/measure_persistence.py`` writes, fits each
detector's persistence against the start-to-start delay, and writes ``src/snowblind/data/persistence_<detector>.asdf``
with ``time``: the delay at which the fit falls to ``--threshold``.  ``PersistenceFlagStep`` uses the detector's time
when it is given no ``time``, and 2500 s for a detector without a file.

The persistence of one pair of exposures (source, target) and one class of source pixels is the table's
``excess_groups - null_groups``, with the error ``excess_groups_err``.  Only the classes of saturated stars are used
(saturated with more than 60000 DN of signal, or with no rate), not ``sat 0k-60k``: a pixel flagged SATURATED with
little signal is mostly a cosmic ray that saturated late in the ramp.  Pairs whose pointing moved by less than
``--min_shift`` pixels are left out, since a star of the source may then still be on the same pixels.  The fit is

    persistence = A (delay / 1000 s)^-alpha

with weights of one over the error squared, the parameter errors scaled by the root of the reduced chi-square when it
is above one, and ``time = 1000 s (A / threshold)^(1 / alpha)``.  A detector whose fit fails, or gives A or alpha of
zero or less, gets no file.  A time beyond the longest delay measured is written with ``extrapolated`` set.  The file
also keeps the pairs pooled in bins of delay (weights of one over the error squared), for reading.

Usage: ``python scripts/package_persistence_defaults.py <measurement dir> --dataset "<programs, exposures>"``
"""
import argparse
import datetime
import glob
import os

import asdf
import numpy as np
from astropy.table import Table
from scipy.optimize import curve_fit

from snowblind.reset_calib import add_history

PACKAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "snowblind", "data")
# the class of SATURATED pixels with little signal, as measure_persistence.py names it
LATE_SATURATION_CLASS = "sat 0k-60k"
# delay bins of the pooled table [s], as in measure_persistence.py
DELAY_EDGES = [0, 150, 300, 600, 1200, 2400, 4800, 9600, 19200, 38400]
MIN_PAIRS = 10  # fewest (pair, class) rows worth a fit
MODEL_FORM = "persistence = amplitude * (delay / 1000 s)**-alpha, delay start to start, persistence in DN/s"


def star_rows(table, min_shift):
    """The rows of a measurement table for the saturated-star classes, with a finite null, of pairs that moved."""
    keep = np.array([cls.startswith("sat") and cls != LATE_SATURATION_CLASS for cls in table["cls"]])
    keep &= np.isfinite(np.asarray(table["null_groups"], dtype=float))
    keep &= np.asarray(table["excess_groups_err"], dtype=float) > 0
    keep &= np.asarray(table["shift_pix"], dtype=float) > min_shift
    return table[keep]


def model(delay, amplitude, alpha):
    """The decay model at ``delay`` seconds."""
    return amplitude * (delay / 1000.0) ** -alpha


def fit(rows):
    """Fit the decay model to measurement rows; returns a dict, or None when the fit fails or is not a decay."""
    delay = np.asarray(rows["delay_starts"], dtype=float)
    persistence = np.asarray(rows["excess_groups"] - rows["null_groups"], dtype=float)
    error = np.asarray(rows["excess_groups_err"], dtype=float)
    try:
        (amplitude, alpha), covariance = curve_fit(
            model, delay, persistence, p0=(0.03, 1.0), sigma=error, absolute_sigma=True, maxfev=20000
        )
    except (RuntimeError, ValueError):
        return None
    if not (amplitude > 0 and alpha > 0):
        return None
    reduced_chisq = float(np.sum(((persistence - model(delay, amplitude, alpha)) / error) ** 2) / (len(rows) - 2))
    amplitude_error, alpha_error = np.sqrt(np.diag(covariance)) * np.sqrt(max(reduced_chisq, 1.0))
    return {
        "amplitude": float(amplitude),
        "amplitude_error": float(amplitude_error),
        "alpha": float(alpha),
        "alpha_error": float(alpha_error),
        "reduced_chisq": reduced_chisq,
        "n_rows": int(len(rows)),
    }


def pooled(rows):
    """The rows pooled per delay bin: lists of bin edges, persistence, error and number of rows."""
    delay = np.asarray(rows["delay_starts"], dtype=float)
    persistence = np.asarray(rows["excess_groups"] - rows["null_groups"], dtype=float)
    weight = 1.0 / np.asarray(rows["excess_groups_err"], dtype=float) ** 2
    out = {"delay_lo": [], "delay_hi": [], "persistence": [], "error": [], "n_rows": []}
    for low, high in zip(DELAY_EDGES[:-1], DELAY_EDGES[1:]):
        inside = (delay >= low) & (delay < high)
        if inside.any():
            out["delay_lo"].append(float(low))
            out["delay_hi"].append(float(high))
            out["persistence"].append(float(np.sum(weight[inside] * persistence[inside]) / np.sum(weight[inside])))
            out["error"].append(float(1.0 / np.sqrt(np.sum(weight[inside]))))
            out["n_rows"].append(int(inside.sum()))
    return out


def main(argv=None):
    """Console entry: fit each detector's table in a measurement directory and write the package files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("measurement_dir", help="directory holding persistence_<detector>.ecsv tables")
    parser.add_argument("--dataset", required=True, help="what was measured, for the provenance")
    parser.add_argument("--threshold", type=float, default=0.005, help="persistence the time is set at [DN/s]")
    parser.add_argument("--min_shift", type=float, default=100.0, help="pointing shift a pair needs [pix]")
    parser.add_argument("--detectors", nargs="*", help="detectors to package, e.g. nrca1 nrca3 (default: all)")
    parser.add_argument("--out_dir", default=os.path.normpath(PACKAGE_DIR), help="where the per-detector files go")
    args = parser.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(args.measurement_dir, "persistence_*.ecsv")))
    paths = [path for path in paths if not path.endswith("persistence_summary.ecsv")]
    if args.detectors:
        wanted = {d.lower() for d in args.detectors}
        paths = [path for path in paths if os.path.basename(path)[len("persistence_"):-len(".ecsv")] in wanted]

    written = 0
    for path in paths:
        table = Table.read(path)
        if len(table) == 0:
            continue
        detector = str(table["detector"][0]).upper()
        rows = star_rows(table, args.min_shift)
        result = fit(rows) if len(rows) >= MIN_PAIRS else None
        if result is None:
            print(f"{detector}: {len(rows)} rows, no decay fitted; no file written (the step uses 2500 s)")
            continue
        time = 1000.0 * (result["amplitude"] / args.threshold) ** (1.0 / result["alpha"])
        longest = float(np.max(rows["delay_starts"]))
        exposures = sorted(str(name) for name in set(table["source"]) | set(table["target"]))
        entry = {
            "detector": detector,
            "time": float(time),
            "threshold": float(args.threshold),
            "extrapolated": bool(time > longest),
            "longest_delay": longest,
            "model": {"form": MODEL_FORM, **result},
            "measured": pooled(rows),
            "provenance": {
                "dataset": args.dataset,
                "n_exposures": len(exposures),
                "first_exposure": exposures[0],
                "last_exposure": exposures[-1],
                "classes": sorted(str(cls) for cls in set(rows["cls"])),
                "min_shift_pix": float(args.min_shift),
                "measurement": "scripts/measure_persistence.py",
                "table": os.path.basename(path),
            },
            "description": (
                "Persistence time of the detector, used by PersistenceFlagStep when it is given no time: the "
                "start-to-start delay at which the fitted persistence of saturated stars falls to the threshold"
            ),
            "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        }
        out = os.path.join(args.out_dir, f"persistence_{detector.lower()}.asdf")
        with asdf.AsdfFile(entry) as handle:
            add_history(handle, f"Persistence time of {detector} fitted from {os.path.basename(path)}")
            handle.write_to(out)
        written += 1
        note = f", beyond the longest delay measured ({longest:.0f} s)" if time > longest else ""
        print(
            f"{detector}: A {result['amplitude']:.4f} +- {result['amplitude_error']:.4f} DN/s, alpha "
            f"{result['alpha']:.2f} +- {result['alpha_error']:.2f}, reduced chi-square {result['reduced_chisq']:.2f}; "
            f"time {time:.0f} s at {args.threshold:g} DN/s{note} -> {out}"
        )
    print(f"wrote {written} files")


if __name__ == "__main__":
    main()

"""Write the persistence time of every measured detector into the package, one file per detector.

Reads the ``persistence_<detector>.ecsv`` tables that ``scripts/measure_persistence.py`` writes, fits each
detector's persistence against the delay from the end of the earlier exposure to the middle of the later one, and
writes ``src/snowblind/data/persistence_<detector>.asdf``.  ``PersistenceFlagStep`` uses the detector's ``time`` when
it is given no time, and 2500 s for a detector without a file.

The fit follows the typical saturated star of the detector.  Persistence varies across a detector (with its quantum
efficiency, as on the similar Euclid NISP detectors) and with each pixel's history of illumination (how long and how
often it saturated), which the typical star does not show, so ``time`` is chosen long: the delay at which a pixel
persisting ``--margin`` times more than the typical star falls to ``--threshold``.  The file also holds
``typical_time``, the delay at which the typical star itself falls to the threshold, for those who would rather flag
less and pass it, or a time between the two, as ``time``.

The persistence of one pair of exposures (source, target) and one class of source pixels is the table's
``excess_groups - null_groups``.  Only the classes of saturated stars are used (saturated with more than 60000 DN of
signal, or with no rate), not ``sat 0k-60k``: pixels flagged SATURATED with little signal (cosmic rays and snowballs
that saturated partway through the ramp, and stars just at the saturation level) persist less than the stars at the
same delay, so the stars set the time.  Pairs whose pointing moved by less than ``--min_shift`` pixels are left out,
since a star of the source may then still be on the same pixels.

The rows are pooled in bins of delay by their median: a few rows have a null far off (a moved group of the source
landed on a star of the target), which the median ignores and a mean weighted by the rows' errors does not.  The
error of a bin's median comes from resampling the source exposures with replacement, since the rows of one source
share its stars: half the range from the 16th to the 84th percentile of the resampled medians.  When the table also
has at least ``MIN_ROWS`` rows measured with time running backwards (``direction`` "backward", where the source cannot
have left persistence), a bin's persistence is its forward median minus the median of all the backward rows, with the
errors added in quadrature: that takes out what does not decay (the pixels of higher response that the classes favor,
and the bias of the group statistic with the target's noise), and is taken over all delays because it does not decay
and the backward rows of short delays are few.  The fit, to the bins with at least ``MIN_ROWS`` rows, is

    persistence = A (delay / 1000 s)^-alpha

with weights of one over the error squared, the parameter errors scaled by the root of the reduced chi-square when it
is above one.  The fit reaches a level at ``1000 s (A / level)^(1 / alpha)``; the bins reach it where they first fall
below it (log-log interpolation between neighbors).  The decay flattens at long delays, where the fit falls faster
than the bins, so each time is the longer of the two: ``typical_time`` at the threshold, ``time`` at the threshold over
the margin.  A detector with fewer than ``MIN_BINS`` such bins, or
whose fit fails or gives A or alpha of zero or less, gets no file.  A time beyond the longest delay measured is written
with ``extrapolated`` set.  The bins are kept in the file for reading.

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
# delay bins [s], from the end of the source to the middle of the target
DELAY_EDGES = [0, 150, 212, 300, 425, 600, 850, 1200, 1700, 2400, 3400, 4800, 6800, 9600, 13600, 19200, 38400]
MIN_ROWS = 20  # fewest (pair, class) rows in a bin worth fitting
MIN_BINS = 4  # fewest bins worth a fit
N_RESAMPLE = 300  # resamplings of the source exposures for the error of a bin's median
ERROR_FLOOR = 1e-4  # DN/s, smallest error of a bin, against a bin of a few sources that happen to agree
MODEL_FORM = (
    "persistence = amplitude * (delay / 1000 s)**-alpha, delay from the end of the saturated exposure to the middle "
    "of the later one, persistence in DN/s"
)


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


def median_and_error(values, sources, rng):
    """Median of ``values`` and its error from resampling the source exposures (``sources``) with replacement.

    The error is half the range from the 16th to the 84th percentile of the resampled medians: their median absolute
    deviation comes out as zero when a bin has few source exposures.
    """
    _, inverse = np.unique(sources, return_inverse=True)
    members = [np.nonzero(inverse == k)[0] for k in range(inverse.max() + 1)]
    medians = []
    for _ in range(N_RESAMPLE):
        pick = rng.integers(0, len(members), len(members))
        medians.append(np.median(values[np.concatenate([members[k] for k in pick])]))
    low, high = np.percentile(medians, [16, 84])
    return float(np.median(values)), float((high - low) / 2)


def is_forward(rows):
    """Which rows of a measurement table were measured with time running forward (all, in a table without directions)."""
    if "direction" not in rows.colnames:
        return np.ones(len(rows), dtype=bool)
    return np.asarray(rows["direction"]).astype(str) == "forward"


def pooled(rows, rng):
    """The rows pooled per delay bin, less the backward level when the rows hold enough backward ones.

    Returns lists, per bin, of the edges, median delay, persistence, its error, forward median, rows and sources; and
    the backward level subtracted (``backward``, 0 without one), its error and its rows.
    """
    delay = np.asarray(rows["delay_mid"], dtype=float)
    excess = np.asarray(rows["excess_groups"] - rows["null_groups"], dtype=float)
    sources = np.asarray(rows["source"]).astype(str)
    forward = is_forward(rows)
    backward = (~forward).sum() >= MIN_ROWS
    back, back_error = median_and_error(excess[~forward], sources[~forward], rng) if backward else (0.0, 0.0)
    out = {"delay_lo": [], "delay_hi": [], "delay_median": [], "persistence": [], "error": [], "forward": [],
           "n_rows": [], "n_sources": [], "backward": back, "backward_error": back_error,
           "n_rows_backward": int((~forward).sum()) if backward else 0}
    for low, high in zip(DELAY_EDGES[:-1], DELAY_EDGES[1:]):
        ahead = forward & (delay >= low) & (delay < high)
        if ahead.sum() < MIN_ROWS:
            continue
        value, error = median_and_error(excess[ahead], sources[ahead], rng)
        out["delay_lo"].append(float(low))
        out["delay_hi"].append(float(high))
        out["delay_median"].append(float(np.median(delay[ahead])))
        out["persistence"].append(value - back)
        out["error"].append(max(float(np.hypot(error, back_error)), ERROR_FLOOR))
        out["forward"].append(value)
        out["n_rows"].append(int(ahead.sum()))
        out["n_sources"].append(len(set(sources[ahead])))
    return out


def fit(bins):
    """Fit the decay model to the pooled bins; returns a dict, or None when the fit fails or is not a decay."""
    delay = np.array(bins["delay_median"])
    persistence = np.array(bins["persistence"])
    error = np.array(bins["error"])
    try:
        (amplitude, alpha), covariance = curve_fit(
            model, delay, persistence, p0=(0.01, 1.0), sigma=error, absolute_sigma=True, maxfev=20000
        )
    except (RuntimeError, ValueError):
        return None
    if not (amplitude > 0 and alpha > 0):
        return None
    reduced_chisq = float(np.sum(((persistence - model(delay, amplitude, alpha)) / error) ** 2) / (len(delay) - 2))
    amplitude_error, alpha_error = np.sqrt(np.diag(covariance)) * np.sqrt(max(reduced_chisq, 1.0))
    return {
        "amplitude": float(amplitude),
        "amplitude_error": float(amplitude_error),
        "alpha": float(alpha),
        "alpha_error": float(alpha_error),
        "reduced_chisq": reduced_chisq,
        "n_bins": int(len(delay)),
    }


def crossing(bins, threshold):
    """Delay at which the pooled bins first fall below ``threshold``, interpolated in log-log; NaN if they do not."""
    delay = np.array(bins["delay_median"])
    persistence = np.array(bins["persistence"])
    below = np.nonzero(persistence < threshold)[0]
    if not len(below) or below[0] == 0:
        return np.nan
    k = below[0]
    x0, x1 = np.log(delay[k - 1]), np.log(delay[k])
    y0, y1 = np.log(persistence[k - 1]), np.log(max(persistence[k], 1e-6))
    return float(np.exp(x0 + (np.log(threshold) - y0) * (x1 - x0) / (y1 - y0)))


def delay_at(result, bins, level):
    """The delay at which the fit and the bins reach ``level``: the fit's, the bins', and the longer of the two."""
    from_fit = 1000.0 * (result["amplitude"] / level) ** (1.0 / result["alpha"])
    from_bins = crossing(bins, level)
    return from_fit, from_bins, float(np.nanmax([from_fit, from_bins]))


def main(argv=None):
    """Console entry: fit each detector's table in a measurement directory and write the package files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("measurement_dir", help="directory holding persistence_<detector>.ecsv tables")
    parser.add_argument("--dataset", required=True, help="what was measured, for the provenance")
    parser.add_argument("--threshold", type=float, default=0.005, help="persistence the time is set at [DN/s]")
    parser.add_argument(
        "--margin", type=float, default=2.0, help="how many times the typical star's persistence the time is set for"
    )
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
        # the same resamplings for every run on the same tables
        bins = pooled(rows, np.random.default_rng(0))
        result = fit(bins) if len(bins["persistence"]) >= MIN_BINS else None
        if result is None:
            print(f"{detector}: {len(bins['persistence'])} bins, no decay fitted; no file written (the step uses 2500 s)")
            continue
        typical_fit, typical_bins, typical_time = delay_at(result, bins, args.threshold)
        time_fit, time_bins, time = delay_at(result, bins, args.threshold / args.margin)
        longest = float(np.max(np.asarray(rows["delay_mid"], dtype=float)[is_forward(rows)]))
        exposures = sorted(str(name) for name in set(table["source"]) | set(table["target"]))
        entry = {
            "detector": detector,
            "time": float(time),
            "typical_time": float(typical_time),
            "margin": float(args.margin),
            "threshold": float(args.threshold),
            "extrapolated": bool(time > longest),
            "longest_delay": longest,
            "model": {"form": MODEL_FORM, **result},
            # the delays [s] of the fit and of the bins (NaN when they do not fall below it) behind each time
            "estimates": {
                "typical_time_fit": typical_fit, "typical_time_bins": typical_bins,
                "time_fit": time_fit, "time_bins": time_bins,
            },
            "measured": bins,
            "provenance": {
                "dataset": args.dataset,
                "n_exposures": len(exposures),
                "first_exposure": exposures[0],
                "last_exposure": exposures[-1],
                "classes": sorted(str(cls) for cls in set(rows["cls"])),
                "min_shift_pix": float(args.min_shift),
                "pooling": (
                    "median per delay bin less the median of the backward rows of all delays, errors from resampling "
                    "the source exposures added in quadrature" if bins["n_rows_backward"] else
                    "median per delay bin, forward only, error from resampling the source exposures"
                ),
                "measurement": "scripts/measure_persistence.py",
                "table": os.path.basename(path),
            },
            "description": (
                "Persistence time of the detector, used by PersistenceFlagStep when it is given no time: the delay "
                "from the end of a saturated exposure to the middle of a later one at which a pixel persisting margin "
                "times more than the detector's typical saturated star falls to the threshold; typical_time is the "
                "delay at which the typical star itself does; each the longer of the fit's and the measured bins'"
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
            f"{result['alpha']:.2f} +- {result['alpha_error']:.2f}, reduced chi-square {result['reduced_chisq']:.2f} "
            f"over {result['n_bins']} bins; typical star at {args.threshold:g} DN/s after {typical_time:.0f} s (fit "
            f"{typical_fit:.0f}, bins {typical_bins:.0f}), {args.margin:g} times that persistence after {time:.0f} s "
            f"(fit {time_fit:.0f}, bins {time_bins:.0f}){note} -> {out}"
        )
    print(f"wrote {written} files")


if __name__ == "__main__":
    main()

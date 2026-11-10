"""Write the snowball persistence times that SnowblindStep uses by default, one file per detector.

For each <detector>_decay.npz from fit_snowball_persistence.py, src/snowblind/data/
snowball_persistence_<detector>.asdf holds:
  core_time      the PERCENTILE-th percentile over the detector's snowballs of the time, in seconds from
                 the end of the integration of the hit, at which a core's saturated pixels fall to the
                 threshold, each core's from a power law with the detector's slope; multiplied by the
                 ratio of the measured to the fitted time at which the mean over cores falls to the
                 threshold, when that ratio is above 1 (the power law falls faster than the data late on)
  neighbor_time  the same for the pixels one pixel from the saturated ones
and the measurement behind them: the fits, the mean excess per time bin, every core's decay time and its
null, and the exposures.

A later dataset's decay files (--later, matched by detector) check that the persistence has not grown
since: the ratio of its mean excess over cores of the saturated pixels to the main dataset's, averaged
over the time bins both have (inverse-variance weighted), and when that ratio is above 1 both times are
lengthened as if every core's excess were higher by it, each by its own alpha (time * ratio**(1/alpha)).
Its neighbors are not compared: their mean excess falls under the noise before the bins a later dataset
of long gaps covers.  The check is saved under later_check.

usage:
  package_snowball_persistence.py --dataset TEXT decay1.npz [decay2.npz ...] [--outdir DIR]
                                  [--later decay1.npz ... --later-dataset TEXT]
The series files named in each decay file are looked for next to it, for their exposure names.
"""
import argparse
import datetime
from pathlib import Path

import asdf
import numpy as np

PERCENTILE = 99.0
SELECTIONS = {"saturated": "saturated", "neighbors": "1px"}
FORM = ("excess = amplitude * (t / 1000 s)**-alpha, t from the end of the integration of the hit (its last "
        "read) in s, excess in DN/s")


def decay_time(amplitude, alpha, threshold):
    return 1000.0 * (max(float(amplitude), 0.0) / threshold) ** (1.0 / float(alpha))


def exposures_of(decay_path, z):
    names = []
    for series in z["series"]:
        path = Path(decay_path).parent / Path(str(series)).name
        if path.is_file():
            names += [str(name) for name in np.load(path)["exposures"]]
    return names


def selection_tree(z, key, threshold):
    keep = z[f"keep_{key}"]
    fitted = decay_time(z[f"a_{key}"], z[f"alpha_{key}"], threshold)
    measured = float(z[f"measured_{key}"])
    scale = max(1.0, measured / fitted) if np.isfinite(measured) else 1.0
    times = z[f"tc_{key}"][keep]
    stack = z[f"stack_{key}"]
    null_stack = z[f"null_stack_{key}"]
    tree = {
        "time": float(np.percentile(times, PERCENTILE) * scale),
        "model": {
            "amplitude": float(z[f"a_{key}"]), "amplitude_error": float(z[f"a_err_{key}"]),
            "alpha": float(z[f"alpha_{key}"]), "alpha_error": float(z[f"alpha_err_{key}"]),
            "fitted_crossing": fitted, "measured_crossing": measured, "scale": scale,
        },
        "measured": {
            "time": z[f"mids_{key}"].astype(np.float32), "excess": stack[:, 0].astype(np.float32),
            "error": stack[:, 1].astype(np.float32), "n_cores": stack[:, 2].astype(np.int32),
            "null_excess": null_stack[:, 0].astype(np.float32), "null_error": null_stack[:, 1].astype(np.float32),
        },
        "per_core": {
            "decay_time": times.astype(np.float32),
            "null_decay_time": z[f"null_tc_{key}"][keep].astype(np.float32),
        },
        "n_cores": int(keep.sum()),
    }
    return tree


def later_ratio(z, later):
    """Weighted mean ratio of the later to the main mean excess of the saturated pixels, its error, the
    ratio and error per time bin, and the time bins used."""
    if not np.array_equal(z["time_edges"], later["time_edges"]):
        raise ValueError("the later decay file has other time bins")
    main, late = z["stack_saturated"], later["stack_saturated"]
    both = np.isfinite(main[:, 0]) & np.isfinite(late[:, 0]) & (main[:, 0] > 0)
    ratio = late[both, 0] / main[both, 0]
    error = np.abs(ratio) * np.hypot(late[both, 1] / late[both, 0], main[both, 1] / main[both, 0])
    weight = error**-2.0
    bins = np.stack([z["time_edges"][:-1][both], z["time_edges"][1:][both]], axis=1)
    return float(np.sum(weight * ratio) / np.sum(weight)), float(np.sum(weight) ** -0.5), ratio, error, bins


def later_check(z, later_path, later_dataset):
    later = np.load(later_path)
    mean, error, ratio, ratio_error, bins = later_ratio(z, later)
    exposures = exposures_of(later_path, later)
    return {
        "dataset": later_dataset,
        "exposures": exposures,
        "n_exposures": len(exposures),
        "ratio": mean,
        "ratio_error": error,
        "time_bins": bins.astype(np.float32),
        "bin_ratio": ratio.astype(np.float32),
        "bin_ratio_error": ratio_error.astype(np.float32),
        "time_scale": {name: max(1.0, mean) ** (1.0 / float(z[f"alpha_{key}"])) for name, key in SELECTIONS.items()},
        "baseline": f"integrations ending before the hit or starting more than {float(later['clean_after']):.0f} s "
                    f"after the end of its integration, at least {int(later['min_clean'])}",
        "cores": f"cores with data in their first {float(later['early_by']):.0f} s after the end of the "
                 "integration of the hit",
    }


def package(decay_path, dataset, outdir, later_path=None, later_dataset=None):
    z = np.load(decay_path)
    detector = str(z["detector"])
    threshold = float(z["threshold"])
    selections = {name: selection_tree(z, key, threshold) for name, key in SELECTIONS.items()}
    check = None
    if later_path is not None:
        check = later_check(z, later_path, later_dataset)
        for name, selection in selections.items():
            selection["time"] *= check["time_scale"][name]
    exposures = exposures_of(decay_path, z)
    tree = {
        "description": (
            "Snowball persistence times of the detector, used by SnowblindStep when it is given none: core_time, "
            f"the {PERCENTILE:g}th percentile over the detector's snowballs of the time in seconds from the end "
            "of the integration of the hit at which the excess rate of a core's saturated pixels falls to "
            "threshold (DN/s), each core's from a power law with the detector's alpha, scaled by the measured "
            "over the fitted time at which the mean over cores does when above 1; neighbor_time, the same for "
            "the pixels one pixel from the saturated ones; both lengthened where later_check, a later dataset, "
            "shows more excess"
        ),
        "detector": detector,
        "core_time": selections["saturated"]["time"],
        "neighbor_time": selections["neighbors"]["time"],
        "threshold": threshold,
        "percentile": PERCENTILE,
        "generated": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "model": {"form": FORM, "fit_from": float(z["fit_from"]), "fit_to": float(z["fit_to"]),
                  **{name: s["model"] for name, s in selections.items()}},
        "measured": {"time_edges": z["time_edges"].astype(np.float32),
                     **{name: s["measured"] for name, s in selections.items()},
                     "two_pixels": {"time": z["mids_2px"].astype(np.float32),
                                    "excess": z["stack_2px"][:, 0].astype(np.float32),
                                    "error": z["stack_2px"][:, 1].astype(np.float32)}},
        "per_core": {name: s["per_core"] for name, s in selections.items()},
        "provenance": {
            "dataset": dataset,
            "exposures": exposures,
            "n_exposures": len(exposures),
            "n_cores": {name: s["n_cores"] for name, s in selections.items()},
            "measurement": "scripts/measure_snowball_persistence.py",
            "fit": "scripts/fit_snowball_persistence.py",
            "saturation": "SaturationStep with n_pix_grow_sat=1; the saturated pixels are a core's pixels 2 px "
                          "or more from its edge, the neighbors its edge pixels",
            "baseline": f"integrations ending before the hit or starting more than {float(z['clean_after']):.0f} s "
                        "after the end of its integration",
            "cores": "cores with data in their first 1000 s after the end of the integration of the hit",
        },
    }
    if check is not None:
        tree["later_check"] = check
    out = Path(outdir) / f"snowball_persistence_{detector.lower()}.asdf"
    asdf.AsdfFile(tree).write_to(out)
    print(f"{detector:9s} core_time {tree['core_time']:7.0f} s  neighbor_time {tree['neighbor_time']:6.0f} s  "
          f"({selections['saturated']['n_cores']} cores, {len(exposures)} exposures) -> {out}")
    if check is not None:
        print(f"          later / main excess {check['ratio']:.2f} +- {check['ratio_error']:.2f} over "
              f"{len(check['bin_ratio'])} bins: times x{check['time_scale']['saturated']:.3f} (saturated), "
              f"x{check['time_scale']['neighbors']:.3f} (neighbors)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("decay", nargs="+")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--outdir", default=str(Path(__file__).resolve().parents[1] / "src" / "snowblind" / "data"))
    parser.add_argument("--later", nargs="+", default=[], help="decay files of a later dataset")
    parser.add_argument("--later-dataset", help="the later dataset, as --dataset")
    args = parser.parse_args()
    if args.later and not args.later_dataset:
        parser.error("--later needs --later-dataset")
    later = {str(np.load(path)["detector"]): path for path in args.later}
    for path in args.decay:
        package(path, args.dataset, args.outdir, later.get(str(np.load(path)["detector"])), args.later_dataset)

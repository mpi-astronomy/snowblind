"""White floor of the column-mean scatter on a NIRSpec rate, measured from the data, and the banding of single amplifiers.

Reads an ``exp_nirspec_rate_median.py`` output folder (``none_ols.npz``,
``stream_clean_half0_ols.npz`` and its scene mask) and the run folder
with the slit or slice footprints (``regions.npz``). The background
pixels are those outside the footprints and the scene mask.

* ``robust_sigma`` and ``std``: the pixel scatter the floor formula uses
  (MAD based) and the plain standard deviation of the same pixels, also
  after leaving out pixels beyond 5 robust sigma.
* ``measured_sigma``: the white noise per pixel measured from the
  difference, in each column, of the mean of the background pixels in the
  first and in the last 16 rows of every 32-row block; banding constant
  over 32 rows cancels, the even-odd row pattern too, the pixel noise does
  not. Each column's difference is scaled by sqrt(n1 n2 / (n1 + n2)) and the
  robust sigma is taken across the columns.
* for whole columns (2048 rows) and single-amplifier column segments (512
  rows): the column-mean scatter (robust sigma of the means), the floor
  from ``robust_sigma`` as the docs metric has it, the floor from
  ``measured_sigma``, and the scatter above the measured floor.

Usage: python scripts/check_banding_floor.py <rate_median_folder> <run_folder>
"""
import json
import os
import sys
import warnings

import numpy as np

from snowblind.metrics import robust_sigma

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_nirspec_field_panels import usable_pixels  # noqa: E402

BLOCK = 32
AMPLIFIER_ROWS = 512


def measured_sigma(rate, background):
    """White noise per pixel from the first-minus-last half-block differences, robust sigma across columns."""
    first = (np.arange(rate.shape[0]) % BLOCK) < BLOCK // 2
    a, b = background & first[:, None], background & ~first[:, None]
    na, nb = a.sum(axis=0), b.sum(axis=0)
    difference = np.nanmean(np.where(a, rate, np.nan), axis=0) - np.nanmean(np.where(b, rate, np.nan), axis=0)
    return robust_sigma(difference * np.sqrt(na * nb / np.maximum(na + nb, 1)))


def column_means(rate, background, rows):
    """Column-mean scatter and the median pixel count of columns cut into segments of ``rows`` rows."""
    means, counts = [], []
    for start in range(0, rate.shape[0], rows):
        cut = background[start:start + rows]
        means.append(np.nanmean(np.where(cut, rate[start:start + rows], np.nan), axis=0))
        counts.append(cut.sum(axis=0))
    means, counts = np.concatenate(means), np.concatenate(counts)
    return robust_sigma(means), float(np.median(counts[counts > 0]))


def main(out_dir, run_dir):
    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
    warnings.filterwarnings("ignore", "invalid value", RuntimeWarning)
    none = np.load(os.path.join(out_dir, "none_ols.npz"))
    regions = np.load(os.path.join(run_dir, "regions.npz"))["regions"]
    scene = np.load(os.path.join(out_dir, "stream_clean_half0_mask.npz"))["mask"]
    background = usable_pixels(none["rate"], none["dq"]) & ~regions & scene
    rates = {"none": none["rate"], "stream_clean": np.load(os.path.join(out_dir, "stream_clean_half0_ols.npz"))["rate"]}

    result = {}
    for name, rate in rates.items():
        values = rate[background]
        robust = robust_sigma(values)
        core = values[np.abs(values - np.median(values)) <= 5 * robust]
        measured = measured_sigma(rate, background)
        result[name] = {
            "robust_sigma": robust, "std": float(np.std(values)), "std_within_5_robust_sigma": float(np.std(core)),
            "measured_sigma": measured,
        }
        for label, rows in (("whole_column", rate.shape[0]), ("amplifier_segment", AMPLIFIER_ROWS)):
            scatter, count = column_means(rate, background, rows)
            floor = measured / np.sqrt(count)
            result[name][label] = {
                "column_mean_scatter": scatter, "pixels_per_column": count,
                "docs_floor": robust / np.sqrt(count), "measured_floor": floor,
                "above_measured_floor": float(np.sqrt(max(scatter**2 - floor**2, 0.0))),
            }
    with open(os.path.join(out_dir, "floor_check.json"), "w") as handle:
        json.dump(result, handle, indent=2)

    for name, r in result.items():
        print(f"{os.path.basename(out_dir)} {name}: pixel sigma robust {1e4 * r['robust_sigma']:.2f}, measured "
              f"{1e4 * r['measured_sigma']:.2f} (ratio {r['measured_sigma'] / r['robust_sigma']:.3f}); std "
              f"{1e4 * r['std']:.1f}, within 5 sigma {1e4 * r['std_within_5_robust_sigma']:.2f} [1e-4 DN/s]")
        for label in ("whole_column", "amplifier_segment"):
            c = r[label]
            print(f"    {label:17s} scatter {1e4 * c['column_mean_scatter']:6.2f}  floor docs {1e4 * c['docs_floor']:5.2f}"
                  f" measured {1e4 * c['measured_floor']:5.2f}  above measured floor {1e4 * c['above_measured_floor']:6.2f}"
                  f"  ({c['pixels_per_column']:.0f} pixels per column)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

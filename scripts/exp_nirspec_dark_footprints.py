"""Banding left inside NIRSpec slit or slice footprints laid over a dark, by the stream clean and by msaexp's column median.

The footprints of a MOS or IFU exposure (``regions.npz`` of an earlier
``exp_nirspec_variants.py`` or ``make_nirspec_field_panels.py`` run) stand in
for the spectra on a dark of the same detector. Each correction is fitted on
the dark outside them, with all of that background as on the science
exposure, and the banding is measured inside them, where nothing was fitted
and a dark has no signal. Measured outside them too, on the pixels the
corrections fitted, for comparison.

Corrections of the dark's ``_jump`` ramp (from an earlier run) and of its
uncleaned OLS_C rate:

* ``stream_clean``: the step with the settings it takes on NIRSpec
  spectroscopy (column offsets off; ``min_period`` 600 on IRS2 read
  patterns), the footprints left out of its background mask;
* ``msaexp_ifu_wcs`` and ``msaexp_ifu``: msaexp's IFU recipe on the rate
  (``exp_nirspec_rate_median.py``), fitted outside the footprints as they
  are and grown by 4 pixels.

Measures, over the usable pixels inside (or outside) the footprints and the
stream clean's scene mask, in DN/s: the column-mean scatter over whole
columns (the docs metric) and over single-amplifier segments of 512 rows
(an extraction aperture lies within one amplifier), each with the white
floor from the white noise measured on the same pixels
(``check_banding_floor.py``) and the scatter above that floor.

Usage: python scripts/exp_nirspec_dark_footprints.py <dark_run_folder> <label>:<footprint_run_folder>... --out <folder>
"""
import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np
import scipy.ndimage as nd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_banding_floor import column_means, measured_sigma  # noqa: E402
from exp_nirspec_rate_median import (  # noqa: E402
    SLICE_GROWTH, grizli_oneoverf, msaexp_ifu_mask, stream_clean_rate, uncleaned_rate,
)
from make_nirspec_field_panels import usable_pixels  # noqa: E402

CORRECTIONS = ["none", "msaexp_ifu", "msaexp_ifu_wcs", "stream_clean"]
# stream clean's NIRSpec spectroscopy setting, which a dark exposure type does not select
SPECTROSCOPY_OPTIONS = {"column_offsets": False}


def banding(rate, pixels):
    """Column-mean scatter, measured floor and scatter above it, over whole columns and amplifier segments."""
    sigma = measured_sigma(rate, pixels)
    result = {"measured_sigma": sigma}
    for label, rows in (("whole_column", rate.shape[0]), ("amplifier_segment", 512)):
        scatter, count = column_means(rate, pixels, rows)
        floor = sigma / np.sqrt(count)
        result[label] = {
            "column_mean_scatter": scatter, "pixels_per_column": count, "floor": floor,
            "above_floor": float(np.sqrt(max(scatter**2 - floor**2, 0.0))),
        }
    return result


def main():
    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
    warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
    warnings.filterwarnings("ignore", "invalid value", RuntimeWarning)
    warnings.filterwarnings("ignore", "divide by zero", RuntimeWarning)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dark_run", help="folder with the dark's _jump ramp")
    parser.add_argument("footprints", nargs="+", help="label:folder with the regions.npz of a science exposure")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    jump_path = glob.glob(os.path.join(args.dark_run, "*_jump.fits"))[0]
    none = uncleaned_rate(jump_path, args.out)
    usable = usable_pixels(none["rate"], none["dq"])
    metrics = {"dark": os.path.basename(jump_path)}
    for item in args.footprints:
        label, run = item.split(":", 1)
        footprints = np.load(os.path.join(run, "regions.npz"))["regions"]
        stream_clean, scene = stream_clean_rate(
            jump_path, os.path.dirname(os.path.abspath(jump_path)), args.out, footprints, label,
            options=SPECTROSCOPY_OPTIONS,
        )
        grown = nd.binary_dilation(footprints, iterations=SLICE_GROWTH)
        rates = {
            "none": none["rate"],
            "msaexp_ifu": grizli_oneoverf(none["rate"], none["err"], none["dq"], msaexp_ifu_mask(none, grown), 2048),
            "msaexp_ifu_wcs": grizli_oneoverf(
                none["rate"], none["err"], none["dq"], msaexp_ifu_mask(none, footprints), 2048
            ),
            "stream_clean": stream_clean,
        }
        inside = usable & scene & footprints
        outside = usable & scene & ~footprints
        metrics[label] = {
            "footprint_source": run, "inside_pixels": int(inside.sum()), "outside_pixels": int(outside.sum()),
        }
        for name in CORRECTIONS:
            metrics[label][name] = {"inside": banding(rates[name], inside), "outside": banding(rates[name], outside)}
        with open(os.path.join(args.out, "metrics.json"), "w") as handle:
            json.dump(metrics, handle, indent=2)

        print(f"{metrics['dark']} with the {label} footprints: scatter above the measured floor [1e-4 DN/s]")
        for name in CORRECTIONS:
            m = metrics[label][name]
            print(
                f"  {name:16s} inside: whole column {1e4 * m['inside']['whole_column']['above_floor']:6.2f},"
                f" amplifier {1e4 * m['inside']['amplifier_segment']['above_floor']:6.2f}"
                f"   outside (fitted): whole column {1e4 * m['outside']['whole_column']['above_floor']:6.2f},"
                f" amplifier {1e4 * m['outside']['amplifier_segment']['above_floor']:6.2f}"
            )


if __name__ == "__main__":
    main()

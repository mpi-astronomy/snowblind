"""Variants of the stream clean on one NIRSpec spectroscopic exposure, measured as ``make_nirspec_field_panels.py`` does.

The Detector1 steps up to jump run once (the ``_jump`` ramp is kept in
the output folder and reused), then each variant of `StreamCleanStep`
runs on it and is fitted with OLS_C.  Variants (``--variants`` picks a
subset):

* ``default``: the step's defaults;
* ``notch``: the Wiener gain notched at the row period, as on WFSS
  exposures, against the slice wings that sit at the same clock in every
  readout row;
* ``dilate4`` and ``dilate8``: the scene mask grown by that many pixels;
* ``no_offsets``: column and even-odd offsets off;
* ``column_offsets``: column offsets on (the NIRSpec default is off);
* ``notch_dilate4``: both.

``metrics.json`` holds, per variant, the background banding metrics and
the change inside the slit or slice footprints, overall and binned by the
uncleaned signal.
"""
import argparse
import datetime
import json
import os
import sys
import warnings

import numpy as np
from jwst.ramp_fitting import RampFitStep
from stdatamodels.jwst import datamodels

from snowblind import StreamCleanStep
from snowblind.reset_calib import run_detector1_to_jump

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_nirspec_field_panels import (  # noqa: E402
    banding_metrics, change_metrics, save_rate, science_regions, usable_pixels,
)

BINS = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999, 1.0]


class NotchedStreamCleanStep(StreamCleanStep):
    """The stream clean with the Wiener gain notched at the row period, whatever the exposure type."""

    class_alias = "stream_clean"

    def _clean_with_mask(self, result, background_mask, flat):
        self._notch_period = float(self._layout.row_clocks)
        return super()._clean_with_mask(result, background_mask, flat)


VARIANTS = {
    "default": (StreamCleanStep, {}),
    "notch": (NotchedStreamCleanStep, {}),
    "dilate4": (StreamCleanStep, {"mask_dilation": 4}),
    "dilate8": (StreamCleanStep, {"mask_dilation": 8}),
    "no_offsets": (StreamCleanStep, {"column_offsets": False, "even_odd_columns": False}),
    "column_offsets": (StreamCleanStep, {"column_offsets": True}),
    "notch_dilate4": (NotchedStreamCleanStep, {"mask_dilation": 4}),
}


def binned_change(rate, reference, inside):
    """Median change in bins of the uncleaned signal's quantiles inside the regions."""
    signal = reference[inside]
    edges = np.quantile(signal[np.isfinite(signal)], BINS)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = inside & (reference >= lo) & (reference < hi)
        change = (rate - reference)[sel]
        rows.append({
            "low": float(lo), "high": float(hi), "n": int(sel.sum()),
            "signal_median": float(np.nanmedian(reference[sel])),
            "change_median": float(np.nanmedian(change)),
        })
    return rows


def main():
    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file")
    parser.add_argument("--out", default=None)
    parser.add_argument("--ngroups", type=int, default=None)
    parser.add_argument(
        "--snowblind", action="store_true", help="run SnowblindStep after the jump step, as snowblind_detector1 does"
    )
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument(
        "--save-cleaned", action="store_true", help="also keep each variant's cleaned ramp as <variant>_cleaned.fits"
    )
    args = parser.parse_args()
    out_dir = args.out or os.path.join("results", f"{datetime.date.today().isoformat()}_nirspec_variants")
    os.makedirs(out_dir, exist_ok=True)
    input_dir = os.path.dirname(os.path.abspath(args.file))
    jump_path = os.path.join(out_dir, os.path.basename(args.file).replace("_uncal.fits", "_jump.fits"))
    if not os.path.exists(jump_path):
        jump = run_detector1_to_jump(
            args.file, max_groups=args.ngroups, max_ints=1, refpix_algorithm="median", snowblind=args.snowblind
        )
        jump.save(jump_path)
        jump.close()
    regions_path = os.path.join(out_dir, "regions.npz")
    none_path = os.path.join(out_dir, "none_ols.npz")
    with datamodels.open(jump_path) as jump:
        if not os.path.exists(regions_path):
            np.savez_compressed(regions_path, regions=science_regions(jump, input_dir))
        if not os.path.exists(none_path):
            rate, _ = RampFitStep.call(jump, algorithm="OLS_C")
            save_rate(none_path, rate)
            rate.close()
    regions = np.load(regions_path)["regions"]
    reference = np.load(none_path)
    usable = usable_pixels(reference["rate"], reference["dq"])

    metrics_path = os.path.join(out_dir, "metrics.json")
    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
    for name in args.variants.split(","):
        step_class, options = VARIANTS[name]
        rate_path = os.path.join(out_dir, f"{name}_ols.npz")
        mask_path = os.path.join(out_dir, f"{name}_mask.npz")
        if not os.path.exists(rate_path):
            step = step_class(output_dir=out_dir, save_mask=True, **options)
            step.input_dir = input_dir
            cleaned = step.run(jump_path)
            rate, _ = RampFitStep.call(cleaned, algorithm="OLS_C")
            save_rate(rate_path, rate)
            if args.save_cleaned:
                cleaned.save(os.path.join(out_dir, f"{name}_cleaned.fits"))
            cleaned.close()
            rate.close()
            saved_mask = os.path.join(out_dir, os.path.basename(jump_path).replace("_jump.fits", "_mask.fits"))
            if os.path.exists(saved_mask):
                with datamodels.open(saved_mask) as mask_model:
                    np.savez_compressed(mask_path, mask=np.asarray(mask_model.data, dtype=bool))
                os.remove(saved_mask)
        rate = np.load(rate_path)["rate"]
        background = usable & ~regions
        if os.path.exists(mask_path):
            background &= np.load(mask_path)["mask"]
        metrics[name] = {
            "options": options, "background_fraction": float(background.mean()),
            "background": banding_metrics(rate, background),
            "change_in_regions": change_metrics(rate, reference["rate"], usable & regions),
            "change_outside_regions": change_metrics(rate, reference["rate"], background),
            "binned_change_in_regions": binned_change(rate, reference["rate"], usable & regions),
        }
        with open(metrics_path, "w") as handle:
            json.dump(metrics, handle, indent=2)
        b, r = metrics[name]["background"], metrics[name]["change_in_regions"]
        top = metrics[name]["binned_change_in_regions"][-1]
        print(
            f"{name:14s} bkg {100 * metrics[name]['background_fraction']:.1f}% columns {b['column_mean_scatter']:.5f} "
            f"(floor {b['column_mean_white_floor']:.5f}) rows {b['row_mean_scatter']:.5f} | in regions mean {r['mean']:+.6f} "
            f"slope {r['slope_vs_signal']:+.5f} | brightest 0.1%: {top['change_median']:+.5f} on {top['signal_median']:.3f} "
            f"({100 * top['change_median'] / top['signal_median']:+.2f}%)"
        )
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

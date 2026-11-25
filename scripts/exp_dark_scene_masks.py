"""Banding left under the masked sources or traces of a NIRCam or NIRISS field laid over a dark, stream clean against grizli.

The scene mask of an earlier stream clean run on a field (its saved
``stream_clean_mask.fits``, True on background) stands in for the sources or
WFSS traces on a dark of the same detector. Each correction is fitted on the
dark outside the masked area, as it would be on the field, and the banding is
measured inside it, where nothing was fitted and a dark has no signal; and
outside it, on the pixels the corrections fitted, for comparison.

The dark's first integration and first ``--groups`` groups are copied to a
small uncal file through a memory map (a full dark is too large to open) and
run through Detector1 up to jump with SIRS refpix, then fitted with OLS_C.

Corrections:

* ``stream_clean``: the step with the settings the field's exposure type gets,
  which a dark does not select: imaging the defaults; GR150C and GRISMC no
  column or even-odd offsets and the Wiener gain notched at the readout row
  period; GR150R no stream filter. The background is the step's default mean
  (a dark has no sky).
* ``grizli``: grizli's ``exposure_oneoverf_correction`` (grizli 754cb05) with
  its defaults (``deg_pix=64``, thresholds 5, 4, 3, source mask eroded on
  imaging and not on grisms) along the instrument's banding axis, the masked
  area as its manual mask (``exp_nirspec_rate_median.grizli_oneoverf``).

Measures along the banding axis (row means on NIRCam, column means on
NIRISS), over the usable pixels inside (or outside) the masked area and the
dark's own scene mask, in DN/s: the mean scatter over whole rows or columns
(the docs metric) and over single-amplifier segments of 512 pixels, each with
the white floor measured on the same pixels and the scatter above it
(``exp_nirspec_dark_footprints.banding``).

Usage: python scripts/exp_dark_scene_masks.py <dark_uncal> <label>:<kind>:<mask.fits>... --out <folder>
with kind one of image, gr150c, gr150r, grismc.
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
from astropy.io import fits
from jwst.ramp_fitting import RampFitStep
from stdatamodels.jwst import datamodels

from snowblind.reset_calib import run_detector1_to_jump

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_nirspec_dark_footprints import banding  # noqa: E402
from exp_nirspec_rate_median import HeldOutStreamCleanStep, grizli_oneoverf, uncleaned_rate  # noqa: E402
from make_nirspec_field_panels import save_rate, usable_pixels  # noqa: E402
from measure_reset_offset_per_integration import _write_integration  # noqa: E402

CORRECTIONS = ["none", "grizli", "stream_clean"]
# stream clean options per kind of field, and whether the Wiener gain is notched at the readout row period
STREAM_CLEAN_SETTINGS = {
    "image": ({}, False),
    "gr150c": ({"column_offsets": False, "even_odd_columns": False}, True),
    "grismc": ({"column_offsets": False, "even_odd_columns": False}, True),
    "gr150r": ({"fit_method": "none"}, True),
}
# grizli erodes its source mask on imaging only
GRIZLI_ERODE = {"image": True, "gr150c": False, "grismc": False, "gr150r": False}
GRIZLI_DEG_PIX = 64


class OverlayStreamCleanStep(HeldOutStreamCleanStep):
    """The held-out stream clean, with the readout-row notch a WFSS exposure gets when ``notch`` is set."""

    class_alias = "stream_clean"

    def _clean_with_mask(self, result, background_mask, flat):
        if self.notch:
            self._notch_period = float(self._layout.row_clocks)
        return super()._clean_with_mask(result, background_mask, flat)


def dark_jump(uncal, groups, out_dir):
    """Path of the dark's ``_jump`` ramp (first integration, first ``groups`` groups), made once."""
    jump_path = os.path.join(out_dir, os.path.basename(uncal).replace("_uncal.fits", "_jump.fits"))
    if not os.path.exists(jump_path):
        cut_path = os.path.join(out_dir, os.path.basename(uncal))
        _write_integration(uncal, 0, groups, cut_path)
        jump = run_detector1_to_jump(cut_path, max_groups=groups, max_ints=1)
        jump.save(jump_path)
        jump.close()
        os.remove(cut_path)
    return jump_path


def stream_clean_rate(jump_path, out_dir, label, kind, sources):
    """Rate and scene mask of the stream clean fitted outside ``sources``, cached as npz files."""
    rate_path = os.path.join(out_dir, f"stream_clean_{label}_ols.npz")
    mask_path = os.path.join(out_dir, f"stream_clean_{label}_mask.npz")
    if not os.path.exists(rate_path):
        options, notch = STREAM_CLEAN_SETTINGS[kind]
        step = OverlayStreamCleanStep(output_dir=out_dir, save_mask=True, **options)
        step.excluded = sources
        step.notch = notch
        cleaned = step.run(jump_path)
        rate, _ = RampFitStep.call(cleaned, algorithm="OLS_C")
        save_rate(rate_path, rate)
        cleaned.close()
        rate.close()
        saved_mask = jump_path.replace("_jump.fits", "_mask.fits")
        with datamodels.open(saved_mask) as mask_model:
            np.savez_compressed(mask_path, mask=np.asarray(mask_model.data, dtype=bool))
        os.remove(saved_mask)
    return np.load(rate_path)["rate"], np.load(mask_path)["mask"]


def main():
    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
    warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
    warnings.filterwarnings("ignore", "invalid value", RuntimeWarning)
    warnings.filterwarnings("ignore", "divide by zero", RuntimeWarning)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("uncal", help="the dark's _uncal file")
    parser.add_argument("masks", nargs="+", help="label:kind:path of a field's stream_clean_mask.fits")
    parser.add_argument("--groups", type=int, default=20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    jump_path = dark_jump(args.uncal, args.groups, args.out)
    # the banding runs along the fast axis: rows on NIRCam (slow axis 2), columns on NIRISS (slow axis 1)
    transpose = abs(int(fits.getheader(jump_path)["SLOWAXIS"])) == 2
    none = uncleaned_rate(jump_path, args.out)
    usable = usable_pixels(none["rate"], none["dq"])

    def along_axis(image):
        return image.T if transpose else image

    metrics = {"dark": os.path.basename(args.uncal), "groups": args.groups, "banding_axis": "rows" if transpose else "columns"}
    for item in args.masks:
        label, kind, mask_file = item.split(":", 2)
        with fits.open(mask_file) as hdul:
            field_background = np.asarray(hdul["SCI"].data, dtype=bool)
        sources = ~(field_background[0] if field_background.ndim == 3 else field_background)
        stream_clean, scene = stream_clean_rate(jump_path, args.out, label, kind, sources)
        grizli = along_axis(grizli_oneoverf(
            along_axis(none["rate"]), along_axis(none["err"]), along_axis(none["dq"]), along_axis(~sources),
            GRIZLI_DEG_PIX, erode=GRIZLI_ERODE[kind],
        ))
        rates = {"none": none["rate"], "grizli": grizli, "stream_clean": stream_clean}
        inside = along_axis(usable & scene & sources)
        outside = along_axis(usable & scene & ~sources)
        metrics[label] = {
            "kind": kind, "mask": mask_file, "inside_pixels": int(inside.sum()), "outside_pixels": int(outside.sum()),
        }
        for name in CORRECTIONS:
            rate = along_axis(rates[name])
            metrics[label][name] = {"inside": banding(rate, inside), "outside": banding(rate, outside)}
        with open(os.path.join(args.out, "metrics.json"), "w") as handle:
            json.dump(metrics, handle, indent=2)

        print(f"{metrics['dark']} with the {label} mask: share of the banding above the floor removed")
        base = metrics[label]["none"]
        for name in CORRECTIONS[1:]:
            cells = []
            for where in ("inside", "outside"):
                for segment in ("whole_column", "amplifier_segment"):
                    left = metrics[label][name][where][segment]["above_floor"]
                    cells.append(f"{100 * (1 - left / base[where][segment]['above_floor']):4.0f}%")
            print(
                f"  {name:13s} inside: whole {cells[0]}, amplifier {cells[1]}"
                f"   outside (fitted): whole {cells[2]}, amplifier {cells[3]}"
            )


if __name__ == "__main__":
    main()

"""Rate images of a science field before and after cleaning (field counterpart of ``make_rate_image_panels.py``).

Four chains on one ``_uncal`` exposure, each run in memory from the
Detector1 steps up to jump with SIRS reference-pixel correction and the
CRDS step parameters:

1. no cleaning, OLS_C (the pipeline default);
2. the jwst ``clean_flicker_noise`` step as it was run for program 6154
   (median method, not by channel, flat applied before fitting, science
   regions masked, the step's default background), OLS_C;
3. snowblind ``stream_clean`` (``--background`` selects its background
   method; a scalar median matches the jwst run on a flat-sky field), OLS_C;
4. the same ``stream_clean`` (its first read cleaned against the reset
   prior of ``--calib``), then snowblind
   ``reset_rampfit`` with the detector's reset calibration (per-channel
   timing, pedestal mean as the prior).

The rate images, the stream_clean background mask and a ``panels.json``
are written to ``results/<date>_field_panels/<detector>/`` and plotted by
``scripts/make_figures.py``.
"""
import argparse
import datetime
import glob
import json
import os

import numpy as np
from jwst.clean_flicker_noise import CleanFlickerNoiseStep
from jwst.ramp_fitting import RampFitStep

from snowblind import ResetRampFitStep, StreamCleanStep
from snowblind.reset_calib import run_detector1_to_jump
from snowblind.stream_clean import reference_in_raw_units, reference_scale


def save(out_dir, name, model):
    np.savez_compressed(os.path.join(out_dir, f"{name}.npz"), rate=model.data.astype(np.float32), dq=model.dq)


def parse_jump_params(text):
    """Turn ``key=value,key=value`` into a dict with numbers and booleans converted."""
    if not text:
        return None
    out = {}
    for item in text.split(","):
        key, value = item.split("=", 1)
        value = value.strip()
        if value.lower() in ("true", "false"):
            out[key.strip()] = value.lower() == "true"
        else:
            try:
                out[key.strip()] = int(value)
            except ValueError:
                out[key.strip()] = float(value)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file")
    parser.add_argument("--calib", required=True, help="reset calibration ASDF for this detector")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--background", default="median",
        help="comma-separated background settings for stream_clean: none, median, crds_background, box32, box256, ... "
        "(boxN is the fitted model with an N-pixel box); the first one also gets the reset fit and the "
        "unsuffixed sirs_stream_clean_* files the figures read",
    )
    parser.add_argument(
        "--apply-flat", action="store_true",
        help="stream_clean divides by the flat before fitting and multiplies back after; needed on sky fields, "
        "where the background is the sky times the detector response",
    )
    parser.add_argument("--snowblind", action="store_true", help="run SnowblindStep after the jump step")
    parser.add_argument(
        "--jump", default=None,
        help="jump step parameters overriding the CRDS reference, as key=value pairs separated by commas, "
        "e.g. min_jump_area=8,expand_factor=2.0",
    )
    args = parser.parse_args()
    jump_params = parse_jump_params(args.jump)

    out_root = args.out or os.path.join("results", f"{datetime.date.today().isoformat()}_field_panels")
    sirs = run_detector1_to_jump(args.file, refpix_algorithm="sirs", snowblind=args.snowblind, jump_params=jump_params)
    detector = sirs.meta.instrument.detector
    out_dir = os.path.join(out_root, detector)
    os.makedirs(out_dir, exist_ok=True)

    rate, _ = RampFitStep.call(sirs, algorithm="OLS_C")
    save(out_dir, "sirs_ols", rate)
    rate.close()

    jwst_clean = CleanFlickerNoiseStep.call(
        sirs, skip=False, fit_method="median", fit_by_channel=False, apply_flat_field=True, mask_science_regions=True
    )
    rate, _ = RampFitStep.call(jwst_clean, algorithm="OLS_C")
    save(out_dir, "sirs_jwst_usage_ols", rate)
    jwst_clean.close()
    rate.close()

    settings = [name.strip() for name in args.background.split(",")]
    for index, name in enumerate(settings):
        if name.lower() == "none":
            kwargs = {"background_method": None}
        elif name.startswith("box"):
            size = int(name[3:])
            kwargs = {"background_method": "integration_model", "background_box_size": (size, size)}
        else:
            kwargs = {"background_method": name}
        cleaned = StreamCleanStep.call(
            sirs, save_mask=index == 0, output_dir=out_dir, reset_calib=args.calib, apply_flat_field=args.apply_flat,
            **kwargs,
        )
        if index == 0:
            masks = sorted(glob.glob(os.path.join(out_dir, "*_mask.fits")))
            if masks:
                os.replace(masks[-1], os.path.join(out_dir, "stream_clean_mask.fits"))
        rate, _ = RampFitStep.call(cleaned, algorithm="OLS_C")
        save(out_dir, f"sirs_stream_clean_{name}_ols", rate)
        if index == 0:
            save(out_dir, "sirs_stream_clean_ols", rate)
        rate.close()
        if index == 0:
            rate, _ = ResetRampFitStep.call(cleaned, reset_calib=args.calib, use_pedestal_mean=True, output_dir=out_dir)
            save(out_dir, "sirs_stream_clean_reset", rate)
            rate.close()
        cleaned.close()

    background_reference, background_scale = None, None
    if "crds_background" in settings:
        # the dispersed background the step scaled out, for the figures to subtract before display and statistics
        import crds
        from stdatamodels.jwst import datamodels as sdm

        refs = crds.getreferences(sirs.get_crds_parameters(), reftypes=["bkg", "flat"], observatory="jwst")
        bkg_path = refs["bkg"]
        with sdm.open(bkg_path) as bkg_model:
            bkg = np.array(bkg_model.data, dtype=np.float64)
        with sdm.FlatModel(refs["flat"]) as flat_model:
            flat = np.array(flat_model.data, dtype=np.float64)
            flat[(flat_model.dq & 1) > 0] = np.nan
        # the step's convention: a flat-fielded reference (NIRCam) goes into raw units before scaling
        bkg, carries_flat = reference_in_raw_units(bkg, flat)
        with sdm.open(os.path.join(out_dir, "stream_clean_mask.fits")) as mask_model:
            mask = mask_model.data.astype(bool)
        rate_sirs = np.load(os.path.join(out_dir, "sirs_ols.npz"))["rate"].astype(np.float64)
        background_scale = reference_scale(rate_sirs, bkg, mask)
        background_reference = os.path.basename(bkg_path)
        np.savez_compressed(
            os.path.join(out_dir, "scaled_background.npz"), scene=(background_scale * bkg).astype(np.float32),
            scale=background_scale, reference=background_reference, reference_carries_flat=carries_flat,
        )

    with open(os.path.join(out_dir, "panels.json"), "w") as handle:
        json.dump(
            {
                "background_reference": background_reference, "background_scale": background_scale,
                "file": os.path.basename(args.file), "detector": detector, "calib": args.calib,
                "ngroups": int(sirs.data.shape[1]), "filter": sirs.meta.instrument.filter,
                "readpatt": sirs.meta.exposure.readpatt,
                "jwst_step": "fit_method=median, fit_by_channel=False, apply_flat_field=True, mask_science_regions=True",
                "stream_clean_backgrounds": settings, "stream_clean_apply_flat": args.apply_flat,
            },
            handle,
        )
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

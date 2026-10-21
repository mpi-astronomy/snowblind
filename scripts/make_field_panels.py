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


def save(out_dir, name, model):
    np.savez_compressed(os.path.join(out_dir, f"{name}.npz"), rate=model.data.astype(np.float32), dq=model.dq)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file")
    parser.add_argument("--calib", required=True, help="reset calibration ASDF for this detector")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--background", default="integration_model",
        help="background_method for stream_clean: integration_model (default), median (flat sky, as the jwst run), or none",
    )
    args = parser.parse_args()

    out_root = args.out or os.path.join("results", f"{datetime.date.today().isoformat()}_field_panels")
    sirs = run_detector1_to_jump(args.file, refpix_algorithm="sirs")
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

    background = None if args.background.lower() == "none" else args.background
    cleaned = StreamCleanStep.call(
        sirs, save_mask=True, output_dir=out_dir, background_method=background, reset_calib=args.calib
    )
    masks = sorted(glob.glob(os.path.join(out_dir, "*_mask.fits")))
    if masks:
        os.replace(masks[-1], os.path.join(out_dir, "stream_clean_mask.fits"))
    rate, _ = RampFitStep.call(cleaned, algorithm="OLS_C")
    save(out_dir, "sirs_stream_clean_ols", rate)
    rate.close()
    rate, _ = ResetRampFitStep.call(cleaned, reset_calib=args.calib, use_pedestal_mean=True, output_dir=out_dir)
    save(out_dir, "sirs_stream_clean_reset", rate)
    rate.close()

    with open(os.path.join(out_dir, "panels.json"), "w") as handle:
        json.dump(
            {
                "file": os.path.basename(args.file), "detector": detector, "calib": args.calib,
                "ngroups": int(sirs.data.shape[1]), "filter": sirs.meta.instrument.filter,
                "jwst_step": "fit_method=median, fit_by_channel=False, apply_flat_field=True, mask_science_regions=True",
                "stream_clean_background": args.background,
            },
            handle,
        )
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

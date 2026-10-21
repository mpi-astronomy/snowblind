"""Rate images for the before-and-after figure (counterpart of Figure 1 of the JWST 1/f page).

Four chains on the first integration and the first ``--ngroups`` groups of
a NIRCam dark, each run in memory:

1. default reference-pixel correction, no cleaning, OLS_C (the jwst default);
2. SIRS reference-pixel correction, no cleaning, OLS_C;
3. SIRS, then the jwst ``clean_flicker_noise`` step with its median method
   per channel, once with no background subtraction and once with its
   default per-difference background model, OLS_C;
4. SIRS, then snowblind ``stream_clean`` with its defaults, fitted both
   with OLS_C and with snowblind ``reset_rampfit`` and the detector's
   reset calibration.

The rate images are written to ``results/<date>_rate_images/<detector>/``
and plotted by ``scripts/make_figures.py``.
"""
import argparse
import datetime
import json
import os

import numpy as np
from jwst.clean_flicker_noise import CleanFlickerNoiseStep
from jwst.ramp_fitting import RampFitStep

from snowblind import ResetRampFitStep, StreamCleanStep
from snowblind.reset_calib import run_detector1_to_jump


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file")
    parser.add_argument("--calib", required=True, help="reset calibration ASDF for this detector")
    parser.add_argument("--ngroups", type=int, default=20)
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-column-offsets", action="store_true", help="pass column_offsets=False to stream_clean")
    parser.add_argument(
        "--background", default="integration_model",
        help="background_method for stream_clean: integration_model (default), median, or none (right for darks)",
    )
    args = parser.parse_args()

    out_root = args.out or os.path.join("results", f"{datetime.date.today().isoformat()}_rate_images")
    median = run_detector1_to_jump(args.file, max_groups=args.ngroups, max_ints=1, refpix_algorithm="median")
    detector = median.meta.instrument.detector
    out_dir = os.path.join(out_root, detector)
    os.makedirs(out_dir, exist_ok=True)

    rate, _ = RampFitStep.call(median, algorithm="OLS_C")
    np.savez_compressed(os.path.join(out_dir, "default_refpix_ols.npz"), rate=rate.data.astype(np.float32), dq=rate.dq)
    median.close()
    rate.close()

    sirs = run_detector1_to_jump(args.file, max_groups=args.ngroups, max_ints=1, refpix_algorithm="sirs")
    rate, _ = RampFitStep.call(sirs, algorithm="OLS_C")
    np.savez_compressed(os.path.join(out_dir, "sirs_ols.npz"), rate=rate.data.astype(np.float32), dq=rate.dq)
    rate.close()

    jwst_clean = CleanFlickerNoiseStep.call(
        sirs, skip=False, fit_method="median", fit_by_channel=True, background_method=None
    )
    rate, _ = RampFitStep.call(jwst_clean, algorithm="OLS_C")
    np.savez_compressed(os.path.join(out_dir, "sirs_jwst_median_ols.npz"), rate=rate.data.astype(np.float32), dq=rate.dq)
    jwst_clean.close()
    rate.close()

    jwst_clean = CleanFlickerNoiseStep.call(
        sirs, skip=False, fit_method="median", fit_by_channel=True, background_method="model"
    )
    rate, _ = RampFitStep.call(jwst_clean, algorithm="OLS_C")
    np.savez_compressed(
        os.path.join(out_dir, "sirs_jwst_median_model_ols.npz"), rate=rate.data.astype(np.float32), dq=rate.dq
    )
    jwst_clean.close()
    rate.close()

    background = None if args.background.lower() == "none" else args.background
    cleaned = StreamCleanStep.call(
        sirs, output_dir=out_dir, column_offsets=not args.no_column_offsets, background_method=background
    )
    rate, _ = ResetRampFitStep.call(cleaned, reset_calib=args.calib, use_pedestal_mean=True, output_dir=out_dir)
    np.savez_compressed(
        os.path.join(out_dir, "sirs_stream_clean_reset.npz"), rate=rate.data.astype(np.float32), dq=rate.dq
    )
    rate_ols, _ = RampFitStep.call(cleaned, algorithm="OLS_C")
    np.savez_compressed(
        os.path.join(out_dir, "sirs_stream_clean_ols.npz"), rate=rate_ols.data.astype(np.float32), dq=rate_ols.dq
    )

    with open(os.path.join(out_dir, "panels.json"), "w") as handle:
        json.dump(
            {
                "file": os.path.basename(args.file), "detector": detector, "ngroups": args.ngroups,
                "calib": args.calib, "column_offsets": not args.no_column_offsets, "background": args.background,
            },
            handle,
        )
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

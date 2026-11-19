"""Rate images of a NIRSpec dark before and after cleaning (NIRSpec counterpart of ``make_rate_image_panels.py``).

Four chains on the first integration and the first ``--ngroups`` groups of
a NIRSpec full-frame dark, each run in memory from the Detector1 steps up
to jump with the jwst reference-pixel correction (``--refpix``; the IRS2
correction runs by itself on IRS2 data) and the CRDS step parameters, then
fitted with OLS_C:

1. no cleaning (the pipeline default for NIRSpec);
2. the jwst ``clean_flicker_noise`` step with its median method per
   channel and no background;
3. the same step with its ``fft`` method (NSClean) and no background.  The
   step refuses the method on a dark, so the copy it cleans is labelled a
   fixed-slit exposure, which changes nothing else in the step;
4. snowblind ``stream_clean`` with its defaults (``--background`` and
   ``--no-column-offsets`` override two of them).

The rate images go to ``results/<date>_nirspec_dark/<detector>/`` as
``<chain>_ols.npz`` with ``metrics.json``: for every chain the pixel
scatter and the scatter of the science-frame row and column means over
the full frame and the central 512 by 512 pixels, with the white-noise
floor each would have from the pixel scatter alone.  On NIRSpec the fast
axis runs along science columns, so the 1/f banding shows in the column
means.
"""
import argparse
import datetime
import json
import os
import warnings

import numpy as np
from jwst.clean_flicker_noise import CleanFlickerNoiseStep
from jwst.ramp_fitting import RampFitStep

from snowblind import StreamCleanStep
from snowblind.metrics import robust_sigma
from snowblind.reset_calib import run_detector1_to_jump

BORDER = 4
CENTER = (slice(768, 1280), slice(768, 1280))


def save_rate(path, rate_model):
    """Write the rate and dq of a rate model as a compressed npz."""
    np.savez_compressed(path, rate=rate_model.data.astype(np.float32), dq=rate_model.dq)


def usable_pixels(reference_rate, reference_dq):
    """Good pixels of the uncleaned rate: finite, dq 0, away from the border and within 5 sigma of the median."""
    good = np.isfinite(reference_rate) & (reference_dq == 0)
    good[:BORDER] = good[-BORDER:] = False
    good[:, :BORDER] = good[:, -BORDER:] = False
    level = np.nanmedian(reference_rate[good])
    sigma = robust_sigma(reference_rate[good])
    return good & (np.abs(reference_rate - level) < 5 * sigma)


def banding_metrics(rate, usable):
    """Pixel scatter, row- and column-mean scatter and their white floors over a region, in DN/s."""
    cut = np.where(usable, rate, np.nan)
    rows = np.nanmean(cut, axis=1)
    columns = np.nanmean(cut, axis=0)
    pixel_sigma = robust_sigma(cut)
    rows_n = np.sum(usable, axis=1)
    columns_n = np.sum(usable, axis=0)
    return {
        "pixel_sigma": pixel_sigma,
        "row_mean_scatter": robust_sigma(rows),
        "row_mean_white_floor": float(pixel_sigma / np.sqrt(np.median(rows_n[rows_n > 0]))),
        "column_mean_scatter": robust_sigma(columns),
        "column_mean_white_floor": float(pixel_sigma / np.sqrt(np.median(columns_n[columns_n > 0]))),
    }


def measure(out_dir, chains):
    """Metrics of every chain's rate against the usable pixels of the uncleaned one."""
    reference = np.load(os.path.join(out_dir, "none_ols.npz"))
    usable = usable_pixels(reference["rate"], reference["dq"])
    metrics = {}
    for name in chains:
        rate = np.load(os.path.join(out_dir, f"{name}_ols.npz"))["rate"]
        metrics[name] = {
            "full_frame": banding_metrics(rate, usable),
            "central_512": banding_metrics(rate[CENTER], usable[CENTER]),
        }
    return metrics


def main():
    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file")
    parser.add_argument("--ngroups", type=int, default=20)
    parser.add_argument("--out", default=None)
    parser.add_argument("--refpix", default="median", help="refpix_algorithm of the jwst refpix step (median or sirs)")
    parser.add_argument("--background", default="mean", help="background_method for stream_clean (mean, median, none)")
    parser.add_argument("--no-column-offsets", action="store_true", help="pass column_offsets=False to stream_clean")
    parser.add_argument("--measure-only", action="store_true", help="only recompute metrics.json from the saved rates")
    args = parser.parse_args()

    out_root = args.out or os.path.join("results", f"{datetime.date.today().isoformat()}_nirspec_dark")
    chains = ["none", "jwst_median", "jwst_fft", "stream_clean"]
    if args.measure_only:
        out_dir = out_root
    else:
        jump = run_detector1_to_jump(args.file, max_groups=args.ngroups, max_ints=1, refpix_algorithm=args.refpix)
        detector = jump.meta.instrument.detector
        out_dir = os.path.join(out_root, detector)
        os.makedirs(out_dir, exist_ok=True)

        rate, _ = RampFitStep.call(jump, algorithm="OLS_C")
        save_rate(os.path.join(out_dir, "none_ols.npz"), rate)
        rate.close()

        cleaned = CleanFlickerNoiseStep.call(jump, skip=False, fit_method="median", fit_by_channel=True, background_method=None)
        rate, _ = RampFitStep.call(cleaned, algorithm="OLS_C")
        save_rate(os.path.join(out_dir, "jwst_median_ols.npz"), rate)
        cleaned.close()
        rate.close()

        as_slit = jump.copy()
        as_slit.meta.exposure.type = "NRS_FIXEDSLIT"
        cleaned = CleanFlickerNoiseStep.call(as_slit, skip=False, fit_method="fft", background_method=None)
        rate, _ = RampFitStep.call(cleaned, algorithm="OLS_C")
        save_rate(os.path.join(out_dir, "jwst_fft_ols.npz"), rate)
        as_slit.close()
        cleaned.close()
        rate.close()

        background = None if args.background.lower() == "none" else args.background
        cleaned = StreamCleanStep.call(
            jump, output_dir=out_dir, background_method=background, column_offsets=not args.no_column_offsets,
            save_spectra=True,
        )
        rate, _ = RampFitStep.call(cleaned, algorithm="OLS_C")
        save_rate(os.path.join(out_dir, "stream_clean_ols.npz"), rate)
        cleaned.close()
        rate.close()

        with open(os.path.join(out_dir, "panels.json"), "w") as handle:
            json.dump(
                {
                    "file": os.path.basename(args.file), "detector": detector, "ngroups": args.ngroups,
                    "refpix": args.refpix, "readpatt": str(jump.meta.exposure.readpatt),
                    "background": args.background, "column_offsets": not args.no_column_offsets,
                },
                handle, indent=2,
            )
        jump.close()

    metrics = measure(out_dir, chains)
    with open(os.path.join(out_dir, "metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2)
    for region in ("full_frame", "central_512"):
        print(region)
        print(f"{'chain':14s} {'pixel':>8s} {'rows':>8s} {'floor':>8s} {'columns':>8s} {'floor':>8s}")
        for name in chains:
            m = metrics[name][region]
            print(
                f"{name:14s} {m['pixel_sigma']:8.5f} {m['row_mean_scatter']:8.5f} {m['row_mean_white_floor']:8.5f} "
                f"{m['column_mean_scatter']:8.5f} {m['column_mean_white_floor']:8.5f}"
            )
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

"""Rate images of a NIRSpec MOS or IFU exposure before and after cleaning, and what the cleaning did to the spectra.

Four chains on the first integration of a full-frame NIRSpec science
exposure, each run in memory from the Detector1 steps up to jump with the
jwst reference-pixel correction and the CRDS step parameters, then fitted
with OLS_C:

1. no cleaning (the pipeline default for NIRSpec);
2. the jwst ``clean_flicker_noise`` step with its median method per
   channel, science regions masked, its default (median) background;
3. the same step with its ``fft`` method (NSClean), science regions masked;
4. snowblind ``stream_clean`` with its defaults, which on these exposures
   mask the slits or slices from the WCS as well as what the scene mask
   finds (``--background`` and ``--no-column-offsets`` override two of
   them).

The MSA metafile of a MOS exposure must sit next to the ``_uncal`` file.
The rate images go to ``results/<date>_nirspec_field/<detector>/`` as
``<chain>_ols.npz``, the stream_clean mask as ``*_mask.fits``, and
``metrics.json`` holds, for every chain:

* the scatter of the science-frame column means (the banding axis on
  NIRSpec) and row means over the background pixels, with the white floor
  from the pixel scatter;
* the change the cleaning made inside the slit or slice footprints
  (``cleaned - none``): its mean and scatter, and the slope of a straight
  line fit of that change against the uncleaned rate, which is the
  fraction of the spectra's signal the cleaning removed (negative) or
  added;
* the same change outside the footprints.
"""
import argparse
import datetime
import glob
import json
import os
import warnings

import numpy as np
from astropy.io import fits
from jwst.clean_flicker_noise import CleanFlickerNoiseStep
from jwst.clean_flicker_noise.clean_flicker_noise import create_mask
from jwst.ramp_fitting import RampFitStep

from snowblind import StreamCleanStep
from snowblind.metrics import robust_sigma
from snowblind.reset_calib import run_detector1_to_jump
from snowblind._jwst_imports import make_processed_rate_image

BORDER = 4


def save_rate(path, rate_model):
    """Write the rate and dq of a rate model as a compressed npz."""
    np.savez_compressed(path, rate=rate_model.data.astype(np.float32), dq=rate_model.dq)


def science_regions(jump, input_dir):
    """Boolean image, True inside the slit or slice footprints, the failed-open shutters and the fixed-slit area."""
    draft = make_processed_rate_image(jump, True, input_dir, jump.meta.exposure.type, True, None, "mean")
    background = np.asarray(create_mask(draft, mask_science_regions=True, n_sigma=0, single_mask=True))
    draft.close()
    return ~background


def usable_pixels(reference_rate, reference_dq):
    """Finite pixels with dq 0 away from the border."""
    good = np.isfinite(reference_rate) & (reference_dq == 0)
    good[:BORDER] = good[-BORDER:] = False
    good[:, :BORDER] = good[:, -BORDER:] = False
    return good


def banding_metrics(rate, usable):
    """Pixel scatter, row- and column-mean scatter and their white floors over the usable pixels, in DN/s."""
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


def change_metrics(rate, reference, where):
    """Mean, scatter and slope against the uncleaned rate of ``rate - reference`` over ``where``."""
    change = (rate - reference)[where]
    signal = reference[where]
    finite = np.isfinite(change) & np.isfinite(signal)
    change, signal = change[finite], signal[finite]
    if change.size < 10:
        return {"n": int(change.size)}
    # robust straight line: clip the 3 sigma outliers of the change once, then least squares
    scatter = robust_sigma(change)
    keep = np.abs(change - np.median(change)) <= 3 * scatter
    slope, intercept = np.polyfit(signal[keep], change[keep], 1)
    return {
        "n": int(change.size),
        "mean": float(np.mean(change)),
        "median": float(np.median(change)),
        "scatter": scatter,
        "slope_vs_signal": float(slope),
        "intercept": float(intercept),
        "signal_median": float(np.median(signal)),
    }


def measure(out_dir, chains, regions):
    """Metrics of every chain against the uncleaned chain."""
    reference = np.load(os.path.join(out_dir, "none_ols.npz"))
    usable = usable_pixels(reference["rate"], reference["dq"])
    mask_files = glob.glob(os.path.join(out_dir, "*_mask.fits"))
    background = usable & ~regions
    if mask_files:
        background &= fits.getdata(mask_files[0]).astype(bool)
    metrics = {
        "regions_fraction": float(regions.mean()),
        "background_fraction": float(background.mean()),
    }
    for name in chains:
        rate = np.load(os.path.join(out_dir, f"{name}_ols.npz"))["rate"]
        metrics[name] = {
            "background": banding_metrics(rate, background),
            "change_in_regions": change_metrics(rate, reference["rate"], usable & regions),
            "change_outside_regions": change_metrics(rate, reference["rate"], background),
        }
    return metrics


def main():
    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file")
    parser.add_argument(
        "--snowblind", action="store_true", help="run SnowblindStep after the jump step, as snowblind_detector1 does"
    )
    parser.add_argument("--ngroups", type=int, default=None, help="keep only the first groups (default all)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--refpix", default="median", help="refpix_algorithm of the jwst refpix step (median or sirs)")
    parser.add_argument("--background", default="mean", help="background_method for stream_clean")
    parser.add_argument("--no-column-offsets", action="store_true", help="pass column_offsets=False to stream_clean")
    parser.add_argument("--measure-only", action="store_true", help="only recompute metrics.json from the saved rates")
    args = parser.parse_args()

    out_root = args.out or os.path.join("results", f"{datetime.date.today().isoformat()}_nirspec_field")
    chains = ["none", "jwst_median", "jwst_fft", "stream_clean"]
    input_dir = os.path.dirname(os.path.abspath(args.file))
    if args.measure_only:
        out_dir = out_root
        regions = np.load(os.path.join(out_dir, "regions.npz"))["regions"]
    else:
        jump = run_detector1_to_jump(
            args.file, max_groups=args.ngroups, max_ints=1, refpix_algorithm=args.refpix, snowblind=args.snowblind
        )
        detector = jump.meta.instrument.detector
        out_dir = os.path.join(out_root, detector)
        os.makedirs(out_dir, exist_ok=True)
        regions = science_regions(jump, input_dir)
        np.savez_compressed(os.path.join(out_dir, "regions.npz"), regions=regions)

        rate, _ = RampFitStep.call(jump, algorithm="OLS_C")
        save_rate(os.path.join(out_dir, "none_ols.npz"), rate)
        rate.close()

        for name, fit_method in (("jwst_median", "median"), ("jwst_fft", "fft")):
            step = CleanFlickerNoiseStep(skip=False, fit_method=fit_method, fit_by_channel=True, mask_science_regions=True)
            step.input_dir = input_dir
            cleaned = step.run(jump)
            rate, _ = RampFitStep.call(cleaned, algorithm="OLS_C")
            save_rate(os.path.join(out_dir, f"{name}_ols.npz"), rate)
            cleaned.close()
            rate.close()

        background = None if args.background.lower() == "none" else args.background
        step = StreamCleanStep(
            output_dir=out_dir, background_method=background, column_offsets=not args.no_column_offsets,
            save_mask=True, save_spectra=True,
        )
        step.input_dir = input_dir
        cleaned = step.run(jump)
        rate, _ = RampFitStep.call(cleaned, algorithm="OLS_C")
        save_rate(os.path.join(out_dir, "stream_clean_ols.npz"), rate)
        cleaned.close()
        rate.close()

        with open(os.path.join(out_dir, "panels.json"), "w") as handle:
            json.dump(
                {
                    "file": os.path.basename(args.file), "detector": detector, "ngroups": int(jump.data.shape[1]),
                    "exposure_type": str(jump.meta.exposure.type), "readpatt": str(jump.meta.exposure.readpatt),
                    "grating": str(jump.meta.instrument.grating), "filter": str(jump.meta.instrument.filter),
                    "refpix": args.refpix, "background": args.background, "column_offsets": not args.no_column_offsets,
                },
                handle, indent=2,
            )
        jump.close()

    metrics = measure(out_dir, chains, regions)
    with open(os.path.join(out_dir, "metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2)
    print(
        f"regions {100 * metrics['regions_fraction']:.1f} percent, "
        f"background {100 * metrics['background_fraction']:.1f} percent"
    )
    print(
        f"{'chain':14s} {'pixel':>8s} {'columns':>8s} {'floor':>8s} {'rows':>8s} {'floor':>8s} | "
        f"in regions: {'mean':>8s} {'slope':>8s} | outside: {'mean':>8s}"
    )
    nan = float("nan")
    for name in chains:
        b = metrics[name]["background"]
        r = metrics[name]["change_in_regions"]
        o = metrics[name]["change_outside_regions"]
        print(
            f"{name:14s} {b['pixel_sigma']:8.5f} {b['column_mean_scatter']:8.5f} {b['column_mean_white_floor']:8.5f} "
            f"{b['row_mean_scatter']:8.5f} {b['row_mean_white_floor']:8.5f} | "
            f"{r.get('mean', nan):8.5f} {r.get('slope_vs_signal', nan):8.5f} | {o.get('mean', nan):8.5f}"
        )
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

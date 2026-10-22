"""Experiment C.3.1: six cleaning variants on a dark, metrics 1 to 3 (plan Part A).

Variants, all run in memory on the first integration and the first
``--ngroups`` groups of an ``_uncal`` dark:

1. ``median_refpix``: Detector1 to jump with the default refpix, no cleaning.
2. ``sirs``: Detector1 to jump with SIRS refpix, no cleaning.
3. ``sirs_jwst_median``: SIRS, then the jwst clean_flicker_noise step with
   its median method per channel.
4. ``sirs_row_median``: SIRS, then StreamCleanStep with ``row_median``.
5. ``sirs_boxcar``: SIRS, then StreamCleanStep with ``boxcar``.
6. ``sirs_wiener``: SIRS, then StreamCleanStep with ``wiener``.
7. ``*_box32``, ``*_box128``, ``*_box256``: StreamCleanStep defaults
   (``background_method='integration_model'``, one background model per
   integration, no per-difference clipping, so the cleaning is the same
   linear operation on every difference and no estimation noise accumulates
   along the ramp) with the given background box size; ``*_clipped`` is the
   jwst-like configuration with a per-difference model background and
   2-sigma clipping; ``sirs_wiener_median_bkg`` uses a scalar median
   background.

Each variant is ramp-fitted with OLS_C.  Metrics: readout-ordered channel
spectra of the group differences, power spectrum of the per-channel row
means of the rate image, and aperture scatter versus aperture size.  The
background mask for the metrics is common to all variants and comes from
the ``sirs`` rate image.  Output: ``summary.json`` and one ``.npz`` of
arrays per variant in ``results/<date>_darks_part_a/<detector>/``.
"""
import argparse
import datetime
import json
import os
import time

import numpy as np
from jwst.clean_flicker_noise import CleanFlickerNoiseStep
from jwst.ramp_fitting import RampFitStep
from stdatamodels.jwst.datamodels import dqflags

from snowblind import StreamCleanStep, readout
from snowblind.metrics import aperture_scatter, channel_psd, robust_sigma, row_banding_power
from snowblind.reset_calib import run_detector1_to_jump

APERTURES = [1, 3, 5, 9, 17, 33, 65, 101]
MAX_LAG = 2**16
DO_NOT_USE = dqflags.pixel["DO_NOT_USE"]


def metric_mask(rate_model):
    """Background mask for the metrics: finite, not DO_NOT_USE, within 5 robust sigma, no reference pixels."""
    data = rate_model.data
    good = np.isfinite(data) & ((rate_model.dq & DO_NOT_USE) == 0)
    level = np.nanmedian(data[good])
    sigma = robust_sigma(data[good])
    good &= np.abs(data - level) < 5.0 * sigma
    good[:readout.NREFPIX] = good[-readout.NREFPIX:] = False
    good[:, :readout.NREFPIX] = good[:, -readout.NREFPIX:] = False
    return good


def band_means(freq, spectra):
    """Mean spectrum per channel in three frequency bands and the white level."""
    row = 1.0 / (512 + readout.LINE_OVERHEAD)
    bands = {
        "slower_than_row": (freq > 0) & (freq < row),
        "row_to_50_clocks": (freq >= row) & (freq < 1.0 / 50.0),
        "faster_than_50_clocks": freq >= 1.0 / 50.0,
    }
    out = {name: [float(np.mean(s[sel])) for s in spectra] for name, sel in bands.items()}
    out["white_level"] = [float(np.median(s[freq > 0.25])) for s in spectra]
    return out


def evaluate(name, jump_model, mask, fastaxis, slowaxis, out_dir, reference_rate=None):
    """Ramp fit a cleaned jump model and compute the three metrics."""
    t0 = time.time()
    rate, _rateints = RampFitStep.call(jump_model, algorithm="OLS_C")
    diffs = np.diff(jump_model.data[0], axis=0)
    diff_mask = mask & np.all(jump_model.groupdq[0, 1:] == 0, axis=0)

    freq, spectra = channel_psd(diffs, diff_mask, fastaxis, slowaxis, max_lag=MAX_LAG)
    row_freq, row_power, row_means = row_banding_power(rate.data, mask, fastaxis, slowaxis)
    scatter = aperture_scatter(rate.data, mask, APERTURES)

    nrows_power = row_power.shape[1]
    low = (row_freq > 0) & (row_freq < 1.0 / 8.0)
    summary = {
        "variant": name,
        "seconds": round(time.time() - t0, 1),
        "rate_robust_sigma": float(robust_sigma(rate.data[mask])),
        "rate_change_vs_reference_robust_sigma": (
            None if reference_rate is None else float(robust_sigma((rate.data - reference_rate)[mask]))
        ),
        "rate_median": float(np.nanmedian(rate.data[mask])),
        "channel_spectrum_bands": band_means(freq, spectra),
        "row_banding_power_slower_than_8_rows": [float(np.mean(p[low])) for p in row_power],
        "row_banding_power_white": [float(np.mean(p[row_freq > 0.25])) for p in row_power],
        "row_means_robust_sigma": [float(robust_sigma(m)) for m in row_means],
        "column_means_robust_sigma": float(
            robust_sigma(np.nanmean(np.where(mask, rate.data, np.nan)[readout.NREFPIX:-readout.NREFPIX], axis=0))
        ),
        "aperture_sizes": APERTURES,
        "aperture_scatter": [float(v) for v in scatter["scatter"]],
        "aperture_white_prediction": [float(v) for v in scatter["white_prediction"]],
        "aperture_count": [int(v) for v in scatter["n_apertures"]],
        "n_row_power_bins": int(nrows_power),
    }
    np.savez_compressed(
        os.path.join(out_dir, f"{name}.npz"),
        freq=freq,
        spectra=spectra,
        row_freq=row_freq,
        row_power=row_power,
        row_means=row_means,
        rate=rate.data.astype(np.float32),
    )
    rate.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file")
    parser.add_argument("--ngroups", type=int, default=20)
    parser.add_argument("--nints", type=int, default=1, help="integrations to keep; metrics use the first")
    parser.add_argument("--out", default=None)
    parser.add_argument("--variants", default="all", help="comma-separated subset of variant names")
    args = parser.parse_args()

    out_root = args.out or os.path.join("results", f"{datetime.date.today().isoformat()}_darks_part_a")
    wanted = None if args.variants == "all" else set(args.variants.split(","))

    sirs = run_detector1_to_jump(args.file, max_groups=args.ngroups, max_ints=args.nints, refpix_algorithm="sirs")
    detector = sirs.meta.instrument.detector
    fastaxis, slowaxis = sirs.meta.subarray.fastaxis, sirs.meta.subarray.slowaxis
    out_dir = os.path.join(out_root, detector)
    os.makedirs(out_dir, exist_ok=True)

    rate_sirs, _ = RampFitStep.call(sirs, algorithm="OLS_C")
    mask = metric_mask(rate_sirs)
    reference_rate = rate_sirs.data.copy()
    rate_sirs.close()
    np.save(os.path.join(out_dir, "metric_mask.npy"), mask)

    clipped = {"background_method": "model", "clip_sigma": 2.0, "iterations": 2, "output_dir": out_dir}

    def linear(box):
        return {"background_method": "integration_model", "background_box_size": (box, box), "output_dir": out_dir}

    variants = {
        "median_refpix": lambda: run_detector1_to_jump(
            args.file, max_groups=args.ngroups, max_ints=args.nints, refpix_algorithm="median"
        ),
        "sirs": lambda: sirs,
        "sirs_jwst_median": lambda: CleanFlickerNoiseStep.call(
            sirs, skip=False, fit_method="median", fit_by_channel=True, background_method="model"
        ),
        "sirs_row_median_clipped": lambda: StreamCleanStep.call(sirs, fit_method="row_median", **clipped),
        "sirs_wiener_clipped": lambda: StreamCleanStep.call(sirs, fit_method="wiener", max_lag=MAX_LAG, **clipped),
        "sirs_row_median_box32": lambda: StreamCleanStep.call(sirs, fit_method="row_median", **linear(32)),
        "sirs_boxcar_box32": lambda: StreamCleanStep.call(sirs, fit_method="boxcar", **linear(32)),
        "sirs_wiener_box32": lambda: StreamCleanStep.call(
            sirs, fit_method="wiener", max_lag=MAX_LAG, save_spectra=True, **linear(32)
        ),
        "sirs_wiener_box128": lambda: StreamCleanStep.call(sirs, fit_method="wiener", max_lag=MAX_LAG, **linear(128)),
        "sirs_wiener_box256": lambda: StreamCleanStep.call(sirs, fit_method="wiener", max_lag=MAX_LAG, **linear(256)),
        "sirs_wiener_median_bkg": lambda: StreamCleanStep.call(
            sirs, fit_method="wiener", max_lag=MAX_LAG, background_method="median", output_dir=out_dir
        ),
        "sirs_wiener_box32_minp100": lambda: StreamCleanStep.call(
            sirs, fit_method="wiener", max_lag=MAX_LAG, min_period=100, **linear(32)
        ),
        "sirs_wiener_box32_minp300": lambda: StreamCleanStep.call(
            sirs, fit_method="wiener", max_lag=MAX_LAG, min_period=300, **linear(32)
        ),
        "sirs_wiener_none": lambda: StreamCleanStep.call(
            sirs, fit_method="wiener", max_lag=MAX_LAG, background_method=None, output_dir=out_dir
        ),
        "sirs_wiener_box32_nocolumns": lambda: StreamCleanStep.call(
            sirs, fit_method="wiener", max_lag=MAX_LAG, column_offsets=False, **linear(32)
        ),
        "sirs_wiener_box32_exposure": lambda: StreamCleanStep.call(
            sirs, fit_method="wiener", max_lag=MAX_LAG, background_method="exposure_model",
            background_box_size=(32, 32), output_dir=out_dir,
        ),
    }

    summaries = []
    for name, make in variants.items():
        if wanted is not None and name not in wanted:
            continue
        print(f"=== {name}", flush=True)
        model = make()
        summary = evaluate(name, model, mask, fastaxis, slowaxis, out_dir, reference_rate)
        summaries.append(summary)
        print(json.dumps(summary, indent=1), flush=True)
        if model is not sirs:
            model.close()

    record = {
        "file": os.path.basename(args.file),
        "detector": detector,
        "ngroups_used": args.ngroups,
        "nints_used": int(sirs.data.shape[0]),
        "max_lag": MAX_LAG,
        "mask_fraction": float(mask.mean()),
        "variants": summaries,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as handle:
        json.dump(record, handle, indent=1)
    print(f"wrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()

"""Experiment C.3.3 (counts and calibration part): reset-anchored fit on a bright field.

The ``_uncal`` exposure is run in memory through Detector1 to jump with SIRS
reference pixel correction, then fitted three ways: OLS_C (pipeline
default), stcal LIKELY, and `ResetRampFitStep` with the given sigma_b
calibration.  Reported:

- pixels with a finite rate under each fit and how many the reset fit adds
  (plan metric 6, count part);
- the t1 calibration of plan B.1.3: for pixels whose first difference is
  between 50 and 2000 DN with no flags, ``r = d_0 / a_rest`` where ``d_0``
  is the first group over the first group time and ``a_rest`` the LIKELY
  rate from the other differences.  Because the fitter multiplies the first
  group's read times by ``t1_scale``, the scale that makes ``d_0`` unbiased
  is the median of ``r`` itself (the plan's B.1.3 says its inverse).  The
  ratio is also binned by first difference and a two-parameter model
  ``R1 = b + k a t1`` is fitted, giving ``t1_scale = k`` and ``reset_offset = b``;
- the chi-square check of plan B.1.4 from the ``d0_resid`` diagnostic of
  the reset fit over the same pixels, with the current t1_scale.

Output: ``summary.json`` and the three rate images in
``results/<date>_reset_field/<detector>/``.
"""
import argparse
import datetime
import json
import os

import numpy as np
from jwst.ramp_fitting import RampFitStep
from stdatamodels.jwst.datamodels import dqflags

from snowblind import ResetRampFitStep, readout
from snowblind.metrics import d0_chisq, recovered_fraction, robust_sigma
from snowblind.reset_calib import read_calibration, run_detector1_to_jump

DO_NOT_USE = dqflags.pixel["DO_NOT_USE"]
SATURATED = dqflags.pixel["SATURATED"]


def first_group_time(model):
    """Mean time since reset of the first group's reads, stcal convention."""
    return model.meta.exposure.frame_time * (model.meta.exposure.nframes + 1) / 2.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file")
    parser.add_argument("--calib", required=True, help="reset calibration ASDF for this detector")
    parser.add_argument("--out", default=None)
    parser.add_argument("--per-pixel", action="store_true")
    parser.add_argument("--t1-scale", type=float, default=None)
    parser.add_argument("--use-pedestal-mean", action="store_true")
    parser.add_argument("--reset-offset", type=float, default=0.0)
    parser.add_argument(
        "--write-timing", action="store_true",
        help="store the fitted detector-wide and per-channel (k, b) in the calibration file",
    )
    args = parser.parse_args()

    jump = run_detector1_to_jump(args.file, refpix_algorithm="sirs")
    detector = jump.meta.instrument.detector
    out_root = args.out or os.path.join("results", f"{datetime.date.today().isoformat()}_reset_field")
    out_dir = os.path.join(out_root, detector)
    os.makedirs(out_dir, exist_ok=True)

    science = np.zeros(jump.data.shape[-2:], dtype=bool)
    science[readout.NREFPIX:-readout.NREFPIX, readout.NREFPIX:-readout.NREFPIX] = True

    ols, _ = RampFitStep.call(jump, algorithm="OLS_C")
    likely, _ = RampFitStep.call(jump, algorithm="LIKELY")
    reset, reset_ints = ResetRampFitStep.call(
        jump, reset_calib=args.calib, per_pixel_sigma_b=args.per_pixel, t1_scale=args.t1_scale,
        use_pedestal_mean=args.use_pedestal_mean, reset_offset=args.reset_offset,
        save_opt=True, output_dir=out_dir,
    )
    import asdf

    resetfit_path = [f for f in os.listdir(out_dir) if f.endswith("_resetfit.asdf")][0]
    with asdf.open(os.path.join(out_dir, resetfit_path), lazy_load=False, memmap=False) as handle:
        used_reset = np.array(handle.tree["used_reset"][0])
        d0_resid = np.array(handle.tree["d0_resid"][0])
        t1_scale_used = float(handle.tree["t1_scale"])

    gdq = jump.groupdq[0]
    first_sat = np.where((gdq & SATURATED).any(axis=0), (gdq & SATURATED).argmax(axis=0), gdq.shape[0])
    one_good = (first_sat == 1) & ((gdq[0] & DO_NOT_USE) == 0) & science

    counts = {
        "ols_vs_likely": recovered_fraction(ols.data, likely.data, science),
        "reset_vs_ols": recovered_fraction(reset.data, ols.data, science),
        "reset_vs_likely": recovered_fraction(reset.data, likely.data, science),
        "n_pixels_saturated_from_group_2": int(one_good.sum()),
        "n_of_those_finite_in_reset": int(np.isfinite(reset.data[one_good]).sum()),
        "n_of_those_finite_in_ols": int(np.isfinite(ols.data[one_good]).sum()),
        "n_of_those_finite_in_likely": int(np.isfinite(likely.data[one_good]).sum()),
    }

    # t1 calibration (B.1.3) on bright, unflagged, unsaturated pixels
    t1 = first_group_time(jump)
    d0 = jump.data[0, 0] / t1
    first_diff = jump.data[0, 1] - jump.data[0, 0]
    clean = np.all(gdq == 0, axis=0) & science & np.isfinite(likely.data) & (likely.data > 0)
    bright = clean & (first_diff > 50.0) & (first_diff < 2000.0)
    ratio = d0 / likely.data
    r_all = ratio[bright]
    fastaxis, slowaxis = jump.meta.subarray.fastaxis, jump.meta.subarray.slowaxis
    bright_det = readout.to_detector_frame(bright, fastaxis, slowaxis)
    ratio_det = readout.to_detector_frame(ratio, fastaxis, slowaxis)
    per_channel = [
        float(np.median(ratio_det[:, cols][bright_det[:, cols]])) for cols in readout.channel_slices(ratio_det.shape[-1])
    ]
    edges = np.array([50.0, 100.0, 200.0, 400.0, 800.0, 1200.0, 2000.0])
    by_flux = []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = clean & (first_diff >= low) & (first_diff < high)
        if selected.sum() > 50:
            r_bin = ratio[selected]
            offset = (d0 - likely.data)[selected] * t1
            by_flux.append(
                {
                    "first_diff_dn": [float(low), float(high)],
                    "n": int(selected.sum()),
                    "median_ratio": float(np.median(r_bin)),
                    "median_offset_dn": float(np.median(offset)),
                    "ratio_standard_error": float(robust_sigma(r_bin) / np.sqrt(selected.sum())),
                }
            )
    # Two-parameter model R1 = b + k a t1, fitted to the bin medians of the
    # offset R1 - a t1 against a t1 (n-weighted).  k is the t1_scale to use
    # (the fitter multiplies the first group's read times by it) and b the
    # reset_offset in DN.
    xs = np.array([np.median((likely.data * t1)[clean & (first_diff >= lo) & (first_diff < hi)])
                   for lo, hi in zip(edges[:-1], edges[1:]) if (clean & (first_diff >= lo) & (first_diff < hi)).sum() > 50])
    ys = np.array([entry["median_offset_dn"] for entry in by_flux])
    ws = np.array([entry["n"] for entry in by_flux], dtype=float)
    slope, intercept = np.polyfit(xs, ys, 1, w=np.sqrt(ws))
    k_fit = float(1.0 + slope)
    faint = clean & (first_diff < 50.0) & (first_diff > -20.0)
    faint_offset = jump.data[0, 0] - k_fit * likely.data * t1
    b_faint = float(np.median(faint_offset[faint]))
    # the same fit per output channel
    channel_ids = readout.to_science_frame(
        np.repeat(np.arange(readout.NOUTPUTS), readout.channel_width(ratio_det.shape[-1]))[None, :]
        * np.ones((ratio_det.shape[0], 1), int), fastaxis, slowaxis,
    )
    per_channel_fit = []
    for k in range(readout.NOUTPUTS):
        in_channel = clean & (channel_ids == k)
        cx, cy, cw = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            selected = in_channel & (first_diff >= lo) & (first_diff < hi)
            if selected.sum() > 50:
                cx.append(np.median((likely.data * t1)[selected]))
                cy.append(np.median((d0 - likely.data)[selected] * t1))
                cw.append(selected.sum())
        faint_channel = faint & (channel_ids == k)
        if len(cx) >= 3:
            c_slope, c_int = np.polyfit(np.array(cx), np.array(cy), 1, w=np.sqrt(np.array(cw, dtype=float)))
            c_k = float(1.0 + c_slope)
        else:
            c_int, c_k = intercept, k_fit
        # the offset the step stores comes from the sky pixels of the channel, which set the zero point
        c_b_faint = float(np.median((jump.data[0, 0] - c_k * likely.data * t1)[faint_channel]))
        per_channel_fit.append(
            {"k": c_k, "b_bright_fit": float(c_int), "b": c_b_faint, "n_bins": len(cx), "n_faint": int(faint_channel.sum())}
        )
    two_parameter = {
        "k_t1_scale": k_fit,
        "b_reset_offset_dn": float(intercept),
        "n_bins": int(xs.size),
        "per_channel": per_channel_fit,
        "b_from_faint_pixels_dn": b_faint,
        "n_faint_pixels": int(faint.sum()),
        "faint_offset_robust_sigma_dn": float(robust_sigma(faint_offset[faint])),
    }
    t1_calib = {
        "by_first_difference": by_flux,
        "two_parameter_fit": two_parameter,
        "t1_nominal_seconds": float(t1),
        "n_pixels": int(bright.sum()),
        "median_ratio": float(np.median(r_all)),
        "ratio_robust_sigma": float(robust_sigma(r_all)),
        "ratio_standard_error": float(robust_sigma(r_all) / np.sqrt(max(bright.sum(), 1))),
        "per_channel_median_ratio": per_channel,
        "implied_t1_scale_single_parameter": float(np.median(r_all)),
        "d0_median_dn_per_s": float(np.median(d0[bright])),
    }

    chisq = d0_chisq(d0_resid, used_reset & bright, None)
    chisq_all = d0_chisq(d0_resid, used_reset & clean, None)

    if args.write_timing:
        from snowblind.reset_calib import update_timing

        update_timing(
            args.calib,
            k_fit,
            b_faint,
            t1_scale_channels=[c["k"] for c in per_channel_fit],
            reset_offset_channels=[c["b"] for c in per_channel_fit],
            provenance={
                "file": os.path.basename(args.file),
                "k": "slope of R1 - a t1 against a t1 over first differences of 50 to 2000 DN, no flags",
                "b": "median of R1 - k a t1 over sky pixels with first differences below 50 DN",
            },
        )
        print(f"wrote timing into {args.calib}")

    calib = read_calibration(args.calib)
    summary = {
        "file": os.path.basename(args.file),
        "detector": detector,
        "readpatt": jump.meta.exposure.readpatt,
        "ngroups": int(jump.data.shape[1]),
        "sigma_b_median_dn": float(calib["sigma_b_median"]),
        "per_pixel_sigma_b": bool(args.per_pixel),
        "use_pedestal_mean": bool(args.use_pedestal_mean),
        "reset_offset_dn": float(args.reset_offset),
        "t1_scale_used": t1_scale_used,
        "counts": counts,
        "t1_calibration": t1_calib,
        "d0_chisq_bright_pixels": chisq,
        "d0_chisq_all_clean_pixels": chisq_all,
        "rate_median_dn_per_s": {
            "ols": float(np.nanmedian(ols.data[science])),
            "likely": float(np.nanmedian(likely.data[science])),
            "reset": float(np.nanmedian(reset.data[science])),
        },
        "rate_difference_reset_minus_likely_robust_sigma": float(robust_sigma((reset.data - likely.data)[clean])),
    }
    np.savez_compressed(
        os.path.join(out_dir, "rates.npz"),
        ols=ols.data.astype(np.float32),
        likely=likely.data.astype(np.float32),
        reset=reset.data.astype(np.float32),
        reset_err=reset.err.astype(np.float32),
        one_good=one_good,
        bright=bright,
    )
    with open(os.path.join(out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=1)
    print(json.dumps(summary, indent=1))
    print(f"wrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()

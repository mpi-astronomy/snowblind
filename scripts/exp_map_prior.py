"""Does a detector's pedestal map transfer to a field exposure, and what does it do to the reset fit?

For one ``_uncal`` exposure and the detector's reset calibration file:

1. Transfer.  On sky pixels (no flags in any group, first difference
   -20 to 50 DN) the field's own reset level is estimated as group 1 minus
   the rate of groups 2 to 5 times the calibrated first-read time
   (k_channel * t1), with each channel's median taken out (the step anchors
   that offset to the sky).  It is compared with the high-passed pedestal
   map the step would use (each channel's median taken out too): the
   robust scatter of the field estimate, of the estimate minus the map, of
   the map, and the least-squares slope of the estimate on the map.  A map
   that transfers gives a slope near 1 and a smaller scatter after
   subtraction.
2. Effect on the fit.  ResetRampFitStep with the scalar prior and with
   ``use_pedestal_mean``, both with the file's per-channel timing and the
   sky-anchored offsets: the chi-square of the reset difference
   (all clean pixels; bright pixels with a first difference of 50 to 2000
   DN), the median sky rate of each against OLS_C, and the map-minus-scalar
   rate difference: its robust scatter and the scatter of its row and
   column means over sky pixels.

Writes ``summary.json`` under ``<out>/<detector>/<exposure>/``, e.g.
``results/2026-11-01_map_prior/NRCB1/jw06154001001_02101_00001/``.
"""
import argparse
import json
import os
import tempfile

import asdf
import numpy as np
from jwst.ramp_fitting import RampFitStep

from snowblind import ResetRampFitStep, readout
from snowblind.metrics import d0_chisq, robust_sigma
from snowblind.reset_calib import read_calibration, run_detector1_to_jump
from snowblind.reset_rampfit import calibrated_pedestal_map, first_group_time


def channel_ids(shape, fastaxis, slowaxis):
    """Output channel (0 to 3) of each science-frame pixel."""
    ids = np.zeros(shape, dtype=int)
    for k, (rows, cols) in enumerate(readout.channel_blocks(shape, fastaxis, slowaxis)):
        ids[rows, cols] = k
    return ids


def minus_channel_medians(values, mask, channels):
    """``values`` with the median over ``mask`` of each channel subtracted in that channel."""
    out = values.astype(np.float64).copy()
    for k in range(readout.NOUTPUTS):
        in_channel = channels == k
        out[in_channel] -= np.median(values[mask & in_channel])
    return out


def line_mean_scatter(image, mask, axis, min_pixels=200):
    """Robust scatter of the means over ``mask`` along ``axis`` (axis 1 gives row means), lines with enough pixels."""
    counts = mask.sum(axis=axis)
    sums = np.where(mask, image, 0.0).sum(axis=axis)
    good = counts >= min_pixels
    return float(robust_sigma(sums[good] / counts[good]))


def reset_fit(jump, calib, use_map, out_dir):
    """Rate image and (used_reset, d0_resid) of ResetRampFitStep with the scalar or the map prior."""
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        rate, _ = ResetRampFitStep.call(jump, reset_calib=calib, use_pedestal_mean=use_map, save_opt=True, output_dir=tmp)
        name = [f for f in os.listdir(tmp) if f.endswith("_resetfit.asdf")][0]
        with asdf.open(os.path.join(tmp, name), lazy_load=False, memmap=False) as handle:
            used = np.array(handle.tree["used_reset"][0])
            resid = np.array(handle.tree["d0_resid"][0])
    return np.array(rate.data, dtype=np.float64), used, resid


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file")
    parser.add_argument("--calib", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    jump = run_detector1_to_jump(args.file, refpix_algorithm="sirs")
    detector = jump.meta.instrument.detector
    exposure = os.path.basename(args.file).split("_nrc")[0]
    out_dir = os.path.join(args.out, detector, exposure)
    os.makedirs(out_dir, exist_ok=True)
    fastaxis, slowaxis = jump.meta.subarray.fastaxis, jump.meta.subarray.slowaxis
    shape = jump.data.shape[-2:]
    channels = channel_ids(shape, fastaxis, slowaxis)

    science = np.zeros(shape, dtype=bool)
    science[readout.NREFPIX:-readout.NREFPIX, readout.NREFPIX:-readout.NREFPIX] = True
    data = np.array(jump.data[0], dtype=np.float64)
    gdq = jump.groupdq[0]
    first_diff = data[1] - data[0]
    clean = np.all(gdq == 0, axis=0) & science
    sky = clean & (first_diff > -20.0) & (first_diff < 50.0)
    bright = clean & (first_diff > 50.0) & (first_diff < 2000.0)

    # 1. transfer of the map
    tree = read_calibration(args.calib)
    k = np.array(tree["t1_scale_channels"], dtype=np.float64)[channels]
    t1 = first_group_time(jump)
    times = t1 + jump.meta.exposure.group_time * np.arange(1, data.shape[0])
    dt = times - times.mean()
    rate_rest = np.tensordot(dt, data[1:] - data[1:].mean(axis=0), axes=1) / np.sum(dt**2)
    field = data[0] - rate_rest * k * t1
    pedestal = calibrated_pedestal_map(tree, shape, fastaxis, slowaxis, highpass=True)
    sky_map = sky & np.isfinite(pedestal)
    field = minus_channel_medians(field, sky_map, channels)
    prior = minus_channel_medians(np.where(np.isfinite(pedestal), pedestal, 0.0), sky_map, channels)
    # clip the field estimate at 8 robust sigma (unflagged cosmic rays and faint stars)
    keep = sky_map & (np.abs(field) < 8 * robust_sigma(field[sky_map]))
    x, y = prior[keep], field[keep]
    slope = float(np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2))
    transfer = {
        "n_sky_pixels": int(keep.sum()),
        "field_reset_scatter_dn": float(robust_sigma(y)),
        "field_minus_map_scatter_dn": float(robust_sigma(y - x)),
        "map_scatter_dn": float(robust_sigma(x)),
        "slope_field_on_map": slope,
        "correlation": float(np.corrcoef(x, y)[0, 1]),
        "field_row_mean_scatter_dn": line_mean_scatter(field, keep, axis=1),
        "field_minus_map_row_mean_scatter_dn": line_mean_scatter(field - prior, keep, axis=1),
        "field_column_mean_scatter_dn": line_mean_scatter(field, keep, axis=0),
        "field_minus_map_column_mean_scatter_dn": line_mean_scatter(field - prior, keep, axis=0),
    }

    # 2. effect on the fit
    ols, _ = RampFitStep.call(jump, algorithm="OLS_C")
    ols = np.array(ols.data, dtype=np.float64)
    fits = {}
    rates = {}
    for name, use_map in (("scalar", False), ("map", True)):
        rate, used, resid = reset_fit(jump, args.calib, use_map, out_dir)
        rates[name] = rate
        ok = sky & np.isfinite(rate) & np.isfinite(ols)
        fits[name] = {
            "d0_chisq_all_clean": d0_chisq(resid, used & clean),
            "d0_chisq_bright": d0_chisq(resid, used & bright),
            "sky_median_rate_ratio_to_ols": float(np.median(rate[ok]) / np.median(ols[ok])),
        }
    both = sky & np.isfinite(rates["map"]) & np.isfinite(rates["scalar"]) & np.isfinite(ols)
    difference = rates["map"] - rates["scalar"]
    fits["map_minus_scalar"] = {
        "robust_scatter_dn_per_s": float(robust_sigma(difference[both])),
        "median_dn_per_s": float(np.median(difference[both])),
        "row_mean_scatter_dn_per_s": line_mean_scatter(difference, both, axis=1),
        "column_mean_scatter_dn_per_s": line_mean_scatter(difference, both, axis=0),
        "ols_sky_rate_robust_scatter_dn_per_s": float(robust_sigma(ols[both])),
        "ols_sky_median_dn_per_s": float(np.median(ols[both])),
    }

    summary = {
        "file": os.path.basename(args.file),
        "detector": detector,
        "calibration": os.path.basename(args.calib),
        "superbias_data": jump.meta.ref_file.superbias.name,
        "superbias_calibration": tree.get("provenance", {}).get("superbias"),
        "t1_nominal_s": float(t1),
        "t1_scale_channels": [float(v) for v in tree["t1_scale_channels"]],
        "transfer": transfer,
        "fits": fits,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=1)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()

"""Check the assumed channel readout directions on a real dark (plan task A.4.3).

Two independent tests are run on the group differences of a dark, processed
in memory through superbias subtraction only, in the detector frame, channel
by channel.

1. Boundary continuity.  With the per-row, per-channel mean removed, the
   stream is laid out under both candidate directions and the correlation
   between samples that straddle a row boundary is summed over clock lags
   13 to 100.  Under the right direction those samples really are that
   close in time and inherit the short-range correlation of the readout
   chain; under the wrong direction they are about two rows apart and
   uncorrelated.  The statistic is the mean over group differences of
   (forward minus backward) with its standard error.

2. Hot-pixel trailing.  The mean group difference of a dark shows hot
   pixels.  Stacking the horizontal profile around isolated hot pixels, the
   pixel read just after a hot pixel carries a settling residual while the
   one read just before does not.  The sign of the left-right asymmetry
   therefore flips between channels read in opposite directions.  A positive
   asymmetry (right minus left) means the channel is read toward increasing
   detector column.

Only the first integration and the first ``--ngroups`` groups of an
``_uncal`` file are used.  The only output written is a JSON record.

Usage:
    python scripts/check_readout_order.py FILE [--ngroups N] [--out DIR]
"""
import argparse
import datetime
import json
import os
import warnings

import numpy as np
from astropy.stats import mad_std
from jwst.dq_init import DQInitStep
from jwst.group_scale import GroupScaleStep
from jwst.saturation import SaturationStep
from jwst.superbias import SuperBiasStep
from scipy import ndimage
from stdatamodels.jwst import datamodels

from snowblind import readout

NEDGE = readout.NREFPIX
LAGS = np.arange(readout.LINE_OVERHEAD + 1, 101)
HOT_HALFWIDTH = 4


def trimmed_ramp(path, ngroups):
    """Open an uncal file and keep the first integration and ``ngroups`` groups."""
    with datamodels.open(path) as full:
        model = datamodels.RampModel(full.data[:1, :ngroups].astype(np.float32))
        model.update(full)
        model.groupdq = np.zeros(model.data.shape, dtype=np.uint8)
        model.pixeldq = np.zeros(model.data.shape[-2:], dtype=np.uint32)
    model.meta.exposure.nints = 1
    model.meta.exposure.ngroups = ngroups
    return model


def to_superbias_subtracted(model):
    """Run group_scale, dq_init, saturation and superbias in memory."""
    model = GroupScaleStep.call(model)
    model = DQInitStep.call(model)
    model = SaturationStep.call(model)
    model = SuperBiasStep.call(model)
    return model


def channel_differences(model):
    """Return (ndiffs, nrows, width) arrays of masked group differences per channel, in the detector frame."""
    fastaxis = model.meta.subarray.fastaxis
    slowaxis = model.meta.subarray.slowaxis
    data = readout.to_detector_frame(model.data[0], fastaxis, slowaxis)
    gdq = readout.to_detector_frame(model.groupdq[0], fastaxis, slowaxis)
    diffs = np.diff(data, axis=0).astype(np.float32)
    diffs[(gdq[1:] != 0) | (gdq[:-1] != 0)] = np.nan
    diffs[:, :NEDGE] = np.nan
    diffs[:, -NEDGE:] = np.nan
    channels = []
    for k, columns in enumerate(readout.channel_slices(data.shape[-1])):
        block = np.array(diffs[:, :, columns])
        if k == 0:
            block[:, :, :NEDGE] = np.nan
        if k == readout.NOUTPUTS - 1:
            block[:, :, -NEDGE:] = np.nan
        channels.append(block)
    return channels


def boundary_statistic(channel_diffs):
    """Mean cross-boundary correlation over LAGS for both directions, per group difference."""
    ndiffs, nrows, width = channel_diffs.shape
    row_of_clock = np.repeat(np.arange(nrows), width + readout.LINE_OVERHEAD)
    forward, backward = [], []
    for diff in channel_diffs:
        centered = diff - np.nanmedian(diff, axis=1, keepdims=True)
        finite = np.isfinite(centered)
        values = np.nan_to_num(centered)
        var = np.mean(values[finite] ** 2)
        for direction, store in ((1, forward), (-1, backward)):
            stream = readout.to_stream(values, direction)
            valid = readout.to_stream(finite, direction, fill=False)
            corr = []
            for lag in LAGS:
                ok = valid[:-lag] & valid[lag:] & (row_of_clock[:-lag] != row_of_clock[lag:])
                if ok.any():
                    corr.append(np.mean((stream[:-lag] * stream[lag:])[ok]) / var)
            store.append(np.mean(corr))
    forward = np.array(forward)
    backward = np.array(backward)
    delta = forward - backward
    return {
        "forward": float(forward.mean()),
        "backward": float(backward.mean()),
        "forward_minus_backward": float(delta.mean()),
        "standard_error": float(delta.std(ddof=1) / np.sqrt(delta.size)) if delta.size > 1 else float("nan"),
    }


def hot_pixel_statistic(channel_diffs, nsigma=20.0):
    """Left-right asymmetry of the stacked horizontal profile around isolated hot pixels."""
    rate = np.nanmean(channel_diffs, axis=0)
    finite = np.isfinite(rate)
    rate = np.nan_to_num(rate, nan=np.nanmedian(rate))
    level = np.median(rate)
    scatter = mad_std(rate)
    hot = (rate - level > nsigma * scatter) & finite
    # keep hot pixels whose neighbourhood holds no other hot pixel and no edge
    neighbours = ndimage.uniform_filter(hot.astype(float), size=2 * HOT_HALFWIDTH + 1) * (2 * HOT_HALFWIDTH + 1) ** 2
    isolated = hot & (np.round(neighbours) == 1)
    isolated[:, : HOT_HALFWIDTH + 1] = False
    isolated[:, -(HOT_HALFWIDTH + 1):] = False
    rows, cols = np.nonzero(isolated)
    if rows.size < 20:
        return {"n_hot": int(rows.size), "profile": None, "asymmetry": float("nan"), "standard_error": float("nan")}
    offsets = np.arange(-HOT_HALFWIDTH, HOT_HALFWIDTH + 1)
    profiles = (rate[rows[:, None], cols[:, None] + offsets[None, :]] - level) / (rate[rows, cols] - level)[:, None]
    profile = np.median(profiles, axis=0)
    asym = profiles[:, HOT_HALFWIDTH + 1] - profiles[:, HOT_HALFWIDTH - 1]
    return {
        "n_hot": int(rows.size),
        "profile": [float(v) for v in profile],
        "asymmetry": float(np.median(asym)),
        "standard_error": float(1.2533 * np.std(asym) / np.sqrt(asym.size)),
    }


def analyze(model):
    """Run both tests on every channel."""
    results = []
    for k, channel_diffs in enumerate(channel_differences(model)):
        boundary = boundary_statistic(channel_diffs)
        hot = hot_pixel_statistic(channel_diffs)
        results.append(
            {
                "channel": k,
                "boundary": boundary,
                "boundary_direction": int(np.sign(boundary["forward_minus_backward"])) or 0,
                "hot_pixel": hot,
                "hot_pixel_direction": int(np.sign(hot["asymmetry"])) if np.isfinite(hot["asymmetry"]) else 0,
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file")
    parser.add_argument("--ngroups", type=int, default=40)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    model = to_superbias_subtracted(trimmed_ramp(args.file, args.ngroups))
    warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
    record = {
        "file": os.path.basename(args.file),
        "detector": model.meta.instrument.detector,
        "fastaxis": model.meta.subarray.fastaxis,
        "slowaxis": model.meta.subarray.slowaxis,
        "ngroups_used": args.ngroups,
        "lags": [int(LAGS[0]), int(LAGS[-1])],
        "channels": analyze(model),
    }
    print(json.dumps(record, indent=2))

    out = args.out or os.path.join("results", f"{datetime.date.today().isoformat()}_readout_order")
    os.makedirs(out, exist_ok=True)
    name = os.path.join(out, record["file"].replace("_uncal.fits", "_readout_order.json"))
    with open(name, "w") as handle:
        json.dump(record, handle, indent=2)
    print(f"wrote {name}")


if __name__ == "__main__":
    main()

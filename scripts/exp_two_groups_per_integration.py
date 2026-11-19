"""Experiment for issue #47: the reset offset per integration and channel from 2 and 3 groups.

An ``_uncal`` exposure with several integrations is run through
``snowblind_detector1`` with the ramp fit skipped, on its first 2 and its
first 3 groups.  In each integration and output channel the reset offset is
measured as ``sky_anchored_offsets`` does (median over faint unflagged
pixels of the first group minus the sky signal expected there), with the
sky slope from group 1 minus group 0 (2 groups) or from groups 1 and 2
(3 groups, as the step does on the first integration).  The same median on
the two halves of a checkerboard of pixels gives the statistical error: half
the difference of the two halves.

Output: ``offsets.json`` in ``results/<date>_two_groups_per_integration/<exposure>/``.
"""
import argparse
import datetime
import json
import os

import numpy as np
from jwst import datamodels
from stdatamodels.jwst.datamodels import dqflags

import snowblind.reset_rampfit as reset_rampfit
from snowblind import readout
from snowblind.pipeline import Detector1Pipeline
from snowblind.reset_calib import default_calibration


def _first_groups(path, ngroups):
    """The exposure's first ``ngroups`` groups as a RampModel, its header saying so."""
    with datamodels.RampModel(path) as opened:
        trimmed = datamodels.RampModel(np.array(opened.data[:, :ngroups], dtype=np.float32))
        trimmed.update(opened)
        trimmed.groupdq = np.zeros(trimmed.data.shape, dtype=np.uint8)
        trimmed.pixeldq = np.zeros(trimmed.data.shape[-2:], dtype=np.uint32)
        if getattr(opened, "int_times", None) is not None and len(opened.int_times) > 0:
            trimmed.int_times = opened.int_times
        trimmed.meta.exposure.ngroups = ngroups
        trimmed.meta.filename = opened.meta.filename
    return trimmed


def _offsets_per_integration(ramp, t1_scales, max_signal=50.0):
    """Reset offset, its checkerboard half-difference and the sky pixel count, per integration and channel."""
    nints, ngroups, nrows, ncols = ramp.data.shape
    group_time = float(ramp.meta.exposure.group_time)
    t1 = reset_rampfit.first_group_time(ramp)
    blocks = readout.channel_blocks((nrows, ncols), ramp.meta.subarray.fastaxis, ramp.meta.subarray.slowaxis)
    rows, cols = np.indices((nrows, ncols))
    checker = (rows + cols) % 2 == 0
    not_dnu = (ramp.pixeldq & dqflags.pixel["DO_NOT_USE"]) == 0
    out = []
    for integ in range(nints):
        data = np.asarray(ramp.data[integ], dtype=np.float64)
        if ngroups == 2:
            slope = (data[1] - data[0]) / group_time
        else:
            slope = (data[2] - data[1]) / group_time
        clean = np.all(ramp.groupdq[integ] == 0, axis=0) & not_dnu & np.isfinite(slope) & np.isfinite(data[0])
        per_channel = []
        for (r, c), k in zip(blocks, t1_scales):
            signal = slope[r, c] * (t1 - (1.0 - k) * t1)
            faint = clean[r, c] & (signal < max_signal) & (signal > -0.4 * max_signal)
            residual = data[0][r, c] - signal
            half_a = np.median(residual[faint & checker[r, c]])
            half_b = np.median(residual[faint & ~checker[r, c]])
            per_channel.append(
                {
                    "offset": float(np.median(residual[faint])),
                    "half_difference": float(abs(half_a - half_b) / 2),
                    "sky_pixels": int(faint.sum()),
                    "sky_signal_median": float(np.median(signal[faint])),
                }
            )
        out.append(per_channel)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    name = os.path.basename(args.file).replace("_uncal.fits", "")
    out_root = args.out or os.path.join("results", f"{datetime.date.today().isoformat()}_two_groups_per_integration")
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)

    summary = {"exposure": name}
    for ngroups in (2, 3):
        ramp = Detector1Pipeline.call(_first_groups(args.file, ngroups), steps={"ramp_fit": {"skip": True}})
        tree = default_calibration(ramp.meta.instrument.detector)
        t1 = reset_rampfit.first_group_time(ramp)
        scales = [reset_rampfit.rescale_t1(k, float(tree["t1_seconds"]), t1) for k in tree["t1_scale_channels"]]
        summary["readpatt"] = ramp.meta.exposure.readpatt
        summary["exp_type"] = ramp.meta.exposure.type
        summary["first_group_time"] = t1
        summary["t1_scales"] = scales
        summary["packaged_offsets"] = [float(b) for b in tree["reset_offset_channels"]]
        summary[f"g{ngroups}"] = _offsets_per_integration(ramp, scales)
        del ramp
        for integ, channels in enumerate(summary[f"g{ngroups}"]):
            print(f"g{ngroups} int {integ + 1}: " + "  ".join(
                f"{c['offset']:+.2f} (+-{c['half_difference']:.2f}, n={c['sky_pixels']})" for c in channels
            ), flush=True)

    with open(os.path.join(out_dir, "offsets.json"), "w") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()

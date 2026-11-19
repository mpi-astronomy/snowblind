"""Experiment for issue #47: the reset-anchored fit on 2-group exposures.

An ``_uncal`` exposure with many groups is run through ``snowblind_detector1``
with the ramp fit skipped, once on all groups and once each on its first 2
and first 3 groups, as if those were exposures of their own.  The full ramp
fitted with the jwst OLS_C fit gives the reference rate.  The short ramps are
fitted with:

- ``ols``: the jwst OLS_C fit, what ``--reset_fit=False`` gives;
- ``reset``: `ResetRampFitStep` with its defaults; at 2 groups (allowed here
  by lowering ``MIN_NGROUPS``) it cannot anchor the reset offset to the sky
  and uses the packaged calibration's offsets;
- ``reset_anchor2`` (2 groups): offsets measured per channel from the sky
  pixels as the step does, with the sky slope taken from group 1 minus
  group 0 instead of the later groups;
- ``reset_full_offsets`` (2 groups): offsets anchored on the full ramp of the
  same exposure, the best this exposure allows;
- ``reset_file`` (3 groups): the step with ``anchor_offset_to_sky=False``.

Reported per fit, for pixels with no DO_NOT_USE or SATURATED flag: the
median and robust sigma of (short-ramp rate - reference rate) in DN/s on sky
pixels (reference rate within 5 robust sigma of its median), per output
channel and overall, and the median ratio to the reference in brighter bins.

Output: ``summary.json``, ``rates.npz`` and the log in
``results/<date>_two_groups/<exposure>/``.
"""
import argparse
import datetime
import json
import os

import numpy as np
from jwst import datamodels
from jwst.ramp_fitting import RampFitStep
from stdatamodels.jwst.datamodels import dqflags

import snowblind.likely_fit_reset as likely_fit_reset
import snowblind.reset_rampfit as reset_rampfit
from snowblind import ResetRampFitStep, readout
from snowblind.metrics import robust_sigma
from snowblind.pipeline import Detector1Pipeline

BAD = dqflags.pixel["DO_NOT_USE"] | dqflags.pixel["SATURATED"]
BRIGHT_BINS = [(10.0, 30.0), (30.0, 100.0), (100.0, 300.0), (300.0, 1000.0)]


def _first_groups(path, ngroups):
    """The exposure's first ``ngroups`` groups (all when None) as a RampModel, its header saying so."""
    with datamodels.RampModel(path) as opened:
        ngroups = opened.data.shape[1] if ngroups is None else ngroups
        trimmed = datamodels.RampModel(np.array(opened.data[:, :ngroups], dtype=np.float32))
        trimmed.update(opened)
        trimmed.groupdq = np.zeros(trimmed.data.shape, dtype=np.uint8)
        trimmed.pixeldq = np.zeros(trimmed.data.shape[-2:], dtype=np.uint32)
        if getattr(opened, "int_times", None) is not None and len(opened.int_times) > 0:
            trimmed.int_times = opened.int_times
        trimmed.meta.exposure.ngroups = ngroups
        trimmed.meta.filename = opened.meta.filename
    return trimmed


def _prepared_ramp(path, ngroups):
    """``snowblind_detector1`` up to the ramp fit on the first ``ngroups`` groups."""
    return Detector1Pipeline.call(_first_groups(path, ngroups), steps={"ramp_fit": {"skip": True}})


def _fit_record(rate):
    """The reset fit's record of offsets (the first integration's) and first-read scales on a rate model."""
    record = dict(rate.meta.instance.get("snowblind", {}).get("reset_rampfit", {}))
    if record.get("reset_offset"):
        record["reset_offset"] = [float(b) for b in record["reset_offset"][0]]
    return record


def _with_channel_offsets(offsets):
    """A stand-in for default_calibration whose per-channel reset offsets are ``offsets``."""
    packaged = reset_rampfit.default_calibration

    def calibration(detector):
        tree = dict(packaged(detector))
        tree["reset_offset_channels"] = [float(b) for b in offsets]
        return tree

    return calibration


def _offsets_from_first_difference(ramp, t1_scales, fallback):
    """Sky-anchored reset offsets per channel with the sky slope from group 1 minus group 0."""
    data = np.asarray(ramp.data[0], dtype=np.float64)
    t1 = reset_rampfit.first_group_time(ramp)
    slope = (data[1] - data[0]) / float(ramp.meta.exposure.group_time)
    clean = (
        np.all(ramp.groupdq[0] == 0, axis=0)
        & ((ramp.pixeldq & dqflags.pixel["DO_NOT_USE"]) == 0)
        & np.isfinite(slope)
        & np.isfinite(data[0])
    )
    blocks = readout.channel_blocks(data.shape[-2:], ramp.meta.subarray.fastaxis, ramp.meta.subarray.slowaxis)
    offsets = reset_rampfit.sky_anchored_offsets(ramp, 0.0, t1_scales, blocks, False, resultant=(data[0], t1, clean, slope, t1))
    return [b if np.isfinite(b) else kept for b, kept in zip(offsets, fallback)]


def _compare(rate, reference, good, sky, blocks):
    """Differences of one rate image from the reference on sky pixels, per channel, and ratios in bright bins."""
    diff = rate - reference
    usable = good & np.isfinite(diff)
    on_sky = usable & sky
    out = {
        "sky_pixels": int(on_sky.sum()),
        "sky_median_diff": float(np.median(diff[on_sky])),
        "sky_sigma_diff": robust_sigma(diff[on_sky]),
        "channel_median_diff": [float(np.median(diff[rows, cols][on_sky[rows, cols]])) for rows, cols in blocks],
        "bright": [],
    }
    for low, high in BRIGHT_BINS:
        in_bin = usable & (reference >= low) & (reference < high)
        ratio = rate[in_bin] / reference[in_bin]
        out["bright"].append(
            {"range": [low, high], "pixels": int(in_bin.sum()), "median_ratio": float(np.median(ratio)) if in_bin.any() else None}
        )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    name = os.path.basename(args.file).replace("_uncal.fits", "")
    out_root = args.out or os.path.join("results", f"{datetime.date.today().isoformat()}_two_groups")
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)

    full = _prepared_ramp(args.file, None)
    ngroups_full = full.data.shape[1]
    reference, _ = RampFitStep.call(full, algorithm="OLS_C")
    full_reset, _ = ResetRampFitStep.call(full)
    full_record = _fit_record(full_reset)
    blocks = readout.channel_blocks(full.data.shape[-2:], full.meta.subarray.fastaxis, full.meta.subarray.slowaxis)
    meta = {
        "exposure": name,
        "exp_type": full.meta.exposure.type,
        "readpatt": full.meta.exposure.readpatt,
        "ngroups_full": ngroups_full,
        "group_time": float(full.meta.exposure.group_time),
        "first_group_time": reset_rampfit.first_group_time(full),
        "full_reset_record": full_record,
    }
    del full

    reference_rate = np.array(reference.data, dtype=np.float64)
    good = (reference.dq & BAD) == 0
    science = np.zeros(reference_rate.shape, dtype=bool)
    science[readout.NREFPIX:-readout.NREFPIX, readout.NREFPIX:-readout.NREFPIX] = True
    good &= science & np.isfinite(reference_rate)
    center = float(np.median(reference_rate[good]))
    spread = robust_sigma(reference_rate[good])
    sky = np.abs(reference_rate - center) < 5 * spread
    meta["reference_sky_median"] = center
    meta["reference_sky_sigma"] = spread
    meta["reference_sky_median_err"] = float(np.median(reference.err[good & sky]))

    rates = {"reference": reference.data, "full_reset": full_reset.data}
    results = {"full_reset": _compare(np.array(full_reset.data, dtype=np.float64), reference_rate, good, sky, blocks)}
    offsets = {"full_ramp_anchored": full_record.get("reset_offset")}
    errs = {}

    def add(label, rate):
        rates[label] = rate.data
        fit_good = good & ((rate.dq & BAD) == 0)
        results[label] = _compare(np.array(rate.data, dtype=np.float64), reference_rate, fit_good, sky, blocks)
        errs[label] = float(np.median(rate.err[fit_good & sky]))
        print(f"{label}: sky median diff {results[label]['sky_median_diff']:+.4f} DN/s, "
              f"sigma {results[label]['sky_sigma_diff']:.4f}", flush=True)

    three = _prepared_ramp(args.file, 3)
    add("g3_ols", RampFitStep.call(three, algorithm="OLS_C")[0])
    rate3 = ResetRampFitStep.call(three)[0]
    offsets["g3_anchored"] = _fit_record(rate3).get("reset_offset")
    add("g3_reset", rate3)
    add("g3_reset_file", ResetRampFitStep.call(three, anchor_offset_to_sky=False)[0])
    del three

    two = _prepared_ramp(args.file, 2)
    add("g2_ols", RampFitStep.call(two, algorithm="OLS_C")[0])
    reset_rampfit.MIN_NGROUPS = likely_fit_reset.MIN_NGROUPS = 2
    rate2 = ResetRampFitStep.call(two)[0]
    record2 = _fit_record(rate2)
    offsets["g2_file"] = record2.get("reset_offset")
    meta["g2_reset_record"] = record2
    add("g2_reset", rate2)

    anchor2 = _offsets_from_first_difference(two, record2["first_read_time_scale"], record2["reset_offset"])
    offsets["g2_anchor_first_difference"] = anchor2
    packaged = reset_rampfit.default_calibration
    for label, chosen in (("g2_reset_anchor2", anchor2), ("g2_reset_full_offsets", full_record["reset_offset"])):
        reset_rampfit.default_calibration = _with_channel_offsets(chosen)
        add(label, ResetRampFitStep.call(two)[0])
        reset_rampfit.default_calibration = packaged

    summary = {"meta": meta, "offsets_dn": offsets, "sky_median_err": errs, "fits": results}
    with open(os.path.join(out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    np.savez_compressed(os.path.join(out_dir, "rates.npz"), **{k: np.asarray(v, dtype=np.float32) for k, v in rates.items()})
    print(json.dumps({"offsets_dn": offsets, "sky_median_err": errs}, indent=2))


if __name__ == "__main__":
    main()

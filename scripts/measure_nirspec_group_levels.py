"""Per-group median signal of a NIRSpec ramp after each Detector1 step, against the reference dark.

Usage: MAXG=44 python scripts/measure_nirspec_group_levels.py <uncal.fits> <out.json> [mean]

``MAXG`` keeps only the first groups; ``mean`` turns on the refpix step's
``irs2_mean_subtraction``.

For every stage (raw, superbias, refpix, linearity, dark) and every group k
prints the median over the usable science pixels of data[k] - data[0] in DN,
globally and per amplifier (512-row bands of the 2048 x 2048 frame once the
reference pixels are stripped), and the same for the reference dark the dark
step used.  Also the median group-to-group difference, to show curvature.
"""
import json
import os
import sys

import numpy as np
from jwst.dark_current import DarkCurrentStep
from jwst.dq_init import DQInitStep
from jwst.group_scale import GroupScaleStep
from jwst.linearity import LinearityStep
from jwst.refpix import RefPixStep
from jwst.refpix.irs2_subtract_reference import make_irs2_mask
from jwst.saturation import SaturationStep
from jwst.superbias import SuperBiasStep
from stdatamodels.jwst import datamodels

BORDER = 4


def per_group_medians(cube, good):
    """Median of cube[k] - cube[0] over good pixels, for every k, plus per-amplifier (512-row bands)."""
    base = cube[0]
    out = {"all": [], "amps": [[] for _ in range(4)], "step": []}
    nrows = cube.shape[1]
    band = nrows // 4
    prev = None
    for k in range(cube.shape[0]):
        diff = cube[k] - base
        out["all"].append(float(np.median(diff[good])))
        for a in range(4):
            sel = good[a * band:(a + 1) * band]
            out["amps"][a].append(float(np.median(diff[a * band:(a + 1) * band][sel])))
        if prev is not None:
            out["step"].append(float(np.median((cube[k] - prev)[good])))
        prev = cube[k]
    return out


def main():
    path, out_path = sys.argv[1], sys.argv[2]
    mean_subtraction = len(sys.argv) > 3 and sys.argv[3] == "mean"
    model = datamodels.RampModel(path)
    max_groups = int(os.environ.get("MAXG", "0"))
    if max_groups and model.data.shape[1] > max_groups:
        trimmed = datamodels.RampModel(model.data[:1, :max_groups].astype(np.float32))
        trimmed.update(model)
        trimmed.groupdq = np.zeros(trimmed.data.shape, dtype=np.uint8)
        trimmed.pixeldq = np.zeros(trimmed.data.shape[-2:], dtype=np.uint32)
        trimmed.meta.exposure.ngroups = max_groups
        trimmed.meta.exposure.nints = 1
        model = trimmed
    results = {"file": path, "irs2_mean_subtraction": mean_subtraction, "ngroups": int(model.data.shape[1])}

    data = model.data[0]
    ny, nx = data.shape[-2:]
    # science pixels of the raw IRS2 frame, along whichever axis is interleaved
    if nx == 3200:
        mask1d = make_irs2_mask(nx, ny, 16, 4)
        sci = np.zeros((ny, nx), bool)
        sci[:, mask1d] = True
    elif ny == 3200:
        mask1d = make_irs2_mask(ny, nx, 16, 4)
        sci = np.zeros((ny, nx), bool)
        sci[mask1d, :] = True
    else:
        sci = np.ones((ny, nx), bool)
    good_raw = sci.copy()
    results["raw"] = per_group_medians(data, good_raw)
    if sci.all():
        # traditional readout: the border reference pixels stand in for both reference classes
        border = np.zeros((ny, nx), bool)
        border[:4] = border[-4:] = True
        border[:, :4] = border[:, -4:] = True
        results["raw_refout"] = per_group_medians(data, border)
        results["raw_interleaved"] = per_group_medians(data, border)
        good_raw = ~border
        results["raw"] = per_group_medians(data, good_raw)
    else:
        # the reference output is the longest run of non-science positions, the rest are interleaved references
        runs, start = [], None
        for i, m in enumerate(list(mask1d) + [True]):
            if not m and start is None:
                start = i
            if m and start is not None:
                runs.append((start, i))
                start = None
        refout_run = max(runs, key=lambda r: r[1] - r[0])
        refout = np.zeros_like(mask1d)
        refout[refout_run[0]:refout_run[1]] = True
        interleaved = ~mask1d & ~refout
        results["refout_run"] = list(refout_run)
        ro = np.zeros((ny, nx), bool)
        il = np.zeros((ny, nx), bool)
        if nx == 3200:
            ro[:, refout] = True
            il[:, interleaved] = True
        else:
            ro[refout, :] = True
            il[interleaved, :] = True
        results["raw_refout"] = per_group_medians(data, ro)
        results["raw_interleaved"] = per_group_medians(data, il)
    print("raw done", flush=True)

    model = GroupScaleStep.call(model)
    model = DQInitStep.call(model)
    model = SaturationStep.call(model)
    model = SuperBiasStep.call(model)
    good_bias = good_raw & (model.pixeldq == 0) if model.pixeldq.shape == good_raw.shape else good_raw
    results["superbias"] = per_group_medians(model.data[0], good_bias)
    print("superbias done", flush=True)

    model = RefPixStep.call(model, refpix_algorithm="median", irs2_mean_subtraction=mean_subtraction)
    good = (model.pixeldq == 0)
    good[:BORDER] = good[-BORDER:] = False
    good[:, :BORDER] = good[:, -BORDER:] = False
    results["refpix"] = per_group_medians(model.data[0], good)
    results["shape_after_refpix"] = list(model.data.shape)
    print("refpix done", model.data.shape, flush=True)

    model = LinearityStep.call(model)
    results["linearity"] = per_group_medians(model.data[0], good)
    print("linearity done", flush=True)

    before = model.data[0].copy()
    step = DarkCurrentStep()
    dark_name = step.get_reference_file(model, "dark")
    results["dark_reference"] = dark_name
    model = step.run(model)
    results["dark_subtracted"] = per_group_medians(model.data[0], good)
    # what the step removed, group by group (the reference dark trimmed to the data's groups)
    removed = before - model.data[0]
    results["reference_dark"] = per_group_medians(removed, good)
    results["reference_dark_group0_median"] = float(np.median(removed[0][good]))
    print("dark done", dark_name, flush=True)

    with open(out_path, "w") as handle:
        json.dump(results, handle, indent=1)

    tgroup = model.meta.exposure.group_time
    print(f"\n{path}: median of data[k]-data[0] over usable science pixels, DN (group time {tgroup:.3f} s)")
    print(f"irs2_mean_subtraction={mean_subtraction}; raw columns: science, reference output, interleaved reference pixels")
    columns = ("raw", "raw_refout", "raw_interleaved", "refpix", "linearity", "dark_subtracted", "reference_dark")
    print(f"{'k':>3s} " + " ".join(f"{c:>15s}" for c in columns))
    for k in range(len(results["raw"]["all"])):
        print(f"{k:3d} " + " ".join(f"{results[c]['all'][k]:15.3f}" for c in columns))
    n = len(results["raw"]["all"]) - 1
    for stage in ("raw", "refpix", "linearity", "dark_subtracted", "reference_dark"):
        slope = results[stage]["all"][-1] / (n * tgroup)
        amps = [results[stage]["amps"][a][-1] / (n * tgroup) for a in range(4)]
        print(f"{stage:16s} slope over groups 0..{n}: {slope:+.5f} DN/s, amplifiers {[round(a, 5) for a in amps]}")
    print(f"reference dark group 0 median: {results['reference_dark_group0_median']:.3f} DN")


if __name__ == "__main__":
    main()

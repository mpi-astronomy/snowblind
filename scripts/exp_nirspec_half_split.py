"""Does the cleaning take signal out of the slits?  A split-ramp test free of selection on the noise.

Selecting pixels by their uncleaned rate and measuring what the cleaning
changed in them is biased: the bright bins hold the pixels whose 1/f
noise happened to be positive, and the cleaning rightly removes it, so
every cleaner looks as if it took a fraction of the signal.  Here the
ramp is split in two halves whose 1/f noise is independent (the filter
removes periods below a second; the halves are minutes apart): pixels are
binned by the uncleaned rate of the first half, and the change the
cleaning made is measured in the second half.  A cleaner that removes
signal shows a change that grows with the bin's signal; one that removes
only noise shows none.

Usage:
    python scripts/exp_nirspec_half_split.py JUMP_RAMP REGIONS_NPZ name=CLEANED_RAMP [name=CLEANED_RAMP ...]

The rate of each half is the jwst OLS_C fit of the half ramp (pixels
with a DQ flag in the fit are left out).  Pixels inside the slit or slice
footprints of ``REGIONS_NPZ`` (as ``make_nirspec_field_panels.py`` writes
it) are binned by the first half's uncleaned rate at the quantiles below,
and the slopes are fitted between its 1st and 99.9th percentiles.
"""
import os
import sys
import warnings

import numpy as np
from jwst.ramp_fitting import RampFitStep
from stdatamodels.jwst import datamodels

BINS = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999, 1.0]


def half_rates(path):
    """Per-pixel OLS_C rates (DN/s) of the first and second half of the groups of a ramp, cached next to it."""
    cache = path.replace(".fits", "_halves.npz")
    if os.path.exists(cache):
        loaded = np.load(cache)
        return loaded["first"], loaded["second"]
    halves = []
    with datamodels.open(path) as model:
        ngroups = model.data.shape[1]
        for groups in (slice(0, ngroups // 2), slice(ngroups // 2, ngroups)):
            half = datamodels.RampModel(np.ascontiguousarray(model.data[:, groups]))
            half.update(model)
            half.groupdq = np.ascontiguousarray(model.groupdq[:, groups])
            half.pixeldq = np.array(model.pixeldq)
            half.meta.exposure.ngroups = half.data.shape[1]
            rate, _ = RampFitStep.call(half, algorithm="OLS_C")
            slope = np.array(rate.data, dtype=np.float64)
            slope[rate.dq != 0] = np.nan
            halves.append(slope)
            rate.close()
            half.close()
    np.savez_compressed(cache, first=halves[0], second=halves[1])
    return halves[0], halves[1]


def main():
    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
    jump_path, regions_path = sys.argv[1], sys.argv[2]
    cleaned = dict(arg.split("=", 1) for arg in sys.argv[3:])
    regions = np.load(regions_path)["regions"]
    first, second = half_rates(jump_path)
    inside = regions & np.isfinite(first) & np.isfinite(second)
    edges = np.quantile(first[inside], BINS)
    print(f"{len(cleaned)} cleaned ramps; {int(inside.sum())} pixels inside the footprints; bins of the first half's rate (DN/s)")
    header = f"{'bin':>22s} {'pixels':>8s} {'2nd half':>9s}"
    changes = {}
    for name, path in cleaned.items():
        _, cleaned_second = half_rates(path)
        changes[name] = cleaned_second - second
        header += f" {name:>14s}"
    print(header)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = inside & (first >= lo) & (first < hi)
        line = f"{lo:+9.4f} to {hi:+9.4f} {int(sel.sum()):8d} {np.nanmedian(second[sel]):+9.4f}"
        for name in cleaned:
            values = changes[name][sel]
            values = values[np.isfinite(values)]
            line += f" {np.median(values) * 1e4:+8.2f}+-{1.25 * np.std(values) / np.sqrt(values.size) * 1e4:4.2f}"
        print(line + "   (median change x1e-4 DN/s, with its error)")
    low, high = np.quantile(first[inside], [0.01, 0.999])
    print(
        "slope of the second-half change against the first-half rate, "
        f"footprint pixels with first-half rate in [{low:.4f}, {high:.4f}]:"
    )
    for name in cleaned:
        ok = inside & np.isfinite(changes[name]) & (first >= low) & (first <= high)
        slope, intercept = np.polyfit(first[ok], changes[name][ok], 1)
        resid = changes[name][ok] - (slope * first[ok] + intercept)
        err = np.std(resid) / np.sqrt(np.sum((first[ok] - first[ok].mean()) ** 2))
        print(f"   {name:14s} {slope:+.5f} +- {err:.5f} (intercept {intercept * 1e4:+.2f}e-4 DN/s)")
    print("the same, selecting on the second half itself (the biased way), for comparison:")
    for name in cleaned:
        ok = inside & np.isfinite(changes[name]) & (second >= low) & (second <= high)
        slope, _ = np.polyfit(second[ok], changes[name][ok], 1)
        print(f"   {name:14s} {slope:+.5f}")


if __name__ == "__main__":
    main()

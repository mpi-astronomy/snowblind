"""How much the rates change when every integration takes the reset offsets measured on the first.

Each integration of an ``_uncal`` exposure is copied to a single-integration
file (as in ``measure_reset_offset_per_integration.py``) and run through
``snowblind_detector1`` with the ramp fit skipped, then fitted three ways:

- ``ols``: the jwst OLS_C fit, the reference;
- ``own``: `ResetRampFitStep` anchored on this integration's own sky;
- ``first``: `ResetRampFitStep` with the offsets anchored on integration 1,
  as snowblind 1.4.1 and earlier did for every integration.

Reported per integration: the offsets, the median of (reset fit - OLS_C) over
sky pixels (OLS_C rate within 5 robust sigma of its median), and, for pixels
whose first group is usable but whose second is saturated (their rate rests
on the reset point and the first group alone), the median of (first - own)
and of own.

Output: ``<exposure>.json`` in ``--out`` (default
``results/<date>_reset_offset_integration_impact/``).
"""
import argparse
import datetime
import json
import os

import numpy as np
from astropy.io import fits
from jwst import datamodels
from jwst.ramp_fitting import RampFitStep
from stdatamodels.jwst.datamodels import dqflags

import snowblind.reset_rampfit as reset_rampfit
from measure_reset_offset_per_integration import _write_integration
from snowblind import ResetRampFitStep
from snowblind.metrics import robust_sigma
from snowblind.pipeline import Detector1Pipeline

BAD = dqflags.pixel["DO_NOT_USE"] | dqflags.pixel["SATURATED"]
SATURATED = dqflags.pixel["SATURATED"]


def _with_channel_offsets(offsets):
    """A stand-in for default_calibration whose per-channel reset offsets are ``offsets``."""
    packaged = reset_rampfit.default_calibration

    def calibration(detector):
        tree = dict(packaged(detector))
        tree["reset_offset_channels"] = [float(b) for b in offsets]
        return tree

    return calibration


def _offsets(rate):
    """The reset offsets per channel a reset fit of a single-integration ramp recorded on its rate model."""
    return [float(b) for b in rate.meta.instance["snowblind"]["reset_rampfit"]["reset_offset"][0]]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file")
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--work", required=True, help="directory for the single-integration copies")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    name = os.path.basename(args.file).replace("_uncal.fits", "")
    out_dir = args.out or os.path.join("results", f"{datetime.date.today().isoformat()}_reset_offset_integration_impact")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.work, exist_ok=True)
    header = fits.getheader(args.file)
    summary = {"exposure": name, "exp_type": header["EXP_TYPE"], "readpatt": header["READPATT"], "integrations": []}

    first_offsets = None
    packaged = reset_rampfit.default_calibration
    for number in range(1, int(header["NINTS"]) + 1):
        copy = os.path.join(args.work, f"{name}_int{number:03d}_uncal.fits")
        groups = _write_integration(args.file, number - 1, args.max_groups, copy)
        with datamodels.RampModel(copy) as model:
            ramp = Detector1Pipeline.call(model, steps={"ramp_fit": {"skip": True}})
        os.remove(copy)

        ols = np.array(RampFitStep.call(ramp, algorithm="OLS_C")[0].data, dtype=np.float64)
        own_model = ResetRampFitStep.call(ramp)[0]
        own_offsets = _offsets(own_model)
        if first_offsets is None:
            first_offsets = own_offsets
        reset_rampfit.default_calibration = _with_channel_offsets(first_offsets)
        first_model = ResetRampFitStep.call(ramp, anchor_offset_to_sky=False)[0]
        reset_rampfit.default_calibration = packaged

        own = np.array(own_model.data, dtype=np.float64)
        first = np.array(first_model.data, dtype=np.float64)
        good = ((own_model.dq & BAD) == 0) & ((first_model.dq & BAD) == 0) & np.isfinite(own) & np.isfinite(first)
        reference = good & np.isfinite(ols)
        center, spread = float(np.median(ols[reference])), robust_sigma(ols[reference])
        sky = reference & (np.abs(ols - center) < 5 * spread)

        groupdq = ramp.groupdq[0]
        first_group_only = (
            ((groupdq[0] & SATURATED) == 0)
            & ((groupdq[1] & SATURATED) > 0)
            & ((ramp.pixeldq & dqflags.pixel["DO_NOT_USE"]) == 0)
            & np.isfinite(own)
            & np.isfinite(first)
        )
        entry = {
            "integration": number,
            "groups_used": groups,
            "own_offsets": own_offsets,
            "first_offsets": first_offsets,
            "sky_rate": center,
            "sky_own_minus_ols": float(np.median((own - ols)[sky])),
            "sky_first_minus_ols": float(np.median((first - ols)[sky])),
            "first_group_only_pixels": int(first_group_only.sum()),
        }
        if first_group_only.any():
            entry["first_group_only_first_minus_own"] = float(np.median((first - own)[first_group_only]))
            entry["first_group_only_own_rate"] = float(np.median(own[first_group_only]))
        summary["integrations"].append(entry)
        del ramp, own_model, first_model
        print(
            f"{name} int {number}: offset own {np.mean(own_offsets):+.2f} first {np.mean(first_offsets):+.2f} DN; "
            f"sky own-ols {entry['sky_own_minus_ols']:+.4f} first-ols {entry['sky_first_minus_ols']:+.4f} DN/s; "
            f"first-group-only pixels {entry['first_group_only_pixels']}"
            + (
                f", first-own {entry['first_group_only_first_minus_own']:+.3f} DN/s "
                f"at {entry['first_group_only_own_rate']:.0f} DN/s"
                if first_group_only.any() else ""
            ),
            flush=True,
        )
        with open(os.path.join(out_dir, f"{name}.json"), "w") as handle:
            json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()

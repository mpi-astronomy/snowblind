"""The reset offset of each integration and output channel, measured as `ResetRampFitStep` measures it.

Each integration of an ``_uncal`` exposure is copied to a single-integration
``_uncal`` file of its own (a segment, INTSTART = INTEND, at most
``--max-groups`` groups) through a memory map, so the exposure is never
loaded whole, and run through ``snowblind_detector1`` with the ramp fit
skipped.  The offset is then measured with the step's own functions
(``_first_resultant`` and ``sky_anchored_offsets``: the median over faint
unflagged pixels of the first group minus the signal the later groups predict
there), with the first-read scales of the packaged calibration, the way the
calibration's own offset was measured.  The same median on the two halves of
a checkerboard of pixels gives the statistical error: half the difference of
the two halves.

Output: ``<exposure>.json`` in ``--out`` (default
``results/<date>_reset_offset_per_integration/``), with the packaged
calibration's per-channel offsets alongside.
"""
import argparse
import datetime
import json
import os

import numpy as np
from astropy.io import fits
from jwst import datamodels

import snowblind.reset_rampfit as reset_rampfit
from snowblind import readout
from snowblind.pipeline import Detector1Pipeline
from snowblind.reset_calib import default_calibration

STRUCTURAL = ("BZERO", "BSCALE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "NAXIS4", "EXTEND")


def _write_integration(path, integ, max_groups, out_path):
    """Copy one integration (and at most ``max_groups`` groups) of an uncal file to a segment file of its own."""
    # astropy does not scale memory-mapped data, so BZERO and BSCALE are applied here
    with fits.open(path, memmap=True, do_not_scale_image_data=True) as hdul:
        primary = hdul[0].header.copy()
        keep = int(primary["NGROUPS"]) if max_groups is None else min(int(primary["NGROUPS"]), max_groups)
        sci_header = hdul["SCI"].header.copy()
        raw = np.asarray(hdul["SCI"].section[integ:integ + 1, :keep])
        data = raw.astype(np.int64) * int(sci_header.get("BSCALE", 1)) + int(sci_header.get("BZERO", 0))
        for key in STRUCTURAL:
            sci_header.remove(key, ignore_missing=True)
        primary["NGROUPS"] = keep
        primary["INTSTART"] = integ + 1
        primary["INTEND"] = integ + 1
        hdus = [fits.PrimaryHDU(header=primary), fits.ImageHDU(data.astype(np.uint16), header=sci_header, name="SCI")]
        if "INT_TIMES" in hdul:
            rows = hdul["INT_TIMES"].data
            hdus.append(fits.BinTableHDU(rows[rows["integration_number"] == integ + 1], name="INT_TIMES"))
        fits.HDUList(hdus).writeto(out_path, overwrite=True)
    return keep


def _offsets(ramp, t1_scales):
    """Per channel: the step's anchored offset, the checkerboard half-difference and the sky pixel count."""
    first, t_first, clean, rest, t1 = reset_rampfit._first_resultant(ramp, False)
    blocks = readout.channel_blocks(first.shape, ramp.meta.subarray.fastaxis, ramp.meta.subarray.slowaxis)
    offsets = reset_rampfit.sky_anchored_offsets(ramp, 0.0, t1_scales, blocks, False, resultant=(first, t_first, clean, rest, t1))
    rows, cols = np.indices(first.shape)
    checker = (rows + cols) % 2 == 0
    out = []
    for (r, c), k, offset in zip(blocks, t1_scales, offsets):
        signal = rest[r, c] * (t_first - (1.0 - k) * t1)
        faint = clean[r, c] & (signal < 50.0) & (signal > -20.0)
        residual = first[r, c] - signal
        halves = [np.median(residual[faint & checker[r, c]]), np.median(residual[faint & ~checker[r, c]])]
        out.append({
            "offset": float(offset),
            "half_difference": float(abs(halves[0] - halves[1]) / 2),
            "sky_pixels": int(faint.sum()),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file")
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--integrations", type=int, nargs="*", default=None, help="1-based integration numbers; default all")
    parser.add_argument("--work", required=True, help="directory for the single-integration copies")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    name = os.path.basename(args.file).replace("_uncal.fits", "")
    out_dir = args.out or os.path.join("results", f"{datetime.date.today().isoformat()}_reset_offset_per_integration")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.work, exist_ok=True)
    header = fits.getheader(args.file)
    nints = int(header["NINTS"])
    integrations = args.integrations or list(range(1, nints + 1))

    tree = default_calibration(header["DETECTOR"])
    summary = {
        "exposure": name,
        "exp_type": header["EXP_TYPE"],
        "readpatt": header["READPATT"],
        "date_obs": header["DATE-OBS"],
        "nints": nints,
        "ngroups": int(header["NGROUPS"]),
        "calibration_offsets": [float(b) for b in tree["reset_offset_channels"]],
        "calibration_provenance": {
            "b": tree.get("timing_provenance", {}).get("file"),
            "date": tree.get("timing_provenance", {}).get("date"),
        },
        "integrations": [],
    }
    for number in integrations:
        copy = os.path.join(args.work, f"{name}_int{number:03d}_uncal.fits")
        groups = _write_integration(args.file, number - 1, args.max_groups, copy)
        # the copy has no ASDF extension, so it goes in as a model whose metadata come from its FITS keywords
        with datamodels.RampModel(copy) as model:
            ramp = Detector1Pipeline.call(model, steps={"ramp_fit": {"skip": True}})
        t1 = reset_rampfit.first_group_time(ramp)
        scales = [reset_rampfit.rescale_t1(k, float(tree["t1_seconds"]), t1) for k in tree["t1_scale_channels"]]
        channels = _offsets(ramp, scales)
        summary["integrations"].append({"integration": number, "groups_used": groups, "t1_scales": scales, "channels": channels})
        del ramp
        os.remove(copy)
        values = "  ".join(f"{c['offset']:+.2f} (+-{c['half_difference']:.2f})" for c in channels)
        print(f"{name} int {number}: {values}", flush=True)
        with open(os.path.join(out_dir, f"{name}.json"), "w") as handle:
            json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()

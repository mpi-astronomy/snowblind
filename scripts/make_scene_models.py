"""Turn the blotted median mosaics of an Image3 run into scene models for ``stream_clean``.

The blots are in the units of the cal files they were made from.  With
photometric calibration skipped in Image2 that is DN/s after flat
fielding, so the scene before flat fielding, which is what the group
data hold, is the blot times the flat reference used in Image2.  Pixels
where the flat is unusable or the blot has no coverage are NaN.
"""
import argparse
import glob
import os
import re

import numpy as np
from astropy.io import fits
from jwst.flatfield import FlatFieldStep
from stdatamodels.jwst import datamodels


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pass_dir", help="directory of an Image3 run with *_blot.fits and *_cal.fits")
    parser.add_argument("--out", default=None, help="where to write <root>_scene.fits (default: pass_dir)")
    args = parser.parse_args()
    out = args.out or args.pass_dir
    os.makedirs(out, exist_ok=True)

    step = FlatFieldStep()
    for blot_path in sorted(glob.glob(os.path.join(args.pass_dir, "*_blot.fits"))):
        # the blot carries the association id, e.g. <root>_a3001_blot.fits; the cal and scene use the exposure root
        root = re.sub(r"_a\d{4}$", "", os.path.basename(blot_path).replace("_blot.fits", ""))
        cal_path = os.path.join(args.pass_dir, f"{root}_cal.fits")
        with datamodels.open(cal_path) as cal:
            flat_path = step.get_reference_file(cal, "flat")
            with datamodels.FlatModel(flat_path) as flat_model:
                flat = np.array(flat_model.data, dtype=np.float32)
                flat_bad = (flat_model.dq & 1) > 0
        with datamodels.open(blot_path) as blot:
            scene = np.array(blot.data, dtype=np.float32) * flat
        scene[flat_bad | ~np.isfinite(flat) | (flat == 0)] = np.nan
        scene_path = os.path.join(out, f"{root}_scene.fits")
        header = fits.Header()
        header["BUNIT"] = "DN/s"
        header["SCENESRC"] = (os.path.basename(blot_path), "blotted median mosaic")
        header["FLATREF"] = (os.path.basename(flat_path), "flat multiplied back in")
        fits.HDUList([fits.PrimaryHDU(header=header), fits.ImageHDU(scene, name="SCI")]).writeto(scene_path, overwrite=True)
        print(f"{scene_path}: median {np.nanmedian(scene):.4f} DN/s, {100 * np.mean(~np.isfinite(scene)):.2f} percent NaN")


if __name__ == "__main__":
    main()

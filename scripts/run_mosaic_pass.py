"""One pass of the mosaic scene-model loop on the dithered exposures of one detector.

For every ``_uncal`` file: Detector1 to jump with SIRS, ``stream_clean``
(scalar median background, or the given scene models), ``reset_rampfit``,
saved as ``_rate.fits``; then Image2 without photometric calibration (so
the cal files stay in DN/s, flat-fielded) and without resampling; then
Image3 on all of them with tweakreg and skymatch skipped, the source
catalog skipped, and the outlier-detection step saving its intermediate
products, which include the median mosaic blotted back into each
exposure's frame (``_blot.fits``).  Those blots are the scene models for
the next pass.
"""
import argparse
import glob
import os

from jwst.associations.asn_from_list import asn_from_list
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base
from jwst.pipeline import Image2Pipeline, Image3Pipeline

from snowblind import ResetRampFitStep, StreamCleanStep
from snowblind.reset_calib import run_detector1_to_jump


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="_uncal files of one detector")
    parser.add_argument("--calib", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene-dir", default=None, help="directory with <root>_scene.fits models from the previous pass")
    parser.add_argument("--product", default="mosaic")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rates = []
    for path in args.files:
        root = os.path.basename(path).replace("_uncal.fits", "")
        rate_path = os.path.join(args.out, f"{root}_rate.fits")
        if not os.path.exists(rate_path):
            jump = run_detector1_to_jump(path, refpix_algorithm="sirs")
            kwargs = {"background_method": "median"}
            if args.scene_dir is not None:
                kwargs = {"scene_model": os.path.join(args.scene_dir, f"{root}_scene.fits")}
            cleaned = StreamCleanStep.call(jump, reset_calib=args.calib, save_mask=True, output_dir=args.out, **kwargs)
            rate, _ = ResetRampFitStep.call(cleaned, reset_calib=args.calib, use_pedestal_mean=True)
            rate.save(rate_path)
            rate.close()
            cleaned.close()
            jump.close()
        rates.append(rate_path)

    cals = []
    for rate_path in rates:
        cal_path = rate_path.replace("_rate.fits", "_cal.fits")
        if not os.path.exists(cal_path):
            Image2Pipeline.call(
                rate_path, steps={"photom": {"skip": True}, "resample": {"skip": True}},
                save_results=True, output_dir=args.out,
            )
        cals.append(cal_path)

    # members are resolved relative to the association file, which sits next to them
    asn = asn_from_list([os.path.basename(c) for c in cals], rule=DMS_Level3_Base, product_name=args.product)
    asn_path = os.path.join(args.out, f"{args.product}_asn.json")
    with open(asn_path, "w") as handle:
        _name, serialized = asn.dump(format="json")
        handle.write(serialized)
    Image3Pipeline.call(
        asn_path,
        steps={
            "tweakreg": {"skip": True},
            "skymatch": {"skip": True},
            "outlier_detection": {"save_intermediate_results": True},
            "source_catalog": {"skip": True},
        },
        save_results=True, output_dir=args.out,
    )
    print("blots:", sorted(glob.glob(os.path.join(args.out, "*blot*"))))


if __name__ == "__main__":
    main()

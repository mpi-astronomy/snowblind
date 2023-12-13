Algorithms for cleaning JWST data.

 - `SnowblindStep`: mask cosmic ray showers and snowballs
 - `JumpPlusStep`: Propagate JUMP_DET and SATURATED flags in GROUPDQ properly for frame-averaged groups
 - `PersistenceFlagStep`: flag pixels effected by persistence exposure-to-exposure
 - `RcSelfCalStep`: flag new hot pixels, open pixels or RC pixels


## Installation


    pip install snowblind


## Usage

The steps in snowblind run like any other pipeline steps.  From the command line:

    strun snowblind jw001234_010203_00001_nrcalong_jump.fits --suffix=snowblind

In Python:

    from snowblind import SnowblindStep
    from jwst.pipeline import Detector1Pipeline
    from jwst.step import RampFitStep
    from jwst.step import GainScaleStep


    steps = {
        "jump": {
            "save_results": True,
        },
        "ramp_fit": {
            "skip": True,
        },
        "gain_scale": {
            "skip": True,
        },
    }

    Detector1Pipeline.call("jw001234_010203_00001_nrcalong_uncal.fits", steps=steps)
    SnowblindStep.call("jw001234_010203_00001_nrcalong_jump.fits", save_results=True, suffix="snowblind")
    rate, rateints = RampFitStep.call("jw001234_010203_00001_nrcalong_snowblind.fits")
    rate = GainScaleStep.call(rate)
    rate.save(rate.meta.filename.replace("snowblind", "rate"))

More to come on the other steps available.

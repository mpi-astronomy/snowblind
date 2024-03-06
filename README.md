Algorithms for cleaning JWST data.

 - `SnowblindStep`: mask cosmic ray showers and snowballs
 - `JumpPlusStep`: Propagate JUMP_DET and SATURATED flags in GROUPDQ properly for frame-averaged groups
 - `PersistenceFlagStep`: flag pixels effected by persistence exposure-to-exposure
 - `OpenPixelStep`: flag new open pixels, hot pixels, or open adjacent pixels


## Installation


    pip install snowblind


## Usage

The steps in snowblind run like any other pipeline steps.  From the command line you can run the step on the result file from JumpStep:

    strun snowblind jw001234_010203_00001_nrcalong_jump.fits --suffix=snowblind

Or you can run it as a post-hook in a full pipeline

    strun calwebb_detector1 jw001234_010203_00001_nrcalong_uncal.fits --steps.jump.post_hooks="snowblind.SnowblindStep","snowblind.JumpPlusStep"

In Python, we can insert `SnowblindStep` and `JumpPlusStep` after `JumpStep` as a post-hook:

    from snowblind import SnowblindStep, JumpPlusStep
    from jwst.pipeline import Detector1Pipeline


    steps = {
        "jump": {
            "save_results": True,
            "flag_large_events": False,
            "post_hooks": [
                "snowblind.SnowblindStep",
                "snowblind.JumpPlusStep",
            ],
        },
    }

    Detector1Pipeline.call("jw001234_010203_00001_nrcalong_uncal.fits", steps=steps, save_results=True)

More to come on the other steps available.

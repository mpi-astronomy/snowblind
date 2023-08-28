Mask cosmic ray showers and snowballs in JWST data


## Installation


    pip install git+https://github.com/jdavies-st/snowblind


## Usage

The steps in snowblind run like any other pipeline steps.  From the command line:

    $ strun snowblind jw001234_010203_00001_nrcalong_jump.fits --suffix=snowblind

or from within python:

    from snowblind import SnowblindStep
    from jwst.pipeline import Detector1Pipeline
    from jwst.ramp_fitting import RampFitStep


    steps = {
        "jump": {
            "save_results": True,
        }
        "ramp_fit": {
            "skip": True
        }
    }

    Detector1Pipeline.call("jw001234_010203_00001_nrcalong_uncal.fits", steps=steps)
    SnowblindStep.call("jw001234_010203_00001_nrcalong_jump.fits", save_results=True, suffix="snowblind")
    rate, rateints = RampFitStep.call("jw001234_010203_00001_nrcalong_snowblind.fits")
    rate.save(cal.meta.filename.replace("snowblind", "rate"))

More to come.

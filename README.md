Algorithms for cleaning JWST data.

 - `SnowblindStep`: mask cosmic ray showers and snowballs
 - `JumpPlusStep`: Propagate JUMP_DET and SATURATED flags in GROUPDQ properly for frame-averaged groups
 - `PersistenceFlagStep`: flag pixels effected by persistence exposure-to-exposure
 - `OpenPixelStep`: flag new open pixels, hot pixels, or open adjacent pixels via self-cal
 - `StreamCleanStep`: remove 1/f (flicker) noise from NIR group data by filtering each output channel's readout stream
 - `ResetRampFitStep`: likelihood ramp fitting with the superbias-subtracted reset level as an extra resultant, so that pixels with a single unsaturated group still get a rate


# Installation


    pip install snowblind


# Usage

## SnowblindStep and JumpPlusStep

The steps in snowblind run like any other pipeline steps.  From the command line you can run `SnowblindStep` (aliased as `snowblind`) on the result file from JumpStep:

    strun snowblind jw001234_010203_00001_nrcalong_jump.fits --suffix=snowblind

Or you can run `SnowblindStep` and `JumpPlusStep` as post-hooks after `JumpStep` in a full pipeline, remembering to turn off the default snowball flagging.

    strun calwebb_detector1 jw001234_010203_00001_nrcalong_uncal.fits --steps.jump.post_hooks="snowblind.SnowblindStep","snowblind.JumpPlusStep" --steps.jump.flag_large_events=False

In Python, we can insert `SnowblindStep` and `JumpPlusStep` after `JumpStep` as a post-hook:

```python
from snowblind import SnowblindStep, JumpPlusStep
from jwst.pipeline import Detector1Pipeline

steps = {
    "jump": {
        "save_results": True,
        "flag_large_events": False,
        "post_hooks": [
            SnowblindStep,
            JumpPlusStep,
        ],
    },
}

Detector1Pipeline.call("jw001234_010203_00001_nrcalong_uncal.fits", steps=steps, save_results=True)
```

## PersistenceFlagStep and OpenPixelStep

The steps `PersistenceFlagStep` and `OpenPixelStep` need to be run on an association of _rate or _cal files, because they are essentially self-calibration.  Here's an example for `PersistenceFlagStep`:

Run `Detector1Pipeline` on each input _uncal.fits file, skipping the built-in `PersistenceStep` and saving the output of `JumpStep`.  We will need the output later.
```python
from glob import glob
from jwst.pipeline import Detector1Pipeline


steps = dict(
    persistence=dict(
        skip=True,
    ),
    jump=dict(
        save_results=True,
        rejection_threshold=5.0,
    ),
)

uncal_files = glob("jw*_uncal.fits")
uncal_files.sort()
for file in uncal_files:
    Detector1Pipeline.call(file, steps=steps)
```

And then run `Image2Pipeline` to produce all the _cal.fits files.  Afterwards, to flag persistence in these:

```python
from glob import glob
from jwst.datamodels import ModelContainer
from jwst.steps import PersistenceFlagStep


cal_files = glob("jw*_cal.fits")
cal_files.sort()
combined_asn = ModelContainer(cal_files)
PersistenceFlagStep.call(combined_asn, save_results=True, suffix="cal")
```

which will overwrite _cal.fits files with the same file but with a new DQ array with PERSISTENCE and DO_NOT_USE flag set.  If you don't want to overwrite and will rename later, just don't use the `suffix` arg.

Finally, both these steps can be inserted as pre-hooks into the `Image3Pipeline` using the same method as shown above with `SnowblindStep`.

## StreamCleanStep and ResetRampFitStep

Both steps work on the `_jump` product and need the reference pixel step run with the SIRS algorithm:

    strun calwebb_detector1 jw..._uncal.fits --steps.refpix.refpix_algorithm=sirs --steps.jump.save_results=True
    strun stream_clean jw..._jump.fits --fit_method=wiener --save_spectra=True

`ResetRampFitStep` needs the reset-level scatter of the detector, measured once from dark ramps:

    snowblind-reset-calib jw..._nrca1_uncal.fits -o nrca1_reset_calib.asdf
    strun reset_rampfit jw..._nrca1_jump.fits --reset_calib=nrca1_reset_calib.asdf

The design, the measurements and the open questions are in `PLAN-1f-cleaning-and-reset-anchored-rampfit.md`.

Please open an issue if you have any problems!

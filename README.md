# Algorithms for cleaning JWST data

 - `SnowblindStep`: mask cosmic ray showers and snowball halos by growing the actual shape of each contiguous jump-detected region and the ring around its saturated core; works on ramps, rate images and rateints cubes, so it can be applied after the fact; replaces `expand_large_events` in the standard JWST pipeline jump step
 - `JumpPlusStep`: for frame-averaged readouts (NFRAMES > 1), flag the group after a detected jump, which the cosmic ray also corrupts and which the pipeline's two-point jump detection leaves unflagged (though using the ramp fit likelyhood algorithm accounts for this correctly itself); replaces jump step's `after_jump_flag` mechanism; also flag the group before a saturated group as saturated, a conservative superset of the read-pattern check the pipeline's saturation step
 - `PersistenceFlagStep`: flag pixels that saturated in an earlier exposure of the association for a chosen time afterwards
 - `OpenPixelStep`: find open, hot and cross-shaped open-adjacent pixels that have appeared since the reference bad-pixel mask was made, by self-calibration against the other exposures of the same detector.  Use with caution.  Needs tuning.
 - `StreamCleanStep`: remove 1/f (flicker) noise, the striping, from NIR group data with a Wiener filter along each output channel's readout stream, its gain taken from the exposure's own noise spectrum; the same linear filter on every group difference, so, unlike the per-group medians of the pipeline's `clean_flicker_noise` step, it adds no noise along the ramp and improves with the number of groups; also removes the per-read even-odd and column offsets
 - `ResetRampFitStep`: likelihood ramp fitting as in the standard JWST pipeline, but with the superbias-subtracted reset level as an extra resultant at the beginning, so that pixels with a single unsaturated group or only a ZEROFRAME still get a rate; needs a per-detector calibration from dark ramps, made once with `snowblind-reset-calib`


# Installation


    pip install snowblind


# Usage

## SnowblindStep and JumpPlusStep

The steps in snowblind run like any other pipeline steps.  From the command line you can run `SnowblindStep` (aliased as `snowblind`) to remove the halos from high-energy CR hits, i.e. "snowballs", on the result file from JumpStep:

    strun snowblind jw001234_010203_00001_nrcalong_jump.fits --suffix=snowblind

Or you can run `SnowblindStep` and/or `JumpPlusStep` as post-hooks after `JumpStep` in a full pipeline, remembering to turn off the default snowball flagging.

    strun calwebb_detector1 jw001234_010203_00001_nrcalong_uncal.fits --steps.jump.post_hooks="snowblind.SnowblindStep","snowblind.JumpPlusStep" --steps.jump.flag_large_events=False

In Python, we can insert `SnowblindStep` and `JumpPlusStep` after `JumpStep` as a post-hook too:

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

which runs them with their step default parameters.  If you want more flexibility, run each step independently with its individual parameters.

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

Both steps build on published work:
- the SIRS reference-pixel correction ([Rauscher, Fixsen and Mosby 2022, JATIS 8, 028002](https://arxiv.org/abs/2602.02253))
- the CEERS team's masked row and column striping removal ([Bagley et al. 2023, ApJL 946, L12](https://doi.org/10.3847/2041-8213/acbb08))
- the likelihood ramp fitting of Brandt (2024, PASP, https://doi.org/10.1088/1538-3873/ad38d9 and https://doi.org/10.1088/1538-3873/ad38da) as implemented in `stcal`
- the scene mask follows NoiseChisel ([Akhlaghi and Ichikawa 2015, ApJS 220](https://doi.org/10.1088/0067-0049/220/1/1))
- comparisons are against the jwst `clean_flicker_noise` step, which descends from Chris Willott's image1overf (https://github.com/chriswillott/jwst) and from NSClean ([Rauscher 2024, PASP 136, 015001](https://ui.adsabs.harvard.edu/abs/2024PASP..136a5001R/abstract)).

Please open an issue if you have any problems!

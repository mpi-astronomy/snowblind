# Algorithms for cleaning JWST data

 - `SnowblindStep`: mask cosmic ray showers and snowball halos by growing the actual shape of each contiguous jump-detected region and the ring around its saturated core; works on ramps, rate images and rateints cubes, so it can be applied after the fact; replaces `expand_large_events` in the standard JWST pipeline jump step
 - `JumpPlusStep`: for frame-averaged readouts (NFRAMES > 1), flag the group after a detected jump, which the cosmic ray also corrupts and which the pipeline's two-point jump detection leaves unflagged (though using the ramp fit likelyhood algorithm accounts for this correctly itself); replaces jump step's `after_jump_flag` mechanism; also flag the group before a saturated group as saturated, a conservative superset of the read-pattern check the pipeline's saturation step
 - `PersistenceFlagStep`: flag pixels that saturated in an earlier exposure of the association for a chosen time afterwards
 - `OpenPixelStep`: find open, hot and cross-shaped open-adjacent pixels that have appeared since the reference bad-pixel mask was made, by self-calibration against the other exposures of the same detector.  Use with caution.  Needs tuning.
 - `StreamCleanStep`: remove 1/f (flicker) noise, the striping, from NIR group data with a Wiener filter along each output channel's readout stream, its gain taken from the exposure's own noise spectrum; the same linear filter on every group difference, so, unlike the per-group medians of the pipeline's `clean_flicker_noise` step, it adds no noise along the ramp and improves with the number of groups; also removes the per-read even-odd and column offsets
 - `ResetRampFitStep`: likelihood ramp fitting as in the standard JWST pipeline, but with the superbias-subtracted reset level as an extra resultant at the beginning, so that pixels with a single unsaturated group or only a ZEROFRAME still get a rate; needs a per-detector calibration from dark ramps, made once with `snowblind-reset-calib`


# Citing snowblind

Please cite `snowblind` through its Zenodo record DOI,
[10.5281/zenodo.15791614](https://doi.org/10.5281/zenodo.15791614), that
always points to the latest release.

    @software{snowblind,
      author    = {Davies, James E.},
      title     = {snowblind: algorithms for cleaning JWST data},
      publisher = {Zenodo},
      doi       = {10.5281/zenodo.15791614},
      url       = {https://github.com/mpi-astronomy/snowblind}
    }

The same information is in `CITATION.cff`.

# Installation

snowblind needs jwst 2.0.0 or later.  Earlier releases of snowblind work
with earlier jwst releases.

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

Both steps work on group data after the jump step, and want the reference
pixel step run with the SIRS algorithm.  Their defaults are set for
sky-limited imaging: the background of each group difference is a scalar
median measured on the flat-fielded difference, sources come from a
two-tier scene mask that also follows the column spikes of saturated stars,
and the reset fit uses the reset calibration packaged for every NIRCam
detector and NIRISS, with the reset offset anchored to the exposure's own
sky.

The simplest way to run both is the `snowblind_detector1` pipeline: it is
`calwebb_detector1` with the same steps, parameters and CRDS parameter
references, except that `StreamCleanStep` runs as a post-hook of the jump
step, the pipeline's own `clean_flicker_noise` step is skipped, and
`ResetRampFitStep` sits in the `ramp_fit` slot, so one run gives `_rate`
and `_rateints` products from the cleaned ramp and the reset-anchored fit:

    strun snowblind_detector1 jw..._uncal.fits --steps.refpix.refpix_algorithm=sirs

Parameters of the reset fit go through the slot
(`--steps.ramp_fit.reset_calib=my_calib.asdf`); parameters of the stream
clean go through the hook
(`--steps.jump.post_hooks="snowblind.StreamCleanStep(background_method='median')",`,
the trailing comma making `post_hooks` a list of one).  Either
substitution can be switched off, `--stream_clean=False` or
`--reset_fit=False`, which gives the jwst step back.

The stock pipeline can also be made to do the same with hooks alone.  The
stream clean is a post-hook of the jump step, and the reset fit is a
post-hook of a skipped `ramp_fit` (stpipe runs a step's post-hooks even
when the step is skipped, and takes the hook's return value as the data
that goes on), in its `as_hook` form, which returns the rate image alone
because a hook must return one model, and writes the `_rateints` product
itself:

    strun calwebb_detector1 jw..._uncal.fits --steps.refpix.refpix_algorithm=sirs \
        --steps.jump.post_hooks="snowblind.StreamCleanStep", --steps.clean_flicker_noise.skip=True \
        --steps.ramp_fit.skip=True --steps.ramp_fit.post_hooks="snowblind.ResetRampFitStep(as_hook=True)",

On the command line a hook string can carry one keyword argument only,
since the hook list is split on commas; in Python a hook can be a step
instance with any arguments, `ResetRampFitStep(as_hook=True, reset_calib=...)`.

A pre-hook of `ramp_fit` would also run before the skip, but the skipped
step then marks its output as skipped and the pipeline names it `_ramp`;
the post-hook form sets the ramp-fit record itself, so the product is
`_rate`.  For the stream clean alone, leave `ramp_fit` as it is:

    strun calwebb_detector1 jw..._uncal.fits --steps.refpix.refpix_algorithm=sirs --steps.jump.post_hooks="snowblind.StreamCleanStep", --steps.clean_flicker_noise.skip=True

The same in Python, where a hook can be the step class itself:

```python
from snowblind import Detector1Pipeline, StreamCleanStep

Detector1Pipeline.call(
    "jw..._uncal.fits",
    steps={"refpix": {"refpix_algorithm": "sirs"}, "jump": {"post_hooks": [StreamCleanStep]}},
    save_results=True,
)
```

Both steps also run on their own on a `_jump` product (`strun stream_clean
jw..._jump.fits`, `strun reset_rampfit jw..._jump.fits`).

The packaged calibration holds, per detector, the reset-level scatter
measured from dark ramps and the first-read timing measured on a field (see
the documentation for the measurements).  To measure your own, for example
for a detector or epoch not covered:

    snowblind-reset-calib jw..._nrca1_uncal.fits -o nrca1_reset_calib.asdf
    strun reset_rampfit jw..._nrca1_jump.fits --reset_calib=nrca1_reset_calib.asdf

The file also carries the detector's mean reset-level map, which
`reset_rampfit --use_pedestal_mean=True` uses as the prior; give the same
file to `stream_clean --reset_calib=...` so that the first read is cleaned
against the same prior.  On dark exposures set
`--steps.stream_clean.background_method=None`; for fields with extended
emission larger than a few hundred pixels, `integration_model` (a
box-spline background) keeps that structure out of the noise estimate at
the cost of the 1/f slower than its box, and the step warns when the
unmasked background under the median has that much structure.  In one
exposure, extended emission and slow 1/f cannot be told apart; a scene
model from the other dithers (`scene_model`) is the way to do better.

Dispersed (WFSS) exposures have a smooth dispersed background that no
scalar or box background can model.  On a WFSS exposure the step
therefore defaults to `background_method='crds_background'`: it takes the
CRDS background reference for the grism and filter, scales it once to the
exposure on the masked background pixels, and subtracts it from every
group difference before the noise fit, so the same fixed background
applies to every difference and nothing random-walks along the ramp.  The
ordinary pipeline recipe applies unchanged:

    strun calwebb_detector1 jw..._uncal.fits --steps.refpix.refpix_algorithm=sirs --steps.jump.post_hooks="snowblind.StreamCleanStep", --steps.clean_flicker_noise.skip=True

This is for grisms that disperse across the readout stream, NIRISS GR150C
and NIRCam GRISMC.  NIRISS GR150R and NIRCam GRISMR disperse along the
stream, where a trace's continuum shape and emission lines sit at the
scales the filter treats as noise, so the step skips those exposures with
a warning; `allow_dispersion_along_stream=True` overrides that, at the
cost of the spectral shape of faint traces.

Both steps build on published work:
- the SIRS reference-pixel correction ([Rauscher, Fixsen and Mosby 2022, JATIS 8, 028002](https://arxiv.org/abs/2602.02253))
- the CEERS team's masked row and column striping removal ([Bagley et al. 2023, ApJL 946, L12](https://doi.org/10.3847/2041-8213/acbb08))
- the likelihood ramp fitting of Brandt (2024, PASP, https://doi.org/10.1088/1538-3873/ad38d9 and https://doi.org/10.1088/1538-3873/ad38da) as implemented in `stcal`
- the scene mask follows NoiseChisel ([Akhlaghi and Ichikawa 2015, ApJS 220](https://doi.org/10.1088/0067-0049/220/1/1))
- comparisons are against the jwst `clean_flicker_noise` step, which descends from Chris Willott's image1overf (https://github.com/chriswillott/jwst) and from NSClean ([Rauscher 2024, PASP 136, 015001](https://ui.adsabs.harvard.edu/abs/2024PASP..136a5001R/abstract)).

Please open an issue if you have any problems!

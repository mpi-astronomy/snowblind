# Changelog

## [Unreleased]

### Added

- `StreamCleanStep` accepts `fit_method=None`, no stream filter and only
  the even-odd and column offsets, and `fit_method='auto'`, now the default:
  `'wiener'`, except `'none'` on a grism that disperses along the readout
  stream.
- `SparseResampleStep` (`sparse_resample`): the jwst resample step for
  images with drizzle replaced by spizzle, which builds each exposure's
  square-kernel weights once as a sparse matrix and sums in double
  precision. The variance planes are the exact variance of each output
  pixel; the noise factor, what the correlation between neighboring output
  pixels adds to the noise of a sum over many of them, is computed in a
  second pass over the exposures and saved next to the mosaic as
  `<product>_nfactor.fits` (header card `NFACFILE`); ERR is the root of the
  summed variance planes times the noise factor. `variance="stcal"` gives
  the jwst step's variance planes and ERR instead (header card `VARTYPE`).
  There is no `kernel` (square only) or `propagate_dq`; `weight_type` is
  `ivm` unless given, as the step does not read the jwst step's parameters
  from CRDS; the context array is made only with `enable_ctx`; the pixel
  map is interpolated with splines between the pixels where the WCS is
  evaluated (`pixmap_stepsize` 16, `pixmap_order` 3).
  `snowblind.resample.SparseResampleImage` is the class behind it, jwst's
  `ResampleImage` with stcal's drizzle calls replaced.
- `SparseOutlierDetectionStep` (`sparse_outlier_detection`): the jwst
  outlier detection for images, its method unchanged, drizzling with
  `SparseResampleImage` and blotting with spizzle. Images only; no
  `resample_data`.
- None of these steps has jwst's `good_bits`: each names the DQ flags of
  the pixels it leaves out (resampling: DO_NOT_USE and NON_SCIENCE; the
  outlier steps: DO_NOT_USE), through `snowblind.resample.drizzle_weights`,
  which gives stcal's weights from them.
- `TiledOutlierStep` (`tiled_outlier`): outlier detection for images
  against one reference built from the input pixels of all the exposures on
  the outlier mosaic's grid sampled twice as finely (pixfrac 0.8, weighted
  by overlap and inverse read-noise variance, weighted median then clipped
  mean, one correction step), blotted onto each exposure and compared by
  jwst's two-pass rule with `scale` 0.5 / 0.3. Where fewer than three
  exposures overlap, the reference is their weighted mean and only pixels
  above it are flagged. The grid is processed in tiles, so no mosaic-sized
  array is allocated. `snowblind.outlier_reference` holds the method.
- `snowblind_image3` (`snowblind.Image3Pipeline`): `calwebb_image3` with
  jwst's `outlier_detection` slot replaced by `tiled_outlier`
  (`TiledOutlierStep`) and `sparse_outlier` (`SparseOutlierDetectionStep`),
  `outlier_step` naming the one that runs (`tiled_outlier` by default),
  and `SparseResampleStep` in its `resample` slot.
- The optional extra `snowblind[spizzle]` installs spizzle and numba, which
  these steps need; snowblind imports without them.
- `scripts/compare_sparse_resample.py` and
  `scripts/compare_sparse_outlier_detection.py` compare the steps with the
  jwst steps on a set of `_cal` files; `scripts/exp_injected_outliers.py`
  injects cosmic rays into `_cal` files and compares jwst's outlier
  detection with `TiledOutlierStep`'s method.

### Changed

- `StreamCleanStep`: `column_offsets` and `even_odd_columns` now default to
  `None`, which decides from the grism geometry: on for imaging and for
  grisms that disperse along the readout stream (NIRISS GR150R, NIRCam
  GRISMR), off for WFSS exposures whose traces run along the columns
  (GR150C, GRISMC). `True` or `False` overrides, so the offsets can now be
  kept on GR150C. `even_odd_columns` is therefore off by default on those
  exposures, where it used to be on. The step no longer rewrites its own
  parameters (issue #37).
- `StreamCleanStep` no longer skips a grism that disperses along the readout
  stream (NIRISS GR150R, NIRCam GRISMR). `fit_method='auto'` runs it with
  `'none'`, which removes the even-odd and column offsets and no stream
  noise. Naming `'wiener'` or `'row_median'` runs that filter there, with a
  warning, as `allow_dispersion_along_stream=True` did.
- `snowblind_detector1` no longer runs `JumpPlusStep` after its jump step;
  `SnowblindStep` is the only snowball hook. With the reset fit, its
  extra flags took one more good difference out of the ramp of every
  jump-flagged pixel, the grown snowball halos included, which made the
  halos noisier in the rate than with the jump step's `expand_large_events`
  (issue #25).
- `snowblind_detector1` no longer sets the jump step's
  `after_jump_flag_time1` to 0: the flagging of the groups after large jumps
  is left to the jwst defaults and the CRDS parameters, which set it for
  NIRISS.
- `SnowblindStep` grows only snowballs: jump regions that hold a saturated
  core, the pixels that saturate in the jump group or the next (in a rate
  image or rateints cube, its SATURATED pixels). Large jump regions without
  one, such as cosmic-ray tracks, are no longer grown, and the ring carried
  into the following groups is drawn around these cores only, not around
  every saturated pixel inside a grown region (issue #5). A hole of more
  than 200 pixels is filled too when it is made only of saturated pixels
  and the pixels next to them, a large core; one that also holds sky, such
  as the area around a saturated star in a rate image or one cut off by a
  jump region that crosses the array, is not. EXTNCRS
  counts the snowballs, and the step logs their number, summed over all
  groups and integrations ("Total snowballs = N", as the jump step logs).
- `SnowblindStep` finds and grows snowballs cut by the edge of the
  light-sensitive area (issue #16). Reference pixels (REFERENCE_PIXEL) and
  the area past the array border may hold part of an event: the disk of
  `min_radius` may extend over them as long as its center is on a
  light-sensitive pixel, and the event is completed past the edge before
  its equal-area radius is measured, so it grows by its whole size, in a
  corner too. An event centered on the outermost light-sensitive row or
  column needs no core in view if it holds a disk of `min_radius` that does
  not extend past the edge.
- `SnowblindStep` flags each snowball's core, and the pixels next to it, in
  the later integrations of a ramp, where they persist after the reset: a
  group is flagged on the core while the difference it ends starts less
  than `core_persistence_time` seconds after the end of the integration of
  the hit, and on the core grown by one pixel while it starts less than
  `neighbor_persistence_time` seconds after it, group 0 included. The
  group times come from `int_times`, or from the frame counts. Both default
  to times measured for each detector and packaged with snowblind
  (`snowblind/data/snowball_persistence_<detector>.asdf`, read by
  `snowblind.snowblind.packaged_snowball_times`): for the ten NIRCam
  detectors, NIRISS and the two NIRSpec detectors, the 99th percentile over
  the detector's snowballs of the time at which a pixel falls to
  0.005 DN/s, from about 1400 s to 21000 s for the saturated pixels and
  550 s to 3800 s for their neighbors. A detector without a file gets the
  longest packaged times, each the largest over the files, with a warning.
  `scripts/measure_snowball_persistence.py` measures the excess at the
  cores in dark exposures, `scripts/fit_snowball_persistence.py` fits the
  decay of each core, `scripts/package_snowball_persistence.py` writes the
  files, lengthening both times where darks of the detector taken later
  show more excess (`--later`), and
  `scripts/make_snowball_persistence_figure.py` draws them.
- `snowblind.saturation.flag_saturation_neighbors`, a post-hook for the
  saturation step run with `n_pix_grow_sat=0`: it flags the eight pixels
  around each saturated pixel, group by group, with CHARGELOSS and
  DO_NOT_USE, which keeps them out of the fits and the jump search as the
  jwst step's default growth does, while SATURATED marks only the pixels
  that reached the saturation threshold. As with that growth, the
  neighbors are flagged from the group during whose frames a pixel
  saturated, which the saturation step marks DO_NOT_USE when a group
  averages several frames (`snowblind.saturation.saturated_in_group`), and
  the zero frame of the pixels next to a pixel whose zero frame saturated
  is set to 0, which marks it unusable, as the saturation step does with
  the saturated one.
  NIRSpec IRS2 groups are grown in their normal layout, as the saturation
  step grows them.
- `SnowblindStep`'s `min_radius` defaults to 2 (was 4), close to the jump
  step's `min_jump_area` of 15 pixels; with a core required, small
  cosmic-ray hits no longer pass at that size. `snowblind_detector1` uses
  the default (was 3).
- `SnowblindStep` opens and grows each event in a window around it instead
  of over the whole frame, a few times faster on full-frame ramps.
  `dilate_jump_slice` and `dilate_large_area_jumps` take the saturated and
  reference pixels and return the cores with the grown regions;
  `dilate_saturated_cores` takes the cores.
- `SnowblindStep` flags the ring, each snowball's core grown by
  `ring_width` pixels, from the jump group to the end of the integration,
  where the charge leaking from the core keeps affecting its neighbors
  (`after_jumps` now defaults to None). The ring is drawn around the cores
  only, and `ring_width` is counted from the SATURATED flags, so its reach
  from the pixels that saturated depends on the saturation step's
  `n_pix_grow_sat`. When the core saturated during the frames of the group
  before the jump group, which holds part of the hit, the ring is flagged
  from that group (`dilate_saturated_cores` takes the group DQ).
- `snowblind_detector1` runs the saturation step with `n_pix_grow_sat=0`
  and `flag_saturation_neighbors` as its post-hook, so SATURATED marks only
  the pixels that saturated, and `SnowblindStep` with `ring_width=3` (was
  `after_jumps=5`), which reaches as far from them as before. A caller's
  `n_pix_grow_sat` above 0 leaves the hook out and narrows `ring_width` by
  as much, so the ring still reaches 3 pixels from the pixels that
  saturated.
- `ResetRampFitStep` sets CHARGELOSS in the DQ of the rate and rateints
  products on each pixel that has it in a group of the integration, which
  stcal's ramp fit does not carry over. `PersistenceFlagStep` flags the
  pixels with CHARGELOSS in an earlier exposure as well as those with
  SATURATED: the pixels next to saturated ones persist too, and its
  packaged times were measured with the jwst saturation step's default,
  which marks them SATURATED.

### Deprecated

- `StreamCleanStep`'s `allow_dispersion_along_stream` warns that it is
  deprecated and will be removed. `True` still resolves `fit_method='auto'`
  to `'wiener'` on a grism that disperses along the readout stream; name
  the `fit_method` instead.
- `SnowblindStep`'s `after_jumps` warns that it is deprecated and will be
  removed. Set, it still flags the ring in that many groups after the jump
  only, instead of to the end of the integration.
- `JumpPlusStep` warns that it is deprecated and will be removed. The
  jwst saturation step (since jwst 1.16) flags the group before saturation
  when its frames may have saturated, the jump step flags a group that a
  cosmic ray hit partway through, and the ramp fits leave those groups out;
  the likelihood fits also leave out the difference after a flagged group,
  and in an OLS fit the jump left there is small. What the step adds is
  mostly good data.

### Fixed

- `SnowblindStep` no longer flags JUMP_DET at pixels (0, 0), (0, 1) and
  (1, 0) in every group of a ramp. In a full frame these are reference
  pixels; in a subarray where that corner is light-sensitive, they lost
  their whole ramp.

## [1.3.1] - 2026-11-09

### Changed

- `StreamCleanStep`'s default `background_method` is `'mean'`, the mean
  of each group difference's background pixels. The background only sets
  the level put back after the clean: the mean leaves the sky as it was,
  while `'median'`, still available, puts it at the median of the
  background pixels. On darks the default keeps the dark current, which
  `None` removes, and the docs no longer recommend `None` for darks. WFSS
  exposures still switch to `crds_background`.
- `prepare_difference`, `clean_difference` and `estimate_channel_gains` no
  longer take `background_box_size`, which only jwst's per-difference
  `model` background used, and their parameters after `slowaxis` (after
  `mask` for `prepare_difference`) are keyword-only, so a call written for
  the old order fails instead of passing values to the wrong parameters.
  `clean_ramp` and `prepare_difference` raise a `ValueError` for a
  `background_method` other than `mean`, `median`, None or a model method
  given its background.

### Fixed

- `StreamCleanStep` with `background_method='median'` (the default until
  now) no longer removes the pixels the median clips from the pixels the
  noise is fitted on. The clip is redone on every group difference, so
  each difference was cleaned over a slightly different set of pixels, and
  what one difference took out of a group's read noise was not given back
  by the next. The leftovers added up from group to group as row and
  column banding in the rate, growing with the number of groups and most
  visible where the sky is faint. The median is now computed by
  snowblind's own `clipped_median_background`, which gives the same value
  as jwst's `background_level` without changing the mask.

## [1.3.0] - 2026-11-03

### Added

- `StreamCleanStep` records what it did to the sky level of each exposure
  in `meta.snowblind.stream_clean`, which the reset fit carries to the
  rate file and calwebb_image2 keeps in the `_cal` files:
  `sky_mean_minus_median`, the masked mean minus the masked median of the
  background pixels (the shift a scalar median background imposes on the
  sky), and `sky_change_by_cleaning`, the change of the sky made by the
  cleaning (negative when lowered), both in DN/s of the flat-fielded sky,
  averaged over the group differences and integrations, also in the
  header cards `SKYMNMED` and `SKYCHG`, and with a value per integration.
  The same numbers are logged. `clean_ramp` takes a `stats` dict that
  receives them per integration; `sky_shift_and_change` computes them for
  one difference. A level-matching step can compare its offsets with them.
- `snowblind.step.META_SCHEMA` defines, once for each step, the names
  under `meta.snowblind` and the header cards they are also written to;
  `Step.record` writes them.
- `PersistenceFlagStep` records in each flagged exposure the time it used
  (also in the header card `PERSTIME`), whether that time was given,
  packaged or the 2500 s default, and how many pixels it flagged from how
  many earlier exposures, in `meta.snowblind.persist`.
- `ResetRampFitStep` records in the rate and rateints products the scatter
  of the reset level, the offset of the reset level over its prior and the
  first-read timing in each output channel, and whether they were anchored
  to the exposure or taken from the calibration, in
  `meta.snowblind.reset_rampfit`.

### Changed

- `PersistenceFlagStep`'s `time` defaults to None: each detector then gets
  the persistence time measured for it and packaged with snowblind
  (`snowblind/data/persistence_<detector>.asdf`), and 2500 s, the former
  default, when it has none. The package holds times for the ten NIRCam
  imaging detectors, from 2200 s (NRCA1) to 12600 s (NRCB4): long on
  purpose, since persistence varies across a detector and with each
  pixel's history of illumination, each is the delay at which a pixel
  persisting twice as much as the detector's typical saturated star falls
  to 0.005 DN/s. Each file also holds `typical_time`, when the typical
  star itself does (1200 s to 5000 s), to pass as `time` to flag less. A
  `time` given still applies to every detector.
  `scripts/measure_persistence.py` measures the persistence of saturated
  stars against the delay in a series of `_rate` files, and
  `scripts/package_persistence_defaults.py` fits it and writes the files.
  Each detector is also measured with time running backwards, where
  there can be no persistence, and the forward minus the backward
  measurement is the persistence: that takes out an excess that does not
  decay, from the pixels of higher response that the selection favors
  and a bias that grows with the target's noise.
- `PersistenceFlagStep` measures `time` from the end of the saturated
  exposure to the middle of the later one, instead of from start to start,
  so an exposure longer than `time` still flags the next one.
- `ResetRampFitStep` copies into the rate and rateints products all the
  history entries and the extra primary header cards of the ramp (all but
  NEXTEND, which counts the ramp file's extensions) and its
  `meta.snowblind`, not only the snowblind history entries and `S_`
  cards, as the jwst helpers that build the products leave them out.
- `snowblind_detector1` runs the reference-pixel step with SIRS
  (`refpix_algorithm='sirs'`), whose output the stream clean is matched to;
  the jwst step defaults to a running median. Like the pipeline's jump
  settings, it takes precedence over the CRDS parameter references and
  gives way to the caller's (`--steps.refpix.refpix_algorithm=median`).
- The README is a short overview: what each step does and how it differs
  from the jwst step it replaces or adds to, installation, and examples.
  How to run each step and how it works are in the documentation, which
  has new pages on persistence flagging, with how the packaged times are
  measured, and on the snowball, jump and open-pixel steps.

### Removed

- `snowblind.step.carry_cal_step`: `ResetRampFitStep` is the only step
  that builds new models, and copies what it takes itself.

## [1.2.0] - 2026-11-01

### Added

- `ResetRampFitStep` logs the pixels in which its own likelihood jump
  search found jumps: those the jump step had not flagged, and those it had
  flagged where the search found more groups to exclude (the count of
  flagged differences is at DEBUG). The fitter returns the per-pixel count
  as a `new_jumps` plane of its per-integration results.

### Changed

- `snowblind_detector1` sets up the jump step for the stream clean and the
  reset fit: 7 sigma (three- and four-group thresholds too), neighbours
  flagged above 20 sigma, no large-event expansion or after-jump flagging,
  and `SnowblindStep(min_radius=3, after_jumps=5)` and `JumpPlusStep` as
  its first post-hooks. These take precedence over the CRDS parameter
  references; the caller's jump parameters still win, and
  `expand_large_events=True` gives back the jwst snowball flagging instead
  of the two hooks. With `--reset_fit=False` it warns that the jump step is
  still at 7 sigma.
- `PersistenceFlagStep` takes the saturated pixels from the DQ of the
  `_rate` or `_cal` files it flags (ramp fitting sets SATURATED where any
  group of any integration saturated), so the `_jump` files are no longer
  needed.
- Given file names or an association file with `save_results`,
  `PersistenceFlagStep` reads, flags, writes and closes one exposure at a
  time, holding only the saturation masks of the last `time` seconds, and
  returns the paths written. A container of models is flagged in memory as
  before. Either way it reads every member of an association's first
  product, as `ModelContainer` does, so background and PSF exposures count
  too, and takes bare file names from `input_dir`.
- `PersistenceFlagStep(save_mask=True)` writes the mask of flagged pixels
  (`_persistmask`) instead of the saturation mask of the `_jump` file.
- The documentation of `StreamCleanStep` and `ResetRampFitStep` now
  describes what data they take, how they work and how to choose their
  settings for a given dataset. The measurements on individual fields
  behind the defaults are kept in the plan file of the repository.

### Fixed

- `PersistenceFlagStep` places each exposure on the detector by its
  subarray start, so a subarray is flagged where an earlier full-frame
  exposure saturated and the other way round; exposures of one detector
  with different shapes made it fail. Two inputs of one detector that start
  at the same time, such as the `_rate` and `_cal` files of one exposure,
  are no longer flagged from each other.
- `JumpPlusStep` no longer sets its own `skip` on a ramp without frame
  averaging. As a hook, which it is in `snowblind_detector1`, it kept the
  setting, so every later run of the same pipeline skipped it.
- `PersistenceFlagStep` records its status in the FITS header as
  `S_PERSFL`. `S_PERSIS` is the jwst keyword for its own persistence step,
  and the jwst value replaced the snowblind one.
- `snowblind_detector1` subtracts the superbias with a subclass of the jwst
  step that keeps zeroframe pixels equal to the superbias from coming out
  at exactly 0, the pipeline's marker for an unusable zeroframe pixel. The
  raw data are whole DN, so with superbias references that hold only whole
  and half-integer values, as recent NIRCam ones do, this hit up to about
  1% of the pixels of an exposure, which the later steps left uncorrected
  and the reset fit could not anchor. Their rate uncertainty, about 7%
  above that of the other pixels, is now the same.
- `ResetRampFitStep` no longer marks pixels SATURATED where the zeroframe is
  unusable (zero), which put up to 1% of the pixels, at random on the sky,
  in the SATURATED mask of each rate product. In each integration it fits
  those pixels again without the zeroframe, instead of taking a zero first
  frame out of group 1, which biased their rates by about 0.3 of the rate
  uncertainty. A zeroframe that is 0 everywhere is ignored.

## [1.1.0] - 2026-10-27

### Added

- `StreamCleanStep` now cleans grism WFSS exposures:
  `background_method='crds_background'` (whic is the default for WFSS)
  subtracts the scaled CRDS background from every difference, grisms that
  disperse along the readout stream (NIRISS GR150R, NIRCam GRISMR) are
  skipped unless `allow_dispersion_along_stream` is set, and the per-column
  offsets, the scene mask's destriping and the Wiener gain are adjusted so
  that traces along the columns are not read as noise. A flat-fielded
  background reference (the NIRCam CRDS reffiles) is multiplied by the flat
  before scaling; `reference_units` overrides the automatic choice.

### Changed

- The INFO log of `StreamCleanStep` and `ResetRampFitStep` now says in plain
  words what each step did (the background mask, the rms of the 1/f noise
  removed, how the reset level was anchored); the spectral fits and
  per-channel numbers moved to DEBUG.

### Fixed

- `snowblind_detector1 --reset_fit=False` failed with "Extra value
  'reset_calib'": the ramp_fit parameters built from the snowblind spec were
  handed to the jwst ramp fit step. The reset-fit parameters are now dropped
  from that section when the jwst step takes the slot.
- The "Mean of empty slice" warning the stream clean raised on dispersed
  exposures, from pixels with no usable group in any integration, is gone.

## [1.0.1] - 2026-10-26

### Added

- `SnowblindStep` on a ramp now fills `meta.exposure.extended_emission_events`
  (FITS keyword `EXTNCRS`) with the number of large events it found per
  million pixels per second, replacing the value written by the jump step.

### Fixed

- The FutureWarnings from scikit-image about `binary_opening` and
  `remove_small_holes(area_threshold=...)` are gone; the masks are unchanged.
  The minimum scikit-image version is now 0.26.

- Each step's COMPLETE or SKIPPED status is now also written to the FITS
  primary header (`S_SNOWBL`, `S_JUMPPL`, `S_STRCLN`, `S_RSTRMP`, `S_PERSIS`,
  `S_OPNPIX`); before, it was kept only in the ASDF extension because the
  jwst schema has no keywords for these steps. `PersistenceFlagStep` and
  `OpenPixelStep` now record COMPLETE at all, and `ResetRampFitStep` carries
  the header cards of earlier steps into its rate products.
- Each step also adds a HISTORY entry such as `snowblind 1.0.0 SnowblindStep:
  COMPLETE`, which records the snowblind version in both the FITS header and
  the ASDF extension.

## [1.0.0] - 2026-10-26

### Added

- `StreamCleanStep`: removes 1/f (flicker) noise from NIRCam and NIRISS
  group data by filtering each output channel's readout stream with a
  Wiener gain derived from the exposure's own noise spectrum. The same
  linear filter is applied to every group difference, so nothing
  random-walks along the ramp. A two-tier NoiseChisel-style scene mask
  that also follows the diffraction spikes of saturated stars and leaves
  out the pixels the flat reference marks unusable, a scalar median
  background measured on the flat-fielded difference (default) or a
  box-spline background per integration for extended emission, per-read
  even-odd and per-column offset corrections, and the first read cleaned
  against the reset-level prior.
- `ResetRampFitStep`: likelihood ramp fitting with the superbias-subtracted
  reset level as an extra resultant, so that pixels with a single
  unsaturated group still get a rate. The reset offset is anchored to each
  exposure's own sky per output channel. It fits the `ramp_fit` slot of
  `calwebb_detector1` and has an `as_hook` form for use as a post-hook of
  the stock pipeline's skipped `ramp_fit`.
- `snowblind_detector1`: `calwebb_detector1` with `StreamCleanStep` hooked
  after the jump step and `ResetRampFitStep` in the `ramp_fit` slot,
  keeping the jwst pipeline's steps, parameters and CRDS parameter
  references; either substitution can be switched off.
- `snowblind-reset-calib`: command line tool that measures the reset-level
  scatter and mean pedestal map of a detector from dark ramps and writes
  the calibration file, with ASDF history entries recording the software
  and inputs.
- A packaged scalar reset calibration, one ASDF file per detector for the
  ten NIRCam detectors and NIRISS, which `ResetRampFitStep` uses when no
  calibration file is given.
- `OpenPixelStep` option `flag_low_signal_pix` to also flag pixels below
  the local median.
- Documentation of the 1/f cleaning and reset-anchored fitting, with the
  measurements behind the defaults, and a citation section in the README.

### Changed

- `RcSelfCalStep` is renamed `OpenPixelStep`, its stpipe alias `rc_selfcal`
  becomes `open_pixel`, and its module `snowblind.rc_selfcal` becomes
  `snowblind.selfcal`. Imports, `strun rc_selfcal` invocations and
  parameter files using the old names must be updated.
- Python 3.11 or later and jwst 2.0.0 or later are required; tested on
  Python 3.11 to 3.14.
- All steps log through module loggers under the `snowblind` namespace,
  which a shared base class names to stpipe so the messages reach the
  step log, also when a step runs as a hook inside a jwst pipeline;
  `Step.log`, deprecated by stpipe 0.9, is no longer used.

### Fixed

- Tests build their own `dq` arrays, as current stdatamodels no longer
  creates them.

## [0.2.1] - 2023-12-13

### Fixed

- Require scikit-image 0.20 or later, which provides the isotropic dilation
  that `SnowblindStep` uses.

## [0.2.0] - 2023-12-13

### Added

- `SnowblindStep` runs on single images (`_rate`) and cubes (`_rateints`),
  not only on ramps, and the flag it sets on dilated jumps can be chosen
  with `new_jump_flag`.  This allows its use in [**Grizli**](https://github.com/gbrammer/grizli).
- A negative `growth_factor` erodes the mask instead of growing it.

### Fixed

- Dilation logic and a region-labelling bug in `SnowblindStep`.

## [0.1.2] - 2023-10-27

### Fixed

- `PersistenceFlagStep` reads the `_jump` files from the step's input
  directory.

## [0.1.1] - 2023-10-18

### Fixed

- Opening `_jump.fits` inputs is more robust.

## [0.1.0] - 2023-09-25

First release.

### Added

- `SnowblindStep`: masks cosmic ray showers (snowballs) in JWST ramps by
  growing the jump-detected regions and flagging the groups around
  saturated cores, with `min_radius`, `growth_factor`, `after_jumps` and
  `ring_width` parameters.
- `JumpPlusStep`: propagates JUMP_DET and SATURATED flags properly in
  GROUPDQ for frame-averaged groups.
- `RcSelfCalStep` (now `OpenPixelStep`): flags new open pixels, hot pixels
  and open adjacent pixels by self-calibration against the other exposures
  of a detector, with an optional saved mask.
- `PersistenceFlagStep`: flags pixels affected by persistence from the
  preceding exposures, with an optional saved mask.
- Steps register as stpipe plugins and run with `strun` or as pipeline
  hooks.

[Unreleased]: https://github.com/mpi-astronomy/snowblind/compare/1.3.1...HEAD
[1.3.1]: https://github.com/mpi-astronomy/snowblind/compare/1.3.0...1.3.1
[1.3.0]: https://github.com/mpi-astronomy/snowblind/compare/1.2.0...1.3.0
[1.2.0]: https://github.com/mpi-astronomy/snowblind/compare/1.1.0...1.2.0
[1.1.0]: https://github.com/mpi-astronomy/snowblind/compare/1.0.1...1.1.0
[1.0.1]: https://github.com/mpi-astronomy/snowblind/compare/1.0.0...1.0.1
[1.0.0]: https://github.com/mpi-astronomy/snowblind/compare/0.2.0...1.0.0
[0.2.1]: https://github.com/mpi-astronomy/snowblind/compare/0.2.0...0.2.1
[0.2.0]: https://github.com/mpi-astronomy/snowblind/compare/0.1.2...0.2.0
[0.1.2]: https://github.com/mpi-astronomy/snowblind/compare/0.1.1...0.1.2
[0.1.1]: https://github.com/mpi-astronomy/snowblind/compare/0.1.0...0.1.1
[0.1.0]: https://github.com/mpi-astronomy/snowblind/releases/tag/0.1.0

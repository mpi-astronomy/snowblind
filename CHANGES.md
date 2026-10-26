# Changelog

## [Unreleased]

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

### Added

- `SnowblindStep` on a ramp now fills `meta.exposure.extended_emission_events`
  (FITS keyword `EXTNCRS`) with the number of large events it found per
  million pixels per second, replacing the value written by the jump step.

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

[Unreleased]: https://github.com/mpi-astronomy/snowblind/compare/1.0.0...HEAD
[1.0.0]: https://github.com/mpi-astronomy/snowblind/compare/0.2.0...1.0.0
[0.2.1]: https://github.com/mpi-astronomy/snowblind/compare/0.2.0...0.2.1
[0.2.0]: https://github.com/mpi-astronomy/snowblind/compare/0.1.2...0.2.0
[0.1.2]: https://github.com/mpi-astronomy/snowblind/compare/0.1.1...0.1.2
[0.1.1]: https://github.com/mpi-astronomy/snowblind/compare/0.1.0...0.1.1
[0.1.0]: https://github.com/mpi-astronomy/snowblind/releases/tag/0.1.0

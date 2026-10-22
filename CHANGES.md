# Changelog

All notable changes to snowblind are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses the tag as its version. Each release's section becomes the notes of
its GitHub release.

## [Unreleased]

### Added

- `StreamCleanStep`: removes 1/f (flicker) noise from NIR group data by
  filtering each output channel's readout stream with a Wiener gain
  derived from the exposure's own noise spectrum, with a NoiseChisel-style
  scene mask, a choice of background model, per-read even-odd and
  per-column offset corrections, and the first read cleaned against the
  reset-level prior.
- `ResetRampFitStep`: likelihood ramp fitting with the superbias-subtracted
  reset level as an extra resultant, so that pixels with a single
  unsaturated group still get a rate. Uses a per-detector calibration file
  and corrects its prior when the data had a different superbias
  subtracted.
- `snowblind-reset-calib`: command line tool that measures the reset-level
  scatter and mean pedestal map of a detector from dark ramps and writes
  the calibration file, with ASDF history entries recording the software
  and inputs.
- `OpenPixelStep` option `flag_low_signal_pix` to also flag pixels below
  the local median.
- Documentation of the 1/f cleaning and reset-anchored fitting, with the
  measurements behind the defaults.

### Changed

- `RcSelfCalStep` is renamed `OpenPixelStep`, its stpipe alias `rc_selfcal`
  becomes `open_pixel`, and its module `snowblind.rc_selfcal` becomes
  `snowblind.selfcal`. Imports, `strun rc_selfcal` invocations and
  parameter files using the old names must be updated.
- Python 3.11 or later is required; tested on 3.11 to 3.14.
- All steps log through module loggers under the `snowblind` namespace,
  which a shared base class names to stpipe so the messages reach the
  step log; `Step.log`, deprecated by stpipe 0.9, is no longer used.

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

[Unreleased]: https://github.com/mpi-astronomy/snowblind/compare/0.2.1...HEAD
[0.2.1]: https://github.com/mpi-astronomy/snowblind/compare/0.2.0...0.2.1
[0.2.0]: https://github.com/mpi-astronomy/snowblind/compare/0.1.2...0.2.0
[0.1.2]: https://github.com/mpi-astronomy/snowblind/compare/0.1.1...0.1.2
[0.1.1]: https://github.com/mpi-astronomy/snowblind/compare/0.1.0...0.1.1
[0.1.0]: https://github.com/mpi-astronomy/snowblind/releases/tag/0.1.0

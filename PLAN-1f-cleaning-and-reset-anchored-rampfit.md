# Plan: 1/f noise cleaning and reset-anchored likelihood ramp fitting

Status: design handed off for implementation in `snowblind`. Nothing below is
implemented yet. The plan is written for an implementing agent working in
`~/dev/snowblind` with `~/dev/jwst` and `~/dev/stcal` checked out beside it.

Two deliverables, in order:

- Part A: a new step, `StreamCleanStep`, that removes 1/f (flicker) noise from
  NIR group data by filtering each output channel's readout-ordered sample
  stream after masking the astronomical scene, with the filter gain set by the
  measured noise spectrum rather than by hand.
- Part B: a new step, `ResetRampFitStep`, that runs the stcal likelihood ramp
  fitter with the superbias-subtracted reset level (value zero, time zero)
  prepended as an extra resultant, with a calibrated uncertainty for that
  resultant, so that ramps with only one unsaturated group still get a rate.

Part C is the validation harness both parts use. Build it alongside Part A.

All three live in `snowblind` for now. The goal is working code and measured
results on real data, not a pipeline PR. Forking small pieces of `jwst` and
`stcal` into `snowblind` is acceptable and expected; note each fork with a
comment giving the source file and the commit hash it was copied from.


## 0. Decisions already made (do not re-open)

- Reference pixels are handled upstream by `RefPixStep` with
  `refpix_algorithm = "sirs"`. The SIRS kernels in CRDS exist for every NIRCam
  detector, NIRISS and NIRSpec. `StreamCleanStep` cleans what is left after
  SIRS; it does not touch reference pixels.
- `StreamCleanStep` is a new `snowblind` step, not a modification of
  `jwst.clean_flicker_noise`. It reuses that module's mask and background
  machinery by import, and copies its group-difference loop.
- The cleaner works on group differences (group j+1 minus group j), as the
  jwst step does. It also has an option to clean group 1 against zero, which
  Part B needs.
- The noise filter is a Wiener gain derived from the exposure's own masked
  background stream per channel. Darks are used for validation and as an
  optional prior, not as a required reference file.
- The reset resultant in Part B is implemented as a prepended plane of zeros at
  time zero with its own read-noise term. The algebra of the likelihood fit is
  otherwise unchanged. See section B.2 for exactly which covariance entry
  changes.
- Prototype `sigma_b` (reset-level uncertainty) is a per-detector scalar via the
  effective-read-count trick; the second iteration makes it a per-pixel array
  by forking one function. Do the scalar version first and measure before
  doing the per-pixel version.

Background on why these choices were made is in the conversation that produced
this plan; the short version of each is given inline below.


## 1. Environment and inputs

- Python env: make a fresh env with editable installs of `~/dev/jwst` (commit
  `e15d3e779` or later; needs `clean_flicker_noise` and the SIRS option in
  refpix) and `~/dev/stcal` (1.19.0 or later). The existing `jwst-dev` env
  has jwst 1.13 and is too old. `snowblind` installs editable on top.
- CRDS: `~/crds_cache/jwst_ops` exists. Set `CRDS_PATH` and `CRDS_SERVER_URL`
  as usual. The SIRS kernel files and the current NIRCam darks are already
  cached or will be fetched on first use.
- Test data: see section C.4. Get the file list before running on
  real data; do not go looking for proprietary data.
- Repo conventions (from James's global instructions): no `git commit` or
  `git add` unless told; do not create branches; tests are small, one behavior
  each, no branching inside tests, helpers in the test module; plain ASCII in
  commit messages and docs; avoid violent metaphors in prose and comments.
- Step conventions in this repo: subclass `jwst.stpipe.Step`, define `spec`
  and `class_alias`, set `meta.cal_step.<alias>` to COMPLETE or SKIPPED,
  register in `src/snowblind/__init__.py` (`__all__` and `_get_steps`), test
  in `tests/test_<alias>.py`. See `src/snowblind/jump_plus.py` for the
  smallest example.


## Part A: StreamCleanStep

### A.1 Interface

File: `src/snowblind/stream_clean.py`. Class `StreamCleanStep`, alias
`stream_clean`. Input: a `RampModel` after `JumpStep` (the `_jump` product),
or any group-level product with `groupdq` populated. Output: same model type,
data cleaned, `meta.cal_step.stream_clean = "COMPLETE"`.

```
spec = """
    fit_method = option('wiener', 'boxcar', 'row_median', default='wiener')
        # wiener: data-driven gain per channel (the new thing)
        # boxcar: normalized running mean of window_length clocks (sanity check)
        # row_median: per-row masked median, equivalent to jwst median_clean
    window_length = integer(default=4096)   # clocks; boxcar only
    psd_prior = string(default=None)        # optional ASDF with dark-derived PSD shape per channel
    background_method = option('median', 'model', None, default='model')
    background_box_size = int_list(min=2, max=2, default=None)
    n_sigma = float(default=2.0)
    mask_science_regions = boolean(default=False)
    apply_flat_field = boolean(default=False)
    user_mask = string(default=None)
    clean_first_group = boolean(default=False)  # clean group 1 against zero (Part B needs True)
    iterations = integer(default=2)             # mask, clean, re-mask, clean
    min_density = float(default=0.2)            # below this local background fraction, gain tapers to 0
    save_mask = boolean(default=False)
    save_noise = boolean(default=False)
    save_spectra = boolean(default=False)       # per-channel PSD, white level, gain, as ASDF
    skip = boolean(default=False)
"""
```

Usage target (both must work):

```
strun calwebb_detector1 jw..._uncal.fits \
    --steps.refpix.refpix_algorithm=sirs \
    --steps.jump.post_hooks="snowblind.SnowblindStep","snowblind.JumpPlusStep","snowblind.StreamCleanStep"
strun stream_clean jw..._jump.fits --fit_method=wiener --save_spectra=True
```

### A.2 Algorithm

Per integration, per group difference (and per group 1 against zero when
`clean_first_group` is set), per output channel:

1. Scene mask. Reuse the jwst code path exactly: draft rate image via
   `jwst.clean_flicker_noise.clean_flicker_noise._make_processed_rate_image`
   (it runs `RampFitStep` internally), then
   `_make_scene_mask` to get the background-only boolean mask. Pass through
   `mask_science_regions`, `n_sigma`, `user_mask`. Mask DO_NOT_USE and
   JUMP_DET of the later group with `_mask_unusable`. Mask NaN and saturated.
2. Background level. Reuse `background_level.background_level` with
   `background_method` (default `model`, which uses photutils
   `Background2D`); subtract it from the difference image before filtering and
   add it back after, exactly as `_clean_one_image` does. This is what stops
   the low-frequency end of the filter from eating real sky gradients.
3. Orientation. Transform the difference image, the mask, and the DQ to the
   detector readout frame with
   `jwst.lib.reffile_utils.science_detector_frame_transform(data, fastaxis,
   slowaxis)` (this is what `RefPixStep` uses). In that frame rows are the slow
   axis and the four outputs are 512-column blocks.
4. Stream index. For each output channel build the clock index of every
   pixel: `t = row * (512 + 12) + k`, where `k` is the column index within the
   channel in the direction that channel is actually read (adjacent outputs
   read in opposite directions) and 12 is the line overhead in clocks. The
   grid length is `2048 * 524`. The reference-pixel border rows and columns
   are part of the grid but are masked out of the fit (SIRS already used them).
   Verify the ordering empirically on a dark before trusting it (task A.4.3).
5. Noise spectrum. Compute the autocorrelation of the masked, background-
   subtracted stream for lags 0 to L (start with L = 2**16 clocks, about 125
   rows) as `acf_data(lag) / acf_mask(lag)`, where `acf_data` is the
   autocorrelation of `mask * data` and `acf_mask` that of `mask` (both by
   FFT). This is unbiased for gapped data. Fourier transform the windowed
   autocorrelation to get `S(f)`. Median-combine `S(f)` over all group
   differences in the integration (the noise is stationary; the scene residual
   is not, and the median suppresses it). Fit `S(f) = A / f**alpha + W` with
   `alpha` in [0.5, 1.5] and `W` free; keep the nonparametric smoothed `S(f)`
   too and use the fit only where the nonparametric estimate is noisy
   (highest and lowest frequencies). If `psd_prior` is given, use its shape
   and fit only `A` and `W`.
6. Gain. `G(f) = max(S(f) - W, 0) / S(f)`. Taper `G` to zero at the lowest
   frequencies whose period exceeds the background box size, since anything
   that slow is already in the background model and should not be removed
   twice.
7. Estimate. The noise estimate on the full grid is the normalized masked
   convolution: `n_hat = irfft(G * rfft(mask * d)) / irfft(G * rfft(mask))`.
   Where the denominator (local background density after filtering) is below
   `min_density`, taper `n_hat` toward the per-row masked median rather than
   trusting the ratio. The denominator is cheap to compute once per mask.
8. Subtract `n_hat` from every pixel of the channel, including masked ones (the
   estimate interpolates under sources), map back to the DMS frame, restore
   the background, and rebuild groups as the jwst step does: cleaned group
   j+1 = cleaned group j + cleaned difference. With `clean_first_group`,
   group 1 is treated as the difference against a plane of zeros and is
   rebuilt from the cleaned difference alone.
9. Iterate `iterations` times: re-derive the clip mask from the cleaned
   difference with the lower-half-only sigma estimate (`clip_to_background`
   with `lower_half_only=True`, as `_clean_one_image` does), re-estimate.

Common mode across the four outputs: version 1 estimates each channel
independently. Leave a hook (a function that takes the four `n_hat` streams)
for a later joint estimate; do not implement it until A.5 measurements show
the per-channel residual is dominated by something shared.

The `boxcar` method replaces steps 5 to 7 with a boxcar kernel of
`window_length` clocks in the same normalized convolution. The `row_median`
method replaces them with the masked per-row median, which must reproduce
`jwst.clean_flicker_noise.clean_flicker_noise.median_clean` to numerical
precision; that is the regression test that the orientation and plumbing are
right.

### A.3 Reuse map

| Need | Use |
|---|---|
| Draft rate image and scene mask | `_make_processed_rate_image`, `_make_scene_mask`, `create_mask` in `jwst/clean_flicker_noise/clean_flicker_noise.py` |
| Background level, outlier clipping | `background_level`, `clip_to_background` in `jwst/clean_flicker_noise/background_level.py` |
| DQ masking of later group | `_mask_unusable` |
| Loop over integrations and differences, group rebuild, save products | copy the body of `do_correction` (lines around 1440 to 1545) into the new step and strip the fft and median branches |
| Per-row masked median (regression reference) | `median_clean` |
| DMS to detector frame and back | `jwst.lib.reffile_utils.science_detector_frame_transform` with `meta.subarray.fastaxis` and `slowaxis`, as in `RefPixStep` |
| Clock index with line overhead, density weighting idea | `NSCleanSubarray.fit` in `jwst/clean_flicker_noise/nsclean.py` (read it; do not import it, it is NIRSpec-only) |
| Reference-pixel outlier rejection pattern | `apply_conv_kernel` in `jwst/refpix/optimized_convolution.py` |

The underscore-prefixed jwst functions are private and may move. Pin the jwst
commit in a comment at the import site and wrap the imports so a rename gives
a clear error.

### A.4 Tasks

1. Step skeleton, registration, `row_median` method only, test that it
   reproduces `median_clean` on a synthetic difference image with a mask.
2. Orientation and stream index for full-frame NIRCam (all ten detectors) and
   NIRISS. Unit test: a synthetic image with a known row-constant pattern plus
   a known within-row ramp maps to a stream whose values are monotone in `t`
   within each row, in both channel directions.
3. Empirical ordering check on a real dark (one NIRCam SW, one LW, one NIRISS):
   the autocorrelation of a group difference across a row boundary must be
   continuous with the chosen ordering and discontinuous with the channel
   direction flipped. Write this as a small script under
   `scripts/check_readout_order.py` and record the result in the plan's
   results section (section D).
4. `boxcar` method. Test on a synthetic stream: a known injected 1/f-like
   pattern plus white noise with a masked gap is recovered within the gap by
   interpolation and outside it to the expected precision.
5. Noise spectrum estimator (step 5). Test: for white noise in, `S(f)` is
   flat at the input variance within errors; for a synthetic `1/f` process
   generated by filtering white noise, the fitted `alpha` is within 0.1 of
   the truth and the mask does not bias `S(f)`.
6. `wiener` method end to end. Test on synthetic data: residual variance of
   the noise after cleaning is lower than after `row_median` and after
   `boxcar`, and the injected point sources are recovered with flux bias
   below 0.5 percent at 3 sigma above background.
7. `clean_first_group` and the group rebuild. Test: with `clean_first_group`
   True, group 1 of the output differs from input group 1 by exactly the
   estimated noise plane; with it False, group 1 is untouched.
8. `save_spectra` product (ASDF with `S(f)`, `W`, `G(f)` per channel per
   integration) and `save_noise`. Smoke tests only.
9. Run the harness (Part C) on the datasets in C.4. Fill in section D.

### A.5 What "better" means for Part A

On darks, after SIRS, compare `row_median`, `boxcar`, and `wiener`:

- residual PSD per channel of cleaned group differences (should sit at the
  white floor from the lowest fitted frequency up);
- row-banding power in the rate image (PSD of row means, per channel);
- aperture-mean scatter versus aperture size from 3x3 to 101x101 pixels (the
  SIRS paper's figure 10 is the template; the curve should approach the white
  noise prediction);
- injected-source flux bias as a function of distance from the source and of
  mask density.

Keep `row_median` as the baseline since it is what the jwst step does today.


## Part B: ResetRampFitStep

### B.0 Why

Both stcal fitters are insensitive to the absolute level of the ramp: OLS
fits a free intercept per segment and the likelihood fitter works on
differences. The only place a zero baseline is assumed is the OLS single-good-
group case (`data / one_group_time`), which the pipeline default
`suppress_one_group = True` disables. So a pixel that saturates after group 1
currently gets no rate from either algorithm.

A Gaussian prior on the reset level (mean zero after superbias subtraction,
standard deviation `sigma_b`) is exactly equivalent to prepending a resultant
of value zero at time zero whose read noise is `sigma_b` and which has no
Poisson term. The difference-space likelihood then gains one extra first
difference `d_0 = R_1 / t_1` and everything else (theta/phi recursions, one-
and two-omit jump tests, chi-square) is unchanged. The payoff is entirely in
the saturating regime: for a normal ramp `d_0` has several times the variance
of an ordinary difference and gets weighted down automatically.

Two quantities have to be calibrated, and both can be measured from data:

- `sigma_b`: the per-integration scatter of the true reset level about the
  superbias (kTC noise, bias drift, residual 1/f in the first read), not the
  formal error of the superbias mean. Measure it from the OLS optional-output
  pedestal on dark integrations.
- `t_1`: the mean time since reset of the reads in the first resultant. The
  convention in `stcal.ramp_fitting.likely_fit.get_readtimes` puts the first
  read at one frame time after reset, so `t_1 = frame_time * (nframes + 1) / 2`
  for the first group and `t_1 = frame_time` for a zeroframe. Calibrate the
  offset empirically from bright unsaturated pixels (B.1 task 3).

### B.1 Calibration tool

File: `src/snowblind/reset_calib.py`, exposed as a console script
`snowblind-reset-calib` and importable. Inputs: a list of `_jump` (or
`_uncal`) dark files for one detector, processed through Detector1 up to
`jump` with SIRS refpix. Outputs: an ASDF file per detector with:

- `sigma_b` (2048x2048 float32, DN, single-integration scatter of the
  pedestal), its median, and the number of integrations used;
- `t1_scale` (scalar), the multiplicative correction to the nominal `t_1`
  (default 1.0 until measured);
- provenance: input files, jwst and stcal versions, refpix algorithm.

Tasks:

1. Run `RampFitStep` with `algorithm = "OLS_C"`, `save_opt = True`,
   `suppress_one_group = False` on each dark; read the `pedestal` array from
   the `_fitopt` product (shape nints x nrows x ncols). Pedestal is the fitted
   line extrapolated to the reset; on a dark it is the reset residual.
2. Per pixel: sigma-clipped standard deviation of pedestal across all
   integrations and files, after removing a per-integration per-channel median
   (so that frame-level offsets refpix already handles are not counted twice).
   That is `sigma_b`. Also record the per-pixel mean (the systematic
   superbias error at this epoch) separately; it is diagnostic only and must
   not be folded into `sigma_b`.
3. `t_1` calibration, done on a flat or a bright sparse science field rather
   than a dark: for pixels with 50 to 2000 DN in the first difference and no
   flags, compute `r = d_0 / a_rest`, where `a_rest` is the likelihood rate
   from all other differences. The median of `r` over many pixels is
   `1 / t1_scale`. Report it with its scatter and the per-channel values. If
   it differs from 1 by more than the scatter, store it; the fitter multiplies
   the nominal `t_1` by it.
4. Chi-square check: with `sigma_b` and `t1_scale` applied, the normalized
   residual `(d_0 - a_rest)**2 / var(d_0)` should average to 1 over the same
   pixels. Report it. This is the pass/fail for the calibration.

### B.2 Fitter changes

Fork from `~/dev/stcal/src/stcal/ramp_fitting/likely_fit.py` into
`src/snowblind/likely_fit_reset.py`: `likely_ramp_fit`, `mask_jumps`,
`fit_ramps`, and `compute_alphas_betas`. Import everything else
(`Covar`, `IntegInfo`, `RampResult`, `determine_diffs2use`,
`compute_image_info`, the theta/phi/matrix functions, `get_readtimes`) from
stcal. Changes, all inside the forked functions:

1. Read times. After `get_readtimes`, and after the existing zeroframe
   handling, insert a first entry `np.array([0.0])` so the reset resultant has
   `mean_t = tau = 0` and `N = 1`. The photon-noise entries of `Covar` then
   come out right with no further change (`alpha_phnoise[0] = tau_1 * a / g /
   t_1**2`, which is the variance of the first difference from charge
   accumulated since reset). Apply `t1_scale` by scaling the first group's
   read times before building `Covar`.
2. Data. Prepend a plane of zeros to `data` for the integration. Prepend a DQ
   plane of zeros (always good) to `gdq`, except where the pixel DQ has
   DO_NOT_USE.
3. Zeroframe DQ. Where the existing code copies group 0's DQ onto the
   zeroframe plane, replace it with: zeroframe DQ is SATURATED where the
   zeroframe value is exactly zero (that is the pipeline's convention for an
   unusable zeroframe pixel) and good otherwise. Without this a pixel
   saturated in group 1 loses the only difference that could give it a rate.
4. Read-noise covariance. The only algebraic change. In `Covar`,
   `alpha_readnoise[0] = (1/N_0 + 1/N_1) / dt_0**2` and `beta_readnoise[0] =
   -1 / (N_1 * dt_0 * dt_1)`. We need `alpha_readnoise[0] = (sigma_b**2 +
   sigma_r**2 / N_1) / dt_0**2` and `beta_readnoise[0]` unchanged, where
   `sigma_r` is the single-read noise the code already uses (the input CDS
   read noise divided by sqrt 2 inside `likely_ramp_fit`).
   - Scalar prototype: after constructing `Covar`, set `covar.Nreads[0] =
     sigma_r_median**2 / sigma_b_median**2` (a float) and call
     `covar._compute_alphas_and_betas(covar.mean_t, covar.tau, covar.Nreads,
     covar.delta_t)` again. `Nreads` is used elsewhere only in the jump
     eligibility masks in `mask_jumps`, where index 0 is forced True and the
     two-omit mask excludes it, so a fractional value is harmless. Document
     the approximation (per-detector `sigma_b` and `sigma_r`).
   - Per-pixel version: in the forked `compute_alphas_betas`, accept
     `sigma_b` (ncols,) for the row and overwrite row 0 of `alpha_readnoise`
     with `(sigma_b**2 + rnoise**2 / N_1) / dt_0**2` before forming `alpha`.
     Thread `sigma_b[row]` through `fit_ramps` and `mask_jumps`. Everything
     downstream, including `var_rnoise` in `get_ramp_result`, only sees
     `alpha` and `beta`, so the reported variances automatically include the
     prior's contribution.
5. Outputs. Return the same `image_info`, `integ_info` as stcal, plus a
   per-integration boolean plane `used_reset` (True where `d_0` was in the
   final fit) and a per-integration float plane `d0_resid` (the normalized
   residual of `d_0`), for diagnostics and for the chi-square check in B.1.

Do not touch OLS, `ramp_fit.py`, or the C extension.

### B.3 Interface

File: `src/snowblind/reset_rampfit.py`. Class `ResetRampFitStep`, alias
`reset_rampfit`. Input: a `RampModel` after `JumpStep` (and after
`StreamCleanStep` with `clean_first_group = True` when both are used).
Output: `ImageModel` and `CubeModel` like `RampFitStep`, with the two extra
planes above attached to the cube model as extra arrays or saved as a
separate `_resetfit` product (either is fine; say which in the docstring).

```
spec = """
    reset_calib = string(default=None)   # ASDF from snowblind-reset-calib; required unless sigma_b is set
    sigma_b = float(default=None)        # DN; scalar override for prototyping
    t1_scale = float(default=1.0)
    per_pixel_sigma_b = boolean(default=False)
    use_zeroframe = boolean(default=True)
    rejection_threshold = float(default=None)  # passed through to the likelihood jump search
    save_opt = boolean(default=False)
    skip = boolean(default=False)
"""
```

The step builds `RampData` exactly as `jwst.ramp_fitting.ramp_fit_step`
does (reuse `RampFitStep` for the reference file lookups and `set_groupdq`,
then call the forked `likely_ramp_fit` instead of `stcal.ramp_fitting.ramp_fit`),
and reuses the jwst step's output-model assembly.

### B.4 Tasks

1. Calibration tool with tasks B.1.1 and B.1.2, run on the cycle-4 NIRCam
   darks for at least one SW and one LW detector. Record the median `sigma_b`
   in DN and in units of single-read noise in section D. This number decides
   how much the rest matters, so do it first.
2. Forked fitter, scalar `sigma_b`, synthetic tests:
   - a ramp with all groups good: rate and uncertainty agree with stcal's
     `likely_ramp_fit` to within the expected tiny shift (compute the expected
     shift analytically from the weights and assert it, do not just assert
     "close");
   - a ramp with only group 1 unsaturated: rate equals `R_1 / t_1` and the
     reported variance equals `(sigma_b**2 + sigma_r**2 / N_1 + a * t_1 / g)
     / t_1**2`;
   - a ramp with only the zeroframe unsaturated: same with `t_1 =
     frame_time`;
   - a ramp whose reset level is off by 10 `sigma_b`: the one-omit search
     drops `d_0` and the rate equals the stcal rate.
3. Step wrapper and registration; run on one dark and one science field;
   fill in B.1.3 and B.1.4 from the science field.
4. Per-pixel `sigma_b` version; repeat the tests with a non-uniform
   `sigma_b` image.
5. Harness comparison (C.3).

### B.5 Known caveats to carry into the write-up

- The superbias is the mean first read of dark integrations, so the prior is
  on "what the first resultant reads with no source," which already includes
  reset settling and dark to the first read. `d_0` assumes that settling is
  independent of signal level; that is untested for bright pixels and is one
  of the things C.3 measures.
- The zeroframe is not dark-subtracted in Detector1 (no step touches it), so
  the zeroframe-to-group-1 difference carries about one frame of dark. Small
  for NIR; note it, do not fix it here.
- Absolute timing matters for `d_0` and for nothing else in the fit. Any
  constant offset in read times cancels in every other difference.
- `average_dark_current` reaching ramp fitting is zero for NIRCam (the new
  darks carry no AVDRKCUR), and both stcal fitters add it to a DN/s rate
  before dividing by gain once, which looks like a units mismatch for the
  dark term. Irrelevant here; flag it to James, do not fix it in snowblind.


## Part C: validation harness

### C.1 Layout

`scripts/` at the repo top level (not installed): one script per experiment,
each writing a small CSV or JSON of metrics plus PNG figures into a
`results/<date>_<experiment>/` directory that is git-ignored. Keep the
metric functions in `src/snowblind/metrics.py` so the tests can cover them.

### C.2 Metrics (implement once, use everywhere)

- `channel_psd(diff_cube, mask, fastaxis, slowaxis)`: readout-ordered PSD per
  channel, median over groups.
- `row_banding_power(rate_image)`: PSD of per-channel row means.
- `aperture_scatter(rate_image, sizes)`: standard deviation of the mean in
  square apertures of each size, background pixels only, with the white-noise
  prediction from the readnoise and Poisson terms.
- `injected_flux_bias(rate_image, truth_catalog)`: aperture photometry of
  injected sources versus truth, binned by local mask density and by distance
  from the nearest other source.
- `d0_chisq(resetfit_product)`: mean normalized residual of `d_0` (Part B).
- `recovered_fraction(rate_image, truth)`: fraction of pixels with finite
  rate, and photometry of saturated-core stars against unsaturated references
  (Part B dynamic-range metric).

### C.3 Experiments

1. Darks, Part A: Detector1 to `jump` with SIRS, then `stream_clean` in each
   of its three methods, then `ramp_fit` (OLS_C default). Also the two
   reference pipelines: default refpix with no cleaning, and SIRS with jwst
   `clean_flicker_noise` median. Six variants, metrics 1 to 3.
2. Injected scenes, Part A: add synthetic point sources and a smooth
   background gradient to a dark's groups as `rate * t_group` (inject after
   `linearity` by running Detector1 in two halves, or inject into `_uncal`
   at fluxes well below saturation and accept the small nonlinearity error;
   say which was done). Metric 4 for the same six variants. This is the
   experiment that catches over-subtraction next to sources.
3. Ramp fitting, Part B: on a bright sparse field, compare OLS_C, LIKELY, and
   `reset_rampfit` for metrics 5 and 6, and on darks for rate bias. Report
   the number of additional pixels with finite rates and the photometry of
   the stars whose cores saturate after group 1.

### C.4 Data needed from James

- NIRCam full-frame darks, cycle 4, at least 3 integrations each: one SW
  detector (NRCA1 is already the one whose reference file was examined) and
  one LW detector. RAPID or BRIGHT1 preferred so groups are single frames.
- One NIRCam sparse field with bright stars that saturate early (for Part B
  and for `t_1` calibration), same detectors.
- One crowded or extended field (for Part A mask-density behavior).
- Optional: one NIRISS imaging dark and field, to exercise the second
  detector geometry.

Record file names and program IDs in section D, not in this section.


## D. Results log

Empty until measurements exist. Append dated entries; do not rewrite earlier
ones. Each entry: what was run (files, variants, commit hashes), the numbers
from the metrics above, and one or two sentences on what they mean.

Units used below: a clock is the time to read one pixel, 10 microseconds;
a full-frame row is 524 clocks or 5.24 ms (512 pixels plus 12 of overhead).


## E. Order of work and milestones

1. Part A tasks 1 to 3, Part B task 1 (the `sigma_b` measurement). Milestone:
   readout ordering verified on real darks; `sigma_b` known for one SW and one
   LW detector. If `sigma_b` is much larger than a few single reads, say so
   in D before going further on Part B; Part A continues regardless.
2. Part A tasks 4 to 8, metrics C.2 1 to 4, experiment C.3.1. Milestone: a
   results entry comparing the six variants on darks.
3. Experiment C.3.2. Milestone: flux-bias curves. Decide on the default
   `background_method`, `iterations`, and `min_density` from these.
4. Part B tasks 2 to 4, experiment C.3.3. Milestone: calibrated `t1_scale`
   and `sigma_b`, chi-square near 1, a count of recovered pixels and a
   photometry comparison for saturated stars.
5. Write-up draft in `docs/` summarizing D with the figures. This is the
   material for the conversation with the jwst developers.

Run `/code-review` on each part after its tests pass and before moving on.


## F. References

- Rauscher, Fixsen, Mosby, "Simple Improved Reference Subtraction for H4RG,
  H2RG and H1RG," JATIS 8(2) 028002 (2022); arXiv:2602.02253. Implemented in
  `jwst/refpix/optimized_convolution.py` and selected by
  `refpix_algorithm = "sirs"`.
- Brandt, "Optimal Fitting and Debiasing for Detectors Read Out Up-the-Ramp"
  and the companion jump-detection paper, arXiv:2404.01326 (paper II).
  Section III states the reset value is not fit and an informative prior is
  not treated; section VI.2 is the first-read reconstruction that
  `likely_fit.py` reproduces for the zeroframe.
- Rauscher, NSClean (NIRSpec 1/f cleaning); `jwst/clean_flicker_noise/nsclean.py`.
- `stcal/ramp_fitting/likely_fit.py` and `likely_algo_classes.py` (the
  `Covar` class holds the covariance entries that B.2 changes).
- `jwst/clean_flicker_noise/clean_flicker_noise.py` (mask, background,
  difference loop to reuse).
- NIRCam dark reference `jwst_nircam_dark_0617.fits`: shortwave full-frame
  darks have nominal pixels set to zero and keep hot pixels only (0.7 percent
  of science pixels on NRCA1); longwave darks are non-zero everywhere. No
  AVDRKCUR. This is why `average_dark_current` is zero at ramp fitting.

### 2026-10-20: readout ordering (A.4.3) and first sigma_b (B.4.1)

Environment: `jwst` conda env, jwst e15d3e779 (editable), stcal 1.19.0
(editable), snowblind working tree (uncommitted). Data in
`~/data/jwst_darks/`: program 8758 RAPID 2x150 darks (all ten detectors),
program 9022 BRIGHT1 10x10 and MEDIUM8 7x10 darks (all ten), program 12534
RAPID 1x200 (A side). Results under `results/2026-10-20_readout_order/` and
`results/2026-10-20_reset_calib/`.

Readout ordering, `scripts/check_readout_order.py`, 40 groups of the first
integration, superbias-subtracted only. A single edge-pixel pair per row
boundary has no power: in raw differences the within-row autocorrelation is
about 0.03 at every lag from 2 to 511 (a per-row common offset, the thing
refpix and the row median remove) and 0.16 to 0.32 at lag 1 (inter-pixel
capacitance, direction independent). With per-row means removed the
within-row autocorrelation is 0.011 to 0.037 at lag 2, 0.004 to 0.007 at
lags 3 to 8, about 0.002 at lag 13 and zero beyond lag 100. The test that
works compares both directions on the row-mean-removed stream over clock
lags 13 to 100 and over all 39 differences:

| Detector | fast/slow | ch 0 | ch 1 | ch 2 | ch 3 |
|---|---|---|---|---|---|
| NRCA1 (jw08758003001) | -1 / 2 | +1 (2.5 sigma) | -1 (4.4) | +1 (6.7) | -1 (5.5) |
| NRCB3 (jw08758002001) | 1 / -2 | +1 (6.7) | -1 (9.3) | +1 (9.5) | -1 (10.5) |

So in the detector frame channel 0 reads toward increasing column and the
directions alternate, for both NIRCam axis geometries. This is the default
`first_direction = 1` in `StreamCleanStep`. The hot-pixel trailing test in
the same script finds no asymmetry at the 0.5 percent level and is not
informative. NRCALONG is pending a truncated CRDS saturation file.

What this says about Part A: after the per-row offset is removed there is
very little correlated power within a row. The Wiener gain will be close to
one only at periods of a row or longer. The gain over the row median, if
any, will come from weighting adjacent rows together and from not paying
the median's own noise (1.25 sigma / sqrt(512) per row), not from in-row
filtering. Experiment C.3.1 measures this.

sigma_b, `snowblind-reset-calib`, NRCA1, program 9022 BRIGHT1 10x10
(jw09022004001_02101_00001), Detector1 to jump in memory with SIRS refpix,
OLS_C pedestal, per-integration per-channel median removed, 3-sigma
clipped scatter over 10 integrations:

| Quantity | Value |
|---|---|
| median pedestal scatter, science pixels | 17.6 DN |
| 16 / 84 percentile | 13.5 / 22.2 DN |
| per-channel medians | 17.60, 17.61, 17.61, 17.69 DN |
| median pedestal mean (superbias error) | -0.03 DN |
| readnoise reference (jwst_nircam_readnoise_0321) CDS median | 7.34 DN, single read 5.19 DN |
| gain reference median | 2.05 e/DN |
| scatter in single-read units | 3.4 (16/84: 2.6 / 4.3) |
| scatter in electrons | 36 e |

The pedestal scatter contains the first group's own read noise, which the
likelihood fitter counts separately, so the tool now stores
`sigma_b**2 = scatter**2 - readnoise_single**2 / nframes` (about 16.8 DN
here) as `sigma_b` and the raw value as `pedestal_scatter`. 36 e rms is the
size expected for kTC reset noise on an H2RG, which is reassuring: the
superbias removes the mean reset level but not its per-integration
scatter. In the fitter the reset difference `d_0` will carry about
`(sigma_b**2 + sigma_r**2) / (2 sigma_r**2)`, roughly 6 times, the variance
of an ordinary difference, so it is down-weighted for normal ramps and
decisive only for ramps with one good group, as intended. A few single
reads, not many: Part B proceeds. NRCALONG pending the same CRDS file.

Note for the write-up: stcal's OLS pedestal is
`group0 - slope * ((nframes + 1) / 2 + dropframes) / (nframes + groupgap)`
in group-time units and is written as exactly zero where the integration is
saturated; the tool treats zeros as missing.

### 2026-10-20: why difference cleaning can make the rate noisier (C.3.1, first pass)

Setup: NRCA1 dark jw08758003001_05201_00001, first integration, first 20
groups, Detector1 to jump in memory with SIRS, `scripts/exp_darks_part_a.py`
and two diagnostic scripts. Metric mask: 98.8 percent of pixels. Rate
images from OLS_C on the cleaned ramps. Numbers are robust sigmas in DN/s
over the mask; "row-mean sigma" is the scatter of per-row means in channel
0 of the rate image (white expectation about 0.0008).

First pass, with the plan's defaults (`background_method='model'`,
`n_sigma=2`, `iterations=2`):

| Variant | rate sigma | row-mean sigma | slow-band S | row-to-50-clock S | white S |
|---|---|---|---|---|---|
| median refpix, no cleaning | 0.0208 | 0.0062 | 1682 | 122 | 44.4 |
| SIRS, no cleaning | 0.0205 | 0.0051 | 1204 | 121 | 44.4 |
| SIRS + jwst clean_flicker_noise median | 0.0213 | 0.0078 | 403 | 122 | 44.4 |
| SIRS + stream_clean row_median | 0.0217 | 0.0089 | 397 | 123 | 44.4 |
| SIRS + stream_clean boxcar | 0.0205 | 0.0054 | 726 | 121 | 44.4 |
| SIRS + stream_clean wiener | 0.0227 | 0.0053 | 217 | 43 | 43.9 |

The group differences do get cleaner (the spectra columns; the Wiener
estimate takes the stream spectrum down to the white floor at every
frequency above a row period), yet every cleaned variant has a noisier
rate image than SIRS alone, including the jwst step. The cause is the group
rebuild. Cleaned group j is the input group j minus the sum of the noise
estimates of differences 1 to j. The true 1/f offset of group j enters
differences j and j+1 with opposite signs, so in the sum of differences
that rebuilds a group only the first and last offsets survive and the rest
cancel. Any part of an estimate that does not cancel this way, that is,
any part that is not a fixed linear function of the group data applied
identically to every difference, random-walks along the ramp. The slope
estimator then weights that walk with the cumulative OLS weights, whose sum
of squares is about 27 times that of the per-group weights for 20 groups,
while the true offsets mostly cancel. For the per-row median the part that
does not cancel is the difference between a median and a mean, 0.25 DN per row per difference here, which
predicts a rate noise of 0.0047 DN/s added in quadrature; the measured
change from 0.0203 to 0.0208 (row_median, no background model) is 0.0046.
The per-difference Background2D model is a second, larger nonlinear term
(0.0203 to 0.0216). The per-difference sigma clipping is a third: it
changes the mask, and so the filter, from one difference to the next.

Direct test of the explanation, same data, `StreamCleanStep` only:

| Variant | rate sigma | row-mean sigma |
|---|---|---|
| SIRS, no cleaning | 0.0203 | 0.0051 |
| wiener, no background, no clipping, 1 iteration | 0.0174 | 0.0004 |
| wiener, scalar median background, no clipping | 0.0180 | 0.0014 |
| wiener, no background, clipping at 2 sigma, 2 iterations | 0.0211 | 0.0036 |
| row_median, no background, no clipping | 0.0207 | 0.0068 |
| boxcar, no background, no clipping | 0.0198 | 0.0031 |

A strictly linear Wiener clean lowers the pixel rate scatter by 15 percent
and the row banding of the rate image to below the white level, because a
linear filter applied identically to every difference leaves the cleaned
rate equal to the high-passed uncleaned rate. The same filter with
per-difference clipping is worse than no cleaning. The median is worse
than SIRS alone in every configuration. The scalar median background keeps
the frame-level offsets that the no-background filter had removed as part
of the DC term, which is why it is slightly worse; it does preserve the
dark current level (median rate change 0.00003 versus 0.003 DN/s).

Consequences implemented today:

- New `background_method='integration_model'` (now the step default): one
  Background2D model per integration, fitted to the mean of the usable
  group differences and subtracted from every difference. The sky is still
  protected, the operation is the same on every difference, and the 1/f
  banding averages down in the mean difference before it can enter the
  model. For group 1 cleaned against zero the model is scaled by the ratio
  of the first group's time to the group time.
- The Wiener gain is forced to one at zero frequency and no low-frequency
  taper is applied inside the normalized convolution: a gain that is zero
  at DC cannot be normalized against the mask. The plan's taper exists as
  `protect_period`, implemented as the difference of two normalized
  convolutions, off by default.
- The spec defaults for `n_sigma` (2) and `iterations` (2) are kept until
  C.3.2 shows whether per-difference clipping is needed for photometry; on
  darks it is harmful and the experiment scripts run the "linear" variants
  with `n_sigma=0, iterations=1` alongside the defaults.

Noise spectra of this dark (Wiener fit, per channel): alpha 0.89 to 0.91,
white 32 to 37 DN^2 per difference per bin, corner period 1.0 to 1.3 ms;
the spectrum has a second rise toward the highest frequencies (period 0.1
ms: 114 to 184 versus 58 to 67 at 0.5 ms) that is not 1/f. The gain
is 0.8 at the row period, 0.93 at 20 ms and 0.4 at 0.5 ms.

### 2026-10-20: two stcal details that affect the calibration tool

- `stcal.ramp_fitting.ramp_fit.ramp_fit_data` scales the read-noise array
  it is given in place (`readnoise_2d *= gain_2d / sqrt(2 nframes)` for
  OLS). A caller that reuses the array afterwards gets electrons per single
  read, not the reference file's DN. `snowblind-reset-calib` now passes a
  copy; the first version of the NRCA1 calibration file subtracted a read
  noise of 7.5 DN instead of 5.2 DN and gave sigma_b 15.9 DN. The corrected
  file is `nrca1_bright1_reset_calib_v3.asdf`.
- The OLS_C pedestal is `group0 - slope * ((nframes + 1) / 2 + dropframes)
  / (nframes + groupgap)` with the slope in DN/s and no group time, so it
  is the first group minus about one second of signal rather than the
  extrapolation to the reset. On a synthetic ramp with reset 20 DN, 1 DN/s
  and 10 s groups the pedestal is 29, not 20. For darks the slope is of
  order 0.001 DN/s and the error is negligible, which is why the tool uses
  it as the plan prescribes. Flag for James, do not fix in snowblind.

### 2026-10-20: reset-anchored fit on the program 6154 field (B.4.3, B.1.3, B.1.4)

Data: jw06154001001_02101_00004_nrca1 (BRIGHT1, 1 x 5, t1 = 10.737 s,
group time 21.47 s), Detector1 to jump in memory with SIRS, fitted with
OLS_C, stcal LIKELY and `ResetRampFitStep` (scalar sigma_b from the NRCA1
BRIGHT1 dark). Script `scripts/exp_reset_field.py`, results under
`results/2026-10-20_reset_field*/NRCA1/`.

Recovered pixels. 3530 science pixels saturate from group 2 on; none gets
a rate from OLS_C or LIKELY, all 3530 get one from the reset fit. Over the
whole detector the reset fit has 3723 more finite pixels than OLS_C. The
first run also lost 24 pixels that OLS_C fits: pixels with exactly two good
groups, so one ordinary difference plus the reset difference. With two
differences the one-omit jump search is degenerate (dropping either leaves
a perfect fit) and it dropped the real difference, or both. stcal never
meets this case because such a ramp has a single difference there. The
fork now refuses to drop anything when only two differences are usable.

First-read timing (B.1.3), 242k clean pixels with a first difference of 50
to 2000 DN, ratio r = d_0 / a_rest with a_rest the LIKELY rate:

| first difference (DN) | n | median r | standard error | median R1 - a t1 (DN) |
|---|---|---|---|---|
| 50 to 100 | 117035 | 0.788 | 0.002 | -7.1 |
| 100 to 200 | 60631 | 0.861 | 0.0015 | -9.6 |
| 200 to 400 | 34201 | 0.886 | 0.0013 | -15.9 |
| 400 to 800 | 18751 | 0.893 | 0.0013 | -28.9 |
| 800 to 1200 | 6280 | 0.895 | 0.002 | -51.1 |
| 1200 to 2000 | 5023 | 0.897 | 0.002 | -77.0 |

All pixels: median r = 0.858 +- 0.001; per channel 0.867, 0.864, 0.875,
0.833. The ratio rises with flux and the offset grows with flux, so the
first read is neither a pure timing error (constant r) nor a pure additive
offset (r tending to 1). A two-parameter model fits the bin medians:
R1 = b + k a t1 with k about 0.91 and b about -4 DN. The first read holds
about 9 percent less signal than the slope predicts, as if integration
started one second after the nominal reset time, plus a few DN offset.
Whether k holds for pixels 10 to 50 times brighter than this sample is the
open question for the saturated stars; C.3.3 photometry decides it.

Two corrections to the plan's conventions came out of this. The fitter
multiplies the first group's read times by t1_scale, so the scale that makes
d_0 unbiased is the median ratio itself (0.858), not its inverse as B.1.3
says; the first rerun used 1.166 and made the bright-pixel chi-square worse
(3.24). And the model needs the offset b, so the step gained
`reset_offset` (DN added to the prior mean) and `use_pedestal_mean` (the
per-pixel superbias error map from the dark calibration as the prior mean).

Chi-square (B.1.4), nominal t1 and zero prior mean: 2.50 over the bright
sample and 1.85 over all 4.07 million clean pixels. The all-pixel value
implies a reset scatter of about 22 DN on the field against 16.8 DN in the
dark. The dark calibration's per-pixel pedestal mean (the systematic
superbias error) has a robust scatter of 13.9 DN, of which 5.6 DN is the
10-integration noise, leaving 12.7 DN rms of superbias error per pixel;
16.8 and 12.7 in quadrature give 21.1 DN, which accounts for the excess.
The dark (2026-03-27) and the field (2025-03-28) are a year apart, so the
test of whether that map transfers is the chi-square of the run with
`use_pedestal_mean`, below.

Also seen: with t1_scale 1 the reset fit lowers the median rate of all
pixels by 1 percent relative to LIKELY (0.3842 versus 0.3884 DN/s), because
the low-biased d_0 carries about one sixth of the weight in a 5-group
ramp. The timing calibration therefore matters for every pixel, not only
for the saturated ones.

### 2026-10-20: C.3.1 on the NRCA1 dark, final table of the day

Same data and metrics as above (`results/2026-10-20_darks_part_a_v3/NRCA1/`).
"clipped" is the jwst-like configuration (per-difference Background2D,
2-sigma clipping, 2 iterations); "boxN" is the StreamCleanStep default
(one background model per integration, no clipping, one iteration) with an
N-pixel background box; "median_bkg" uses a scalar median background.
Aperture scatter is the robust sigma of aperture means in DN/s; the white
prediction is 0.0205 / size.

| Variant | rate sigma | row-mean sigma (4 ch) | slow-band S | row-to-50 S | ap 9 | ap 33 | ap 101 |
|---|---|---|---|---|---|---|---|
| white prediction | 0.0188 | 0.0008 | 0 | 0 | 0.0023 | 0.0006 | 0.0002 |
| median refpix, no cleaning | 0.0208 | 0.0060 | 1682 | 122 | 0.0055 | 0.0038 | 0.0029 |
| SIRS, no cleaning | 0.0205 | 0.0053 | 1204 | 121 | 0.0052 | 0.0037 | 0.0027 |
| SIRS + jwst median | 0.0213 | 0.0079 | 403 | 122 | 0.0058 | 0.0046 | 0.0034 |
| SIRS + row_median clipped | 0.0217 | 0.0090 | 397 | 123 | 0.0061 | 0.0048 | 0.0038 |
| SIRS + wiener clipped | 0.0227 | 0.0060 | 217 | 43 | 0.0066 | 0.0051 | 0.0043 |
| SIRS + row_median box32 | 0.0215 | 0.0084 | 222 | 123 | 0.0063 | 0.0049 | 0.0041 |
| SIRS + boxcar box32 | 0.0207 | 0.0056 | 554 | 121 | 0.0058 | 0.0047 | 0.0043 |
| SIRS + wiener box32 | 0.0183 | 0.0044 | 4.5 | 16 | 0.0052 | 0.0046 | 0.0042 |
| SIRS + wiener box128 | 0.0176 | 0.0022 | 4.1 | 16 | 0.0031 | 0.0026 | 0.0025 |
| SIRS + wiener box256 | 0.0175 | 0.0012 | 4.0 | 16 | 0.0023 | 0.0012 | 0.0010 |
| SIRS + wiener, median background | 0.0181 | 0.0015 | 8.7 | 19 | 0.0025 | 0.0009 | 0.0005 |

Reading: the Wiener clean with a linear pipeline removes the 1/f power
from the group differences to below the white floor at all frequencies
above a row period (slow band 1204 to 4, row-to-50-clock band 121 to 16
against a white level of 41 to 44), and it is the only variant that lowers
the pixel rate scatter (0.0205 to 0.0175, with 0.0188 the read-noise
floor for 20 groups). The row median, in any form, raises it. At aperture
scales the limiting factor is the noise of the background model, which the
cleaned rate inherits at scales above the box: a 32-pixel box leaves the
aperture scatter where SIRS had it, 128 halves it, 256 reaches the white
prediction at 9 pixels and gets within a factor 2 to 5 of it at 33 and 101.
On a dark the scalar median background is best of all at large scales,
since there is no sky to protect; on the sky the box size buys protection
of real structure at the price of large-scale noise, and C.3.2 measures that
side (next entry).

### 2026-10-20: chi-square of the reset difference on the 6154 field (B.1.4)

Same exposure, scalar sigma_b 16.8 DN unless stated, two-parameter first
read with k = 0.904 and b = -3.5 DN from the fit above, no pixel lost after
the two-difference fix. `d0_chisq` is the mean squared normalized residual
of the reset difference over pixels where it was used.

| Configuration | chi-square, bright sample | chi-square, all clean pixels |
|---|---|---|
| nominal t1, zero prior mean | 2.50 | 1.85 |
| t1_scale 1.166 (the inverted convention) | 3.24 | 1.76 |
| k, b, zero prior mean | 2.09 | 1.65 |
| k, b, pedestal-mean map as prior mean | 1.96 | 1.48 |
| k, b, sigma_b inflated to 21.1 DN (diagnostic) | 1.57 | 1.12 |

Reading. The all-pixel residual is consistent with an extra 12.7 DN rms of
per-pixel scatter on top of the dark's 16.8 DN, the size of the superbias
error map measured in the dark; using that map as the prior mean recovers
only part of it (1.65 to 1.48), so either the map has moved in the year
between the two epochs or its own 5.6 DN of measurement noise costs as much
as it gains. Inflating sigma_b is not the plan's approach but it shows
where the residual lives. The bright sample stays at 1.6 to 2.0 under
every configuration: the per-channel first-read ratios differ by 4 percent
(0.833 to 0.875), so a single k leaves channel-dependent residuals that a
per-channel (k, b) calibration would remove. That is the natural next step
for the calibration tool, together with storing k and b in the ASDF.

Zero point. With k and b the reset fit's median rate over all pixels is
0.3848 DN/s against 0.3884 for LIKELY and 0.3910 for OLS_C. In a BRIGHT1
5-group ramp the reset difference has about 1/23 of the weight of an
ordinary difference, so a 0.9 percent shift of the median means the sky
pixels' first read sits about 0.9 DN below the two-parameter model fitted
on 50 to 2000 DN pixels. The offset b must therefore be calibrated at the
faint end as well; the field script now reports it from the sky pixels
directly. Until then the reset fit should not be used where the absolute
sky level matters at the percent level.

### 2026-10-20: C.3.2 injected sources on the NRCA1 dark, and the defaults it fixes

Setup: `scripts/exp_inject_part_a.py` on the same 20-group NRCA1 dark; 1500
Gaussian sources (sigma 1.3 px, total flux log-uniform 2 to 2000 DN/s, half
random, half in ten dense patches) and a sky of 0.3 + 0.2 x/2048 + 0.1
sin(2 pi y/400) DN/s added to the jump product as scene times group time
(no Poisson noise, no nonlinearity). Bias is the aperture flux (r = 3 px,
annulus 6 to 10) on the cleaned rate minus that on the uncleaned rate of
the same injected data, over the true flux; the common noise cancels.
"slope vs sky" is the slope of the 64-px-binned cleaning residual against
the injected sky, so -0.09 means 9 percent of the sky structure removed.
Results in `results/2026-10-20_inject_part_a_v2/` and `_v3/`.

| Variant | median bias | bias at 2 to 6 DN/s | 6 to 20 | 20 to 63 | 63 to 200 | slope vs sky |
|---|---|---|---|---|---|---|
| jwst median (clipped, model bkg) | -0.0001 | | | | | -0.126 |
| row_median clipped | -0.0001 | | | | | -0.135 |
| wiener clipped | -0.0012 | | | | | -0.130 |
| row_median box32 | -0.0000 | -0.0046 | -0.0007 | -0.0003 | 0.0000 | -0.083 |
| wiener box32 | -0.0021 | -0.124 | -0.0245 | -0.0070 | -0.0016 | -0.090 |
| wiener box32, min_period 100 | -0.0003 | -0.023 | -0.0042 | -0.0011 | -0.0004 | -0.089 |
| wiener box32, min_period 300 | -0.0001 | -0.010 | -0.0015 | -0.0005 | -0.0002 | -0.088 |
| wiener box32, scene mask at 1 sigma | -0.0004 | -0.052 | -0.0078 | -0.0027 | -0.0006 | -0.230 |
| wiener box128 | -0.0023 | | | | | -0.733 |
| wiener box256 | -0.0022 | | | | | -0.693 |

Two findings.

Faint sources. A low-pass filter removes part of anything the scene mask
did not catch, and the mask (2 sigma on the draft rate) misses sources
below about 20 DN/s here. The loss is 12 percent at 2 to 6 DN/s and still
0.7 percent at 20 to 63 for the full Wiener gain. The median is robust to
unmasked sources (0.5 percent at the faintest, nothing above). Cutting the
Wiener gain to zero at periods shorter than 3 ms (300 clocks, `min_period`, now
the default) makes the filter blind to compact sources and brings the bias
to 1 percent at the faintest bin and below 0.05 percent above 20 DN/s.
Lowering the mask threshold instead costs the sky (slope -0.23).

The price of the cutoff, on the dark (`results/2026-10-20_darks_part_a_v4/`):

| Variant | rate sigma | row-mean sigma | slow-band S | row-to-50 S | ap 9 | ap 33 | ap 101 |
|---|---|---|---|---|---|---|---|
| SIRS, no cleaning | 0.0205 | 0.0053 | 1204 | 121 | 0.0052 | 0.0037 | 0.0027 |
| wiener box32 | 0.0183 | 0.0044 | 4.5 | 16 | 0.0052 | 0.0046 | 0.0042 |
| wiener box32, min_period 100 | 0.0200 | 0.0044 | 3.3 | 70 | 0.0056 | 0.0047 | 0.0042 |
| wiener box32, min_period 300 | 0.0203 | 0.0044 | 2.8 | 111 | 0.0057 | 0.0047 | 0.0043 |

So the 11 percent pixel-rate gain of the full Wiener came from the band
between 0.1 and 3 ms (10 and 300 clocks), which is where the inter-pixel capacitance and
the high-frequency rise of the spectrum live and also where compact
sources live; it cannot be had without biasing faint photometry. What the
cutoff keeps is the removal of row banding and slower noise (row-mean
sigma 0.0053 to 0.0044, slow band 1204 to 3), which is what the row median
is for, done without the median's random walk (row_median box32 gives
0.0084).

Sky versus background box. Any background model smoother than the sky
structure lets the cleaner remove that structure: boxes of 128 and 256
remove 70 percent of a 400-row sinusoid, box 32 removes 8 to 9 percent,
the jwst configuration 13 percent. On darks the large boxes gave the
aperture scatter of a white-noise image, so the gain seen there is not
available on the sky. Within one exposure, 1/f banding and sky structure
at periods of a few hundred rows are degenerate and the model scale picks
the compromise. Box 32 stays the default. At 33 to 101 pixel scales every
box-32 variant is worse than SIRS alone on the dark (0.0046 versus 0.0037
at 33 px) because the cleaned rate inherits the noise of the 32-pixel
background model; a model built from more data than one integration (the
other integrations, or the mosaic) is the way past that.

Defaults fixed by C.3.1 and C.3.2: `fit_method='wiener'`,
`background_method='integration_model'`, `background_box_size` 32 (jwst
default), `clip_sigma=0`, `iterations=1`, `min_period=300`, `n_sigma=2` for
the scene mask. Relative to SIRS alone this removes the row banding and all
slower 1/f from the group differences, leaves the pixel rate scatter and
sky structure essentially unchanged, and biases sources below 20 DN/s by
up to 1 percent. `row_median` with the same linear pipeline is the choice
when faint photometry matters more than row banding.

Field zero point: with the delayed-reset form of t1_scale the reset fit's
median rate is 0.3886 DN/s against 0.3884 for LIKELY; the earlier 0.9
percent deficit came from scaling the first group's read time, which also
stretched the first ordinary interval. Chi-squares are unchanged (1.96 and
1.48).

Confirmation with the final fitter (scalar sigma_b broadcast into the
per-pixel form, delayed-reset t1_scale 0.904, reset_offset -3.5 DN,
pedestal-mean prior): chi-square 1.98 bright, 1.49 all clean pixels;
median rate 0.3885 DN/s against 0.3884 for LIKELY; 3530 of 3530
saturated-from-group-2 pixels recovered, none lost
(`results/2026-10-20_reset_field_final3/NRCA1/`).

### 2026-10-20: saturated-core stars on the 6154 NRCA1 field (C.3.3, photometry)

`scripts/exp_saturated_stars.py` on the rates of the final reset fit
(`results/2026-10-20_reset_field_final3/NRCA1/`). Curve of growth from 410
unsaturated reference stars with peaks above 300 DN/s: core (r <= 2.5 px)
over ring (3.5 to 7 px) = 10.09 with a per-star scatter of 1.10, so the
constant is known to 0.5 percent. 35 stars have a core that saturates from
group 2 on and is fully recovered by the reset fit (no such core is finite
in OLS_C or LIKELY). Their core flux over the ring-predicted core flux,
minus one:

| peak of recovered core (DN/s) | n | median bias | per-star scatter |
|---|---|---|---|
| 2000 to 4000 | 24 | +0.032 | 0.057 |
| 4000 to 8000 | 8 | -0.029 | 0.117 |
| all | 35 | +0.029 | 0.066 |

So the first-read model (k = 0.904, b = -3.5 DN, calibrated on pixels with
first differences of 50 to 2000 DN, that is a t1 below about 1000 DN)
reproduces the photometry of cores at 20 to 80 kDN in the first read to
3 +- 1 percent, with no trend across that range at the present precision.
A per-star scatter of 6 percent is what the ring-based prediction allows
with this PSF; a larger field or several exposures would tighten it.

Per-channel timing is now supported: the field script fits (k, b) per
output channel and `--write-timing` stores detector-wide and per-channel
values in the calibration ASDF (`t1_scale_channels`, `reset_offset_channels`);
`ResetRampFitStep` uses them by default when present, fitting each channel's
block separately, and an explicit `t1_scale` turns that off.

### 2026-10-20: readout ordering on NRCALONG (A.4.3, continued)

jw08758003001_05201_00001_nrcalong, 40 groups, same test as for NRCA1:

| channel | forward minus backward | significance | direction |
|---|---|---|---|
| 0 | 3.4e-4 +- 1.4e-4 | 2.4 sigma | +1 |
| 1 | -0.3e-4 +- 1.2e-4 | 0.2 sigma | undetermined |
| 2 | 2.0e-4 +- 0.4e-4 | 5.6 sigma | +1 |
| 3 | -2.5e-4 +- 0.6e-4 | 3.9 sigma | -1 |

Three of four channels agree with the alternation found on NRCA1 and NRCB3
and nothing contradicts it; the short-range correlation that carries the
signal is weaker on the long-wave detector, and channel 1 would need more
differences. The default `first_direction = 1` stands for NRCALONG.

Per-channel timing on NRCA1 (stored in the v3 calibration file): k =
0.897, 0.897, 0.910, 0.914 and b = -2.2, -2.6, -2.6, -6.0 DN for channels
0 to 3. Fitting each channel with its own values leaves the chi-square of
the bright sample at 1.97 (1.96 with the detector-wide values) and the
all-pixel value at 1.48, while the median rate now equals LIKELY's
(0.3884 DN/s). So the per-channel differences are real but are not what
keeps the bright-pixel chi-square near 2; the remaining candidate is a
pixel-to-pixel spread of the first-read response (k varying per pixel,
for instance from the linearity correction of the first frame), which the
present calibration cannot measure from one field exposure.

### 2026-10-21: readout ordering on NIRISS (A.4.3, second geometry)

jw09279001001_05201_00001_nis (NISRAPID, fastaxis -2, slowaxis -1, the
geometry that transposes), 40 groups of the first integration:

| channel | forward minus backward | significance | direction |
|---|---|---|---|
| 0 | 1.06e-3 +- 0.10e-3 | 10.9 sigma | +1 |
| 1 | -0.93e-3 +- 0.09e-3 | 10.5 sigma | -1 |
| 2 | 0.92e-3 +- 0.14e-3 | 6.6 sigma | +1 |
| 3 | -1.33e-3 +- 0.47e-3 | 2.8 sigma | -1 |

Same alternation as NIRCam, channel 0 toward increasing detector column,
and the short-range correlation is about twice as strong as on NRCA1. The
transpose branch of `science_detector_frame_transform` and the channel
blocks derived from it are therefore right on real data. Default
`first_direction = 1` holds for NIRISS.

### 2026-10-21: sigma_b for NRCALONG and NIRISS, reset fit on the 6154 NRCALONG field

The laptop crashed on the night of the 20th and these runs were repeated.

| Detector | data | pedestal scatter | single-read noise | sigma_b | 16 / 84 pct | pedestal mean |
|---|---|---|---|---|---|---|
| NRCA1 | 9022 BRIGHT1 10 x 10 | 17.6 DN | 5.2 DN | 16.8 DN | 13.5 / 22.2 | -0.03 DN |
| NRCALONG | 9022 BRIGHT1 10 x 10 | 18.3 DN | 4.8 DN | 17.6 DN | 13.1 / 22.6 | -0.07 DN |
| NIRISS | 9279 NISRAPID, 5 ints, first 20 of 120 groups | 23.2 DN | 7.4 DN | 22.0 DN | 12.8 / 33.5 | -0.67 DN |

Files `results/2026-10-20_reset_calib/{nrcalong_bright1_reset_calib_v3,nis_nisrapid_reset_calib}.asdf`.
The NIRISS value rests on 5 integrations and has a wider spread; more
exposures would tighten it, and the second NIRISS dark is still to be
downloaded.

6154 NRCALONG (jw06154001001_02101_00004, BRIGHT1, 5 groups), nominal
timing, zero prior mean: 11036 pixels saturate from group 2 on, 11034 get a
rate from the reset fit, none from OLS_C or LIKELY, no pixel lost. First
read on 855k pixels with first differences of 50 to 2000 DN:

| first difference (DN) | n | median r | median R1 - a t1 (DN) |
|---|---|---|---|
| 50 to 100 | 442431 | 0.869 | -4.3 |
| 100 to 200 | 209520 | 0.912 | -6.0 |
| 200 to 400 | 110158 | 0.923 | -10.6 |
| 400 to 800 | 57550 | 0.929 | -19.3 |
| 800 to 1200 | 19614 | 0.935 | -31.2 |
| 1200 to 2000 | 15351 | 0.944 | -41.9 |

Two-parameter model k = 0.943, b = -2.6 DN (per channel r = 0.898, 0.895,
0.932, 0.905); the long-wave first read is short by 6 percent against 10
on NRCA1, and the faint sky pixels give b = -0.8 DN. Chi-square 1.94
bright, 1.70 all clean pixels; median rate 1.0442 against 1.0433 for
LIKELY with nominal timing. Per-channel timing is stored in the file and
the refit, the saturated-star test and the dark and injection experiments
on NRCALONG follow.

### 2026-10-21: C.3.1 and C.3.2 on the NRCALONG dark

jw08758003001_05201_00001_nrcalong, first integration, 20 groups, same
scripts and metrics as for NRCA1 (`results/2026-10-21_darks_part_a/` and
`results/2026-10-21_inject_part_a/`). The "box32" rows now include the
300-clock cutoff, which became the step default; the row labelled minp300
is the same run. Mask fraction 97.1 percent; white level 37.

| Variant | rate sigma | row-mean sigma | slow-band S | row-to-50 S | ap 9 | ap 33 | ap 101 |
|---|---|---|---|---|---|---|---|
| white prediction | 0.0199 | 0.0008 | 0 | 0 | 0.0022 | 0.0006 | 0.0002 |
| SIRS, no cleaning | 0.0199 | 0.0052 | 1250 | 94 | 0.0053 | 0.0038 | 0.0031 |
| SIRS + jwst median | 0.0202 | 0.0062 | 412 | 95 | 0.0048 | 0.0032 | 0.0024 |
| SIRS + row_median box32 | 0.0204 | 0.0065 | 226 | 95 | 0.0057 | 0.0046 | 0.0041 |
| SIRS + wiener box32 (cutoff 300) | 0.0196 | 0.0047 | 2.8 | 84 | 0.0055 | 0.0046 | 0.0044 |
| SIRS + wiener box256 | 0.0190 | 0.0023 | 2.5 | 84 | 0.0036 | 0.0021 | 0.0015 |

Injection (1500 sources, same sky): median bias 0.0000 for the jwst median
and the row median, -0.0001 for the Wiener; at the faintest bin (2 to 6
DN/s) -0.0024, -0.0022 and -0.0046; sky removed 14, 10 and 10 percent.

Reading. The long-wave detector has less 1/f power relative to its read
noise, and the pattern of NRCA1 repeats in weaker form: the Wiener clean
with the cutoff removes the slow band entirely and is the only variant that
lowers the pixel rate scatter and the row-mean scatter, the row median in
either form raises both. One difference from NRCA1: here the jwst median
step improves the aperture scatter at 33 and 101 pixels (0.0032 and 0.0024
against 0.0038 and 0.0031 for SIRS), while the box-32 variants are worse
there (0.0046, 0.0044), because on this detector the per-row offsets it
removes are larger at those scales than the noise its per-difference
background model adds, and the integration-level model with a 32-pixel box
adds the same noise without removing the slow offsets the model absorbs.
Box 256 gets both. So the background model's noise is the limiting term
on both detectors, and the remedy is a background model with less noise at
the 32-pixel scale rather than a coarser one: the mean difference of all
integrations and exposures of a visit, or a smoothed version of the draft
rate image, would give that.

### 2026-10-21: the reset offset must come from the sky pixels

With b taken from the bright-pixel regression, the reset fit's median rate
on NRCALONG came out 0.46 percent above LIKELY: the sky pixels there sit at
R1 - k a t1 = -0.8 DN where the bright-pixel line extrapolates to -2.6 DN,
so d_0 was biased high for the pixels that set the zero point. The field
script now stores k from the bright-pixel slope and b from the median of
R1 - k a t1 over sky pixels (first difference below 50 DN), detector-wide
and per channel, and the step uses those:

| Detector | k | b (sky) | per-channel k | per-channel b (sky) | chi-sq bright | chi-sq all | median rate, reset / LIKELY |
|---|---|---|---|---|---|---|---|
| NRCA1 | 0.904 | -3.09 | 0.897, 0.897, 0.910, 0.914 | -2.9, -2.0, -1.9, -5.6 | 1.97 | 1.48 | 0.3881 / 0.3884 |
| NRCALONG | 0.943 | -0.84 | 0.946, 0.952, 0.936, 0.943 | -2.0, -2.2, +1.3, -0.1 | 1.69 | 1.48 | 1.0456 / 1.0433 |

NRCA1 is now within 0.1 percent of LIKELY and NRCALONG within 0.2 percent;
the per-channel sky offsets on NRCALONG spread over 3.5 DN, which is why
the per-channel values matter there. Files
`results/2026-10-21_reset_field_{nrca1,nrcalong}_pc3/`.

### 2026-10-21: saturated-core stars with an isolation cut, both detectors

`scripts/exp_saturated_stars.py` now drops reference stars with another
peak above a tenth of their own within the sky annulus, and saturated-core
stars with another peak above a tenth of their recovered core peak. Rates
from the sky-anchored refits above.

| Detector | reference stars | core / ring | per-star scatter | saturated stars | median bias | scatter | 2000 to 4000 DN/s | 4000 to 8000 DN/s |
|---|---|---|---|---|---|---|---|---|
| NRCA1 | 223 | 10.22 +- 0.06 | 0.87 | 30 | +0.016 | 0.069 | +0.024 (n 22) | -0.066 (n 7) |
| NRCALONG | 30 | 10.2 +- 0.5 | 2.57 | 29 | +0.062 | 0.190 | +0.111 (n 20) | -0.066 (n 7) |

NRCA1: the recovered cores agree with the curve-of-growth prediction to
+1.6 +- 1.3 percent, so the first-read model holds from the calibrated
range (a t1 below 1000 DN) up to cores of 20 to 80 kDN in the first read to
a few percent. NRCALONG: the field is too crowded at the long-wave PSF for
a clean reference sample (30 stars with a 25 percent per-star scatter), so
the test is inconclusive at the 5 to 10 percent level; a sparser
long-wave field is needed to repeat it. Earlier NRCALONG runs without the
isolation cut gave +36 percent, which was the reference sample, not the fit.

### 2026-10-21: second NIRISS dark

The file delivered as jw09279002001_02201_00001_nis_uncal.fits is a
byte-for-byte copy of jw09279001001_05201_00001 (same md5, same EXPSTART,
and its FILENAME keyword still names the first exposure), so the NIRISS
sigma_b stays at 22.0 DN from five integrations. A genuine second
exposure, for example jw09279004001_05201_00001 or jw09279005001_04201_00001
from the same program, would double the integrations.

Update, later on 2026-10-21: James re-downloaded jw09279002001_02201_00001
(genuine this time, md5 efb1a6b9..., 2025-07-04, 5 x 120). Calibration on
both exposures, 10 integrations, first 20 groups:
`results/2026-10-20_reset_calib/nis_nisrapid_reset_calib_v2.asdf`,
pedestal scatter 25.5 DN, single-read noise 7.4 DN, sigma_b 24.3 DN,
pedestal mean -0.30 DN. Higher than the 22.0 DN from one exposure: the
scatter across two exposures includes any change of the bias level
between them, which on a science exposure the sky-anchored reset offset
b absorbs, so 24.3 is an upper bound on the per-integration value and
22.0 a lower one; the prior is broad either way. The readout-order check
on the second exposure (`results/2026-10-21_readout_order/`) agrees with
the first: channel 0 forward from the amplifier-boundary and hot-pixel
tests.

### 2026-10-21: the large-scale penalty was the background model's estimator, not its noise

Diagnosis on the NRCA1 dark (two integrations, 20 groups). The Wiener
estimates reproduce the slow row profile of every difference exactly
(correlation 1.00, amplitude ratio 1.00 at 32-row smoothing), the pipeline
fitter and a plain least-squares slope agree to 1e-10, and still the
cleaned rate had 1.4 times the scatter of SIRS at 33 to 101 pixels, with
the excess common to all four channels and only weakly anticorrelated
(-0.32) with the SIRS banding. The resolution: at scales above the
background box the filter removes everything from each difference and the
background model is added back, so the cleaned rate's structure there is
the model's. I had built the model from the mean group difference, which is
the end-to-end estimate of the rate, last group minus first over the
elapsed time. For white noise that estimator has 3.7 times the variance of
the least-squares slope over 20 groups, and more for 1/f, so the cleaned
rate inherited a noisier copy of its own large-scale structure. Doubling the
integrations barely helped (0.0037 to 0.0035 at 33 pixels) because the
excess is in the estimator, not in the number of samples.

`integration_backgrounds` now fits the model to the least-squares slope
of each integration (per group, DQ-aware fallback to the usable
differences), and `exposure_model` averages the slopes over integrations.
Same data, OLS_C rates, two integrations combined, interior and edges:

| Variant | rate sigma | row-mean sigma | slow-band S | ap 9 | ap 33 | ap 101 |
|---|---|---|---|---|---|---|
| white prediction | | 0.0006 | 0 | 0.0016 | 0.0004 | 0.0001 |
| SIRS, no cleaning | 0.0146 | 0.0036 | 1206 | 0.0034 | 0.0022 | 0.0017 |
| SIRS + jwst median | 0.0151 | 0.0053 | 403 | 0.0037 | 0.0027 | 0.0018 |
| SIRS + wiener box32 (mean-difference model, before) | 0.0145 | 0.0033 | | 0.0042 | 0.0037 | 0.0035 |
| SIRS + wiener box32 (slope model) | 0.0141 | 0.0015 | 2.6 | 0.0026 | 0.0017 | 0.0014 |
| SIRS + wiener box32, exposure model | 0.0141 | 0.0015 | 2.5 | 0.0026 | 0.0017 | 0.0014 |
| SIRS + wiener box256 (slope model) | 0.0140 | 0.0007 | 2.5 | 0.0021 | 0.0009 | 0.0006 |

With the slope model the Wiener clean with the default 32-pixel box is
better than SIRS alone at every scale (row banding down 2.4 times, aperture
scatter down 20 to 25 percent at 9 to 101 pixels) and better than the jwst
median everywhere; box 256 reaches the white prediction at 9 pixels and
within a factor 2 to 6 at 33 and 101. The injection check is unchanged
(faintest-bin bias -1.1 percent for box 32, sky structure removed 9
percent for box 32 and 69 percent for box 256), so box 32 stays the
default. Results in `results/2026-10-21_darks_slope_model/` and
`results/2026-10-21_inject_slope_model/`; the single-integration table of
the write-up is being regenerated with the new model.

Single-integration table with the slope model (`results/2026-10-21_darks_part_a_v5/`),
now Table 1 of the write-up: SIRS 0.0205 / 0.0053 / ap 0.0052, 0.0037,
0.0027; jwst median 0.0213 / 0.0079 / 0.0058, 0.0046, 0.0034; Wiener box32
0.0197 / 0.0026 / 0.0039, 0.0028, 0.0024; box128 0.0196 / 0.0018 / 0.0034,
0.0019, 0.0015; box256 0.0195 / 0.0012 / 0.0031, 0.0015, 0.0012; median
background 0.0196 / 0.0016 / 0.0030, 0.0011, 0.0006 (rate sigma / row-mean
sigma / aperture scatter at 9, 33, 101 px). The step now always excludes
the 4-pixel reference border from the fit and the model, and has a
`mask_dilation` option for source wings (James's note of 2026-10-21: the
2-sigma scene mask stops short of the wings of bright sources).

### 2026-10-21: NRCALONG with the slope model, mask dilation, and a NoiseChisel-style scene mask

NRCALONG dark, 20 groups, one integration, slope-based background model
(`results/2026-10-21_darks_part_a_nrcalong_v2/`): SIRS 0.0199 / 0.0052 /
ap 0.0053, 0.0038, 0.0031; jwst median 0.0202 / 0.0062 / 0.0048, 0.0032,
0.0024; Wiener box32 0.0191 / 0.0027 / 0.0042, 0.0031, 0.0029; box256
0.0190 / 0.0014 / 0.0033, 0.0017, 0.0014 (rate sigma / row-mean sigma /
aperture scatter at 9, 33, 101 px). The box-32 clean is now better than or
equal to SIRS at every scale on the long-wave detector too; the earlier
NRCALONG penalty at 33 to 101 px was the mean-difference model.

Mask dilation by 3 px on the NRCA1 injection
(`results/2026-10-21_inject_dilation/`): faintest-bin bias -1.07 to -0.35
percent, but the sky structure removed rises from 9 to 20 percent, because
with 1500 sources plus their grown footprints the 32-pixel background model
loses too many pixels to follow the 400-row sinusoid. A blanket dilation
is the wrong tool; the mask should follow the sources' own wings.

James's notes of 2026-10-21: the jwst scene mask is a per-pixel sigma clip
of the draft rate, with no segmentation and no growth into the wings, and
for real data something like NoiseChisel is wanted. `snowblind.scene_mask.
chisel_mask` follows NoiseChisel's outline (Akhlaghi and Ichikawa 2015;
Gnuastro defaults): Gaussian convolution with a 2-pixel FWHM, a sky model in
32-pixel tiles subtracted first (two passes, the second with the first
pass's detections masked), candidates above the median of the convolved
sky-subtracted image (NoiseChisel's 0.3 quantile lets the convolved noise
of these undersampled images percolate into clusters that pass any
signal-to-noise cut: 18 percent of a pure-noise image masked, against 1.5
percent at the median with a 3-sigma source still caught), two erosions
and one opening to remove what noise produces, a signal-to-noise
test against the pseudo-detections that the same procedure finds in the
sign-flipped image (99th percentile), growth of the survivors back over the
eroded border and into pixels above the 0.9 quantile, holes up to 100
pixels filled. It is the step's default (`scene_mask='chisel'`;
`'clip'` keeps the jwst behaviour). On synthetic tests it masks 98 percent
of a bright source's flux where the 2-sigma clip masks the core only, and
marks under 2 percent of a pure-noise image. The injection experiment
measures it against the clip mask next.

Tuning of the chisel mask, 2026-10-21. Three changes from a literal
transcription of NoiseChisel's defaults, each forced by the data: (1) the
sky is modelled in 32-pixel tiles with a light 3-by-3 filter and
subtracted before thresholding, in two passes, otherwise sky structure
itself is detected (a 400-row sinusoid gave 66 percent of the injected
field masked); (2) the candidate threshold is the median of the convolved
image, not the 0.3 quantile, because at 0.3 the convolved noise of these
undersampled images percolates into clusters of thousands of pixels whose
summed signal-to-noise passes any percentile cut and from which the growth
spreads (18 percent of a pure-noise image masked; 1.5 percent at the
median, with a 3-sigma source still caught and 99.9 percent of a bright
source's flux covered); (3) per-row and per-column medians are removed
before detection, because the draft rate carries the 1/f banding the
cleaner is about to remove and its ridges let detections grow along a band
(growth of 28 to 31 iterations on the dark and the injected field, 19 to
20 after the change, 6 on the dense 6154 field). The growth is data
driven, one pixel per iteration into connected pixels above the 0.9
quantile, and now capped at 16 pixels as a safety net. Masked fractions:
plain dark 1.4 percent, injected field 4.8 (clip 2.8), 6154 NRCA1 field
32.5 (clip about 11). Interim injection numbers with the mask before
change (3): the Wiener's faintest-bin bias -1.07 to -0.42 percent and the
row median's -0.78 to -0.38, sky structure removed unchanged at 9 percent.

Final mask comparison on the NRCA1 injection
(`results/2026-10-21_inject_chisel_v3/`), all with the slope-based
32-pixel background model and the linear pipeline:

| Variant | scene mask | median bias | 2 to 6 DN/s | 6 to 20 | 20 to 63 | sky removed |
|---|---|---|---|---|---|---|
| row median | clip 2 sigma | -0.0001 | -0.0078 | -0.0006 | -0.0003 | 8.8 percent |
| row median | chisel | -0.0000 | -0.0027 | +0.0002 | -0.0001 | 7.6 percent |
| Wiener, 3 ms cutoff | clip 2 sigma | -0.0001 | -0.0107 | -0.0016 | -0.0005 | 9.2 percent |
| Wiener, 3 ms cutoff | chisel | -0.0000 | -0.0026 | +0.0006 | +0.0002 | 7.7 percent |

With the chisel mask the Wiener clean and the row median are equivalent in
photometry down to the faintest injected sources (0.3 percent at 2 to 6
DN/s, nothing above 6), and the sky loss drops slightly for both because
the background model is fitted with fewer source pixels in it. The
`scene_mask='chisel'` default stands.

James, 2026-10-21: removing row and column medians before masking is a
fine first step; an iterative approach is welcome. `StreamCleanStep` has a
`mask_passes` option (default 1): with 2 or more it cleans with the first
chisel mask, builds a draft rate from the cleaned ramp (least-squares slope
per group, free of the banding), masks again and cleans the original data
with the new mask. Each extra pass costs a full cleaning run; not yet
measured on the injection, where the single-pass mask already brings the
faintest-bin bias to 0.3 percent.

Rate-image panels for the write-up (`scripts/make_rate_image_panels.py`,
`results/2026-10-21_rate_images/NRCA1/`), NRCA1 dark, 20 groups, row-mean
scatter of the rate in DN/s: default refpix + OLS_C 0.0060; SIRS + OLS_C
0.0048; SIRS + jwst clean_flicker_noise (median per channel, no
background) + OLS_C 0.0051; SIRS + stream_clean + reset_rampfit 0.0018.
Without the per-difference background model the jwst median no longer
adds blotches, but over 20 groups its random walk gives back as much row
structure as it removed.

### 2026-10-21: even-odd column offsets

The snowblind panel of `dark_rate_images.png` showed faint vertical
structure. Measured on the NRCA1 dark group differences: each amplifier
has a different offset between its even and odd columns, -0.036, -0.023,
+0.035 and -0.055 DN per group for the four amplifiers in one difference,
and the per-column pattern of one integration correlates with the next
at only 0.08, so this is a per-read effect that the superbias cannot
remove, not a fixed pattern. Added `even_odd_columns` (default True) to
`StreamCleanStep`: one even-odd offset per amplifier per difference,
measured from the masked science pixels and subtracted before the stream
filter; linear, so no random walk. Rerun of the darks experiment
(`results/2026-10-21_darks_evenodd/NRCA1/`) and of the rate panels
(`results/2026-10-21_rate_images_v2/NRCA1/`, now used by the figure),
Wiener box 32, off -> on: pixel scatter 0.0197 -> 0.0196, row-mean
0.0026 -> 0.0023, aperture 9/33/101 0.0039/0.0028/0.0024 ->
0.0038/0.0026/0.0022, column-mean scatter of the rate 0.0020 -> 0.0013
DN/s (SIRS alone 0.0020, jwst median 0.0021). In the 512 x 512 panel
region the column-mean scatter of the snowblind rate falls 0.0032 ->
0.0025; the one dark column left is present in the SIRS panel as well.
Nothing got worse. Confirmed along the way that the third panel did run
`clean_flicker_noise` (skip=False, fit_method median, fit_by_channel
True, background_method None); its rate differs from the SIRS rate by
0.0084 DN/s rms.

### 2026-10-21: noise against number of groups (counterpart of Figure 4 of the JWST 1/f page)

`scripts/exp_darks_part_a.py` on the NRCA1 dark for 3 to 100 groups
(`results/2026-10-21_darks_vs_ngroups/`), OLS_C rates, row-mean scatter
and pixel scatter in DN/s:

| groups | SIRS row-mean | jwst median row-mean | snowblind row-mean | SIRS pixel | jwst median pixel | snowblind pixel |
|---|---|---|---|---|---|---|
| 3 | 0.0995 | 0.0550 | 0.0544 | 0.349 | 0.338 | 0.334 |
| 5 | 0.0490 | 0.0331 | 0.0308 | 0.158 | 0.154 | 0.151 |
| 10 | 0.0155 | 0.0124 | 0.0066 | 0.0558 | 0.0550 | 0.0535 |
| 20 | 0.0053 | 0.0079 | 0.0023 | 0.0205 | 0.0213 | 0.0197 |
| 50 | 0.0015 | 0.0046 | 0.0008 | 0.0059 | 0.0074 | 0.0057 |
| 100 | 0.0006 | 0.0032 | 0.0003 | 0.0027 | 0.0042 | 0.0026 |

The jwst median (here with its per-difference background model, the
step's default) halves the row banding at 3 groups, crosses SIRS at about
12 groups and is 5 times worse at 100, where it also raises the pixel
scatter by 55 percent; this is the turnover the JWST documentation's
Figure 4 reports, and the random walk of the group rebuild is its cause.
The snowblind clean is below SIRS at every ramp length, by a factor 1.8 at
3 groups and 2 to 2.3 from 10 groups on, with the pixel scatter 2 to 4
percent below SIRS throughout.

### 2026-10-21: residual column structure and a per-column offset correction

James saw column structure left in the snowblind panel after the even-odd
correction and asked whether a column-wise version of the row cleaning,
as in the CEERS team's striping removal, would work. Diagnostic on the
NRCA1 dark, two integrations, 20 groups, SIRS + stream_clean defaults,
OLS_C rateints, column means over the masked science rows (scratch
script `column_diag.py`):

| quantity | SIRS | stream_clean |
|---|---|---|
| column-mean scatter of the rate, int 1 / int 2 (DN/s) | 0.0020 / 0.0022 | 0.0013 / 0.0016 |
| white-noise expectation for a column mean (DN/s) | 0.00046 | 0.00044 |
| correlation of the column pattern, int 1 vs int 2 | 0.08 | 0.06 |
| correlation between row quarters within int 1 | 0.63 to 0.83 | 0.65 to 0.80 |
| correlation between amplifiers within int 1 | -0.34 to 0.22 | -0.09 to 0.14 |
| per-difference column-mean scatter / white (DN) | 0.61 / 0.17 | 0.22 / 0.16 |
| correlation of column means, first vs last difference | 0.02 | 0.10 |

Reading: the residual is real (3 times white), not a fixed pattern (no
repeat between integrations, so not dark current, superbias or dark
reference), constant along the whole column within an integration (the
quarter correlations are as high as their own white noise allows), white
across column index (flat power spectrum, no even-odd residual) and
independent per amplifier (so not clock-locked pickup). In time it is
mostly a slow drift: per-read column jitter would leave only 0.0004 DN/s
in a 20-group rate, while the observed excess needs about 0.26 DN of
per-column drift over the 215 s integration, and the first and last
differences share a positive column correlation that SIRS alone does not
show. So: one offset per column that wanders during the integration, not
a readout-stream process. The stream filter cannot see it (a column
offset is a once-per-row periodic signal, white across columns, inside
the 3 ms band the filter leaves to the scene).

CEERS (Bagley et al. 2023, section 3.1.3) subtract masked row medians
per amplifier and then masked column medians from the rate image, once.
Inside the ramp a median would random-walk, so the snowblind version is
a masked *mean* per column per difference (`column_offsets` option,
default False until measured), applied per amplifier after the
background model and the even-odd offset and before the stream filter.
Because the OLS slope is a fixed linear combination of the differences,
this equals subtracting the column mean of the rate image and cannot add
noise; columns with fewer than 16 usable pixels get zero. Row banding
does not bias the column means (it adds the same value to every column
of a row), so one pass suffices. `even_odd_pattern` switched from a
median to a mean at the same time for the same reason (no measurable
change at 500k pixels per amplifier). Cost on real data: a bright
source's unmasked wings lift its column mean and are subtracted from all
2040 rows of that column; measured below on the injection.

Runs: `exp_darks_part_a.py ... --out results/2026-10-21_darks_columns
--variants sirs,sirs_wiener_box32,sirs_wiener_box32_columns`,
`make_rate_image_panels.py ... --column-offsets --out
results/2026-10-21_rate_images_v3`, `exp_inject_part_a.py ... --out
results/2026-10-21_inject_columns --variants
sirs_wiener_box32,sirs_wiener_box32_columns`. Results, NRCA1 dark, 20
groups, Wiener box 32, column offsets off -> on: pixel scatter 0.0196 ->
0.0195 DN/s, row-mean 0.0023 -> 0.0023, aperture 9/33/101
0.0038/0.0026/0.0022 -> 0.0037/0.0026/0.0022, column-mean scatter of the
rate 0.0013 -> 0.0010 (SIRS 0.0020). Splitting the column means into a
part smooth over 64 columns and a column-to-column part: SIRS
0.0011/0.0018, box 32 0.0010/0.0010, box 32 + columns 0.0011/0.0002
(white expectation 0.0004; the mean removal takes the white part out
too). So the per-column effect is gone and what is left is the smooth
large-scale profile of the background model, which the step keeps by
design, the same as for rows. A scratch check confirmed the mechanism:
after the correction each cleaned difference's column means over the
step mask scatter by 0.024 DN against 0.17 white, and the rate's column
means are the same over the step mask, the metric mask or all good
pixels. Injection (chisel mask, 1500 sources, off -> on): median bias
0.00 -> 0.00 percent, faintest bin (2 to 6 DN/s) -0.27 +- 2.1 -> -0.20
+- 2.4 percent, sparsest-density bin (27 sources) +0.7 -> -0.01 percent,
gradient slope -0.0771 -> -0.0774, rate change vs reference 0.0110 ->
0.0115 DN/s. No photometric cost at the 0.1 percent level. Rate panels
(`results/2026-10-21_rate_images_v3/NRCA1/`, now used by the figure):
the one dark column left in the previous panel is gone. Default set to
`column_offsets=True`; the experiment scripts' extra variant is now
`sirs_wiener_box32_nocolumns` and the panel script's flag
`--no-column-offsets`.

### 2026-10-21: rate-image panels, one variable only

James asked that the four panels differ only in the cleaning: all fitted
with OLS_C, the jwst step with its default background model, and the
column-mean scatter quoted next to the row-mean. Also asked where the
cutout is (central 512 x 512, rows and columns 768 to 1279, straddling
the two middle amplifiers) and noted that Figure 1 of the jwst-docs page
is the full frame; added a full-frame 2 x 2 figure at one image pixel
per detector pixel (`dark_rate_images_full.jpg`, JPEG quality 85, 9 MB;
the PNG was 44 MB). Panels rerun to
`results/2026-10-21_rate_images_v4/NRCA1/`, with the jwst step saved
both without (`sirs_jwst_median_ols.npz`) and with its background model
(`sirs_jwst_median_model_ols.npz`), and stream_clean fitted with OLS_C
and with reset_rampfit. Scatter of the row and column means in DN/s:

| chain | cutout row | cutout col | full row | full col | full pixel |
|---|---|---|---|---|---|
| default refpix, OLS_C | 0.0060 | 0.0030 | 0.0054 | 0.0021 | 0.0206 |
| SIRS, OLS_C | 0.0048 | 0.0032 | 0.0047 | 0.0020 | 0.0203 |
| SIRS + jwst median, no background, OLS_C | 0.0051 | 0.0026 | 0.0036 | 0.0018 | 0.0207 |
| SIRS + jwst median, background model, OLS_C | 0.0069 | 0.0030 | 0.0047 | 0.0020 | 0.0211 |
| SIRS + stream_clean, OLS_C | 0.0016 | 0.0023 | 0.0022 | 0.0010 | 0.0193 |
| SIRS + stream_clean, reset_rampfit | 0.0018 | 0.0023 | 0.0026 | 0.0006 | 0.0193 |

The background model does change the jwst panel, as the earlier entries
said: worse in the cutout (0.0051 -> 0.0069), level with SIRS over the
full frame (0.0036 -> 0.0047), and its per-difference `Background2D`
tiling is plainly visible as rectangular patches in the full-frame
figure. The cutout is a harder region for the jwst step than the frame
average; the write-up now quotes both. The cutout column-mean scatter of
the snowblind panel (0.0023) is dominated by smooth large-scale structure
over 512 columns; over the full frame the column-to-column part is what
the metric in the previous entry measured.

### 2026-10-21: the smooth gradients left in the cleaned dark

James asked whether the smooth gradients across the fourth full-frame
panel come from the background model fitting the 1/f noise. Diagnostic
(scratch `smooth_diag.py`): two integrations, 20 groups, SIRS and
stream_clean defaults, OLS_C rateints, large-scale component = 64 x 64
block means of the masked rate minus its median, DN/s:

| quantity | SIRS | stream_clean |
|---|---|---|
| block-mean scatter, int 1 / int 2 | 0.0031 / 0.0025 | 0.0024 / 0.0018 |
| white-noise expectation for a block mean | 0.0003 | 0.0003 |
| correlation int 1 vs int 2 | -0.04 | -0.07 |
| correlation SIRS vs clean, int 1 / int 2 | 0.88 / 0.80 | |
| scatter of (clean - SIRS), int 1 / int 2 | 0.0013 / 0.0013 | |
| row-constant part / remaining 2D part, int 1 | 0.0024 / 0.0020 | 0.0022 / 0.0017 |

Answer: yes. The large-scale structure is ten times white noise, does
not repeat between integrations (so it is not dark current, bias or
dark-reference structure), and the cleaned rate's version is 80 to 88
percent correlated with the SIRS rate's, at 80 percent of the amplitude.
It is 1/f noise at scales above the background box, both the slow row
banding (differing per amplifier, which is why half of it looks 2D in
block means) and within-row 1/f at periods above 64 pixel times; the
integration_model background, fitted to this integration's own slope
image, contains it, and the step adds the model back after filtering, so
it survives by design. The clean removes about 20 percent of it and
adds nothing. This is the trend-versus-covariance ambiguity already
noted under Open items: from one exposure the sky and the slow 1/f are
not separable, and the box size is the prior that decides which is
which. On a dark a 256 box or a scalar median background removes it
(Table 1: aperture 101 scatter 0.0024 -> 0.0012 -> 0.0006), and that is
the right setting for darks; on science data the protection of the sky
is the point. The full-frame caption now says this.

### 2026-10-21: no background subtraction on darks; per-channel offsets

James: the amplifier discontinuities in the 1/f are plain in the darks,
a smooth whole-detector background model ignores them, and for darks it
would be better to subtract no background at all, or one scalar per
readout channel; are these options in either step? Options: the jwst
step has `background_method` median, model or None; stream_clean has
median, model, integration_model, exposure_model or None. Neither has a
scalar per channel, and stream_clean does not need one: the Wiener gain
is 1 at zero frequency, so each channel's mean offset in each difference
is removed by the filter whenever the subtracted background is smooth.
What a smooth model does do is blend the banding of one amplifier into
the first 32 to 64 columns of the next and hand it back; measured on the
v4 panel (row-mean scatter of 32-column strips at the 1024 boundary
against the channel centres, and the correlation of the strip's row
means with the neighbouring channel's), there is no detectable ghost.
The larger effect is the one already found: the model carries 1/f
above the box scale and the step reproduces it.

Panels rerun with no background for stream_clean (`--background none`,
`results/2026-10-21_rate_images_v5/NRCA1/`; the jwst no-background
rate was already saved). Scatter in DN/s:

| chain | cutout row | cutout col | full row | full col | full pixel | block 33 | block 101 |
|---|---|---|---|---|---|---|---|
| jwst median, background model | 0.0069 | 0.0030 | 0.0047 | 0.0020 | 0.0211 | 0.0046 | 0.0035 |
| jwst median, no background | 0.0051 | 0.0026 | 0.0036 | 0.0018 | 0.0207 | 0.0017 | 0.0008 |
| stream_clean, integration_model | 0.0016 | 0.0023 | 0.0022 | 0.0010 | 0.0193 | 0.0025 | 0.0022 |
| stream_clean, no background | 0.0008 | 0.0010 | 0.0004 | 0.0002 | 0.0192 | 0.0009 | 0.0003 |
| stream_clean, no background, reset_rampfit | 0.0012 | 0.0010 | 0.0009 | 0.0006 | 0.0191 | 0.0011 | 0.0007 |

White-noise expectations from the pixel scatter: row means 0.0004,
column means 0.0004, 101-pixel blocks 0.0002. So with no background the
stream clean takes a dark to the white-noise floor at every scale; the
smooth gradients of the previous entry are gone with the model that
carried them. The figures (cutout and full frame) now use the
no-background versions for both steps, with the model numbers quoted in
the captions. Note in passing that reset_rampfit on the cleaned dark is
slightly above OLS_C in row and block means (0.0009 against 0.0004):
the pedestal map it anchors to carries its own large-scale structure
from the 10-integration calibration; worth a look when the calibration
is revisited.

### 2026-10-21: NSClean on NIRCam

James asked whether the rate-image panels should include NSClean. In
jwst (3.1.0.dev here) NSClean is `clean_flicker_noise` with
`fit_method='fft'`, and `_check_input` only allows it for the NIRSpec
exposure types NRS_MSASPEC, NRS_IFU, NRS_FIXEDSLIT and NRS_BRIGHTOBJ;
for NRC_IMAGE it logs "Fit method 'fft' cannot be applied" and skips
the step (verified: the output rate is identical to the SIRS input).
The `NSClean` class itself is written for the NRS1/NRS2 detector
geometry (IRS2 zipper, dispersion-direction buffing), so a NIRCam run
would be an off-label port, not the pipeline's method. The median
method in panel 3 is what jwst provides for NIRCam. Not added to the
figure. Noted for the write-up: NSClean's fit is a linear least-squares
fit of a fixed Fourier basis to the masked pixels, so for a fixed mask
and no per-difference background it would share the no-random-walk
property of the Wiener clean; the Wiener filter differs in weighting
frequencies by their measured signal-to-noise instead of fitting a fixed
set, and in filling gaps by normalized convolution instead of an
explicit solve.

### 2026-10-21: Detector1 defaults, snowballs, and the "default refpix" label

James asked whether snowball removal is on in the jump step for the
OLS_C fits, and whether to use the default Detector1Pipeline settings
instead of `run_detector1_to_jump`. Checked with
`Detector1Pipeline.get_config_from_reference` on the NRCA1 dark: the
pipeline parameter reference skips ipc and persistence, sets
`refpix_algorithm='sirs'`, and sets the jump parameters
(`expand_large_events=True`, `min_jump_area=5`, `expand_factor=2`,
`sat_required_snowball=True`, `find_showers=False`), identical to the
jump step's own parameter reference that the helper already pulls (266
snowballs flagged on the 20-group dark). The remaining Detector1 steps
(emicorr, rscd, firstframe, lastframe, reset, picture_frame,
charge_migration, clean_flicker_noise) are MIRI- or NIRSpec-only or
skipped by default. So the helper's chain (group_scale, dq_init,
saturation, superbias, refpix sirs, linearity, dark_current, jump) is
the pipeline's default chain for NIRCam; no change needed. One
correction: the figure's first panel was labelled "default refpix", but
SIRS is now the Detector1 default for NIRCam through the pipeline
parameter reference, and `median` is only the refpix step's own default.
Relabelled "median refpix (pre-SIRS default)" and reworded the write-up.

### 2026-10-21: Omega Cen panels, and a sky bias from the background model

James asked for the dark_rate_images comparison on Omega Cen NRCA1 and
NRCALONG (exposure jw06154001001_02101_00001, BRIGHT1, 5 groups, a
different dither from the one the reset timing was calibrated on).
`scripts/make_field_panels.py`, `results/2026-10-21_field_panels/`:
SIRS + OLS_C; SIRS + jwst clean_flicker_noise as run for this program
(median, fit_by_channel False, flat applied, science regions masked,
default background) + OLS_C; SIRS + stream_clean + OLS_C; the same +
reset_rampfit. Figures `field_rate_images_{nrca1,nrcalong}.png` (cutout)
and `_full.jpg`.

First result was bad: the stream_clean panels showed dark streaks along
rows next to bright stars and a lowered sky. Measured on the background
(chisel-mask) pixels, stream_clean minus SIRS was -0.038 DN/s on NRCA1
(16 percent of the 0.235 DN/s sky) and -0.22 DN/s on NRCALONG (34
percent of 0.65), growing with source density but present even in
blocks with more than 90 percent background (-0.024 / -0.15); the jwst
step changed the sky by +0.0004 / -0.005. Per difference on NRCA1: masked
mean of the data 5.6 DN per group, of the integration_model background
4.53, and the filter removed the difference, 1.1 to 1.9 DN per group,
because its gain is 1 at zero frequency: whatever the model misses at
scales above the box is treated as noise. The model came from the jwst
`background_level` helper with `method='model'`: photutils Background2D
with a 3-sigma clipped median per box, a 5 x 5 box filter and the
default `exclude_percentile=10`, which drops every box more than 10
percent masked and interpolates it from the emptiest boxes. On a dark
that is unbiased; on a crowded field with a skewed background
distribution it sits 15 to 20 percent below the masked mean.

Fix: `masked_mean_background` in stream_clean.py, Background2D with a
mean estimator, no sigma clipping, `exclude_percentile=100`, 3 x 3
filter, used for integration_model and exposure_model (the 'model'
method still calls the jwst helper). Source rejection is the scene
mask's job, not the model's. Checked on the slope image of the NRCA1
exposure: masked mean of the slope 5.542 DN per group, of the new model
5.565 (ratio 1.004; 0.999 with filter 1), plain block means 5.542. With
it the sky change becomes +0.0013 DN/s in the mean (+0.5 percent, within
the jwst step's +0.0010), flat across source density, median +0.008
because the change distribution is broad and skewed: 5th/95th
percentiles -0.086/+0.063 against -0.029/+0.032 for the jwst step, the
negative tail sitting within 8 pixels of masked sources. Full-frame
row-mean scatter of the background: SIRS 0.0248, jwst 0.0076 (from the
earlier panel run), stream_clean 0.0132 DN/s. So on this field the
stream clean is unbiased in the mean but removes scene next to bright
stars (the streaks) and cleans the rows less well than the row median.
Variants under test on the saved NRCA1 jump model (mask dilation 4,
min_period 1000, row_median, box 64 + dilation); dark and injection
regression runs with the new model
(`results/2026-10-21_darks_meanmodel`, `_inject_meanmodel`) in progress.

Results of the variants on the NRCA1 exposure (saved jump model, OLS_C,
background = chisel-mask pixels, full frame; "change" is rate minus the
SIRS rate over those pixels, DN/s; "near" within 8 pixels of a masked
source, "far" beyond 16):

| variant | row-mean scatter | col-mean scatter | change mean | median | p5 | p95 | p5 near sources |
|---|---|---|---|---|---|---|---|
| SIRS | 0.0248 | 0.0337 | 0 | 0 | 0 | 0 | 0 |
| jwst as run for 6154 | 0.0157 | 0.0334 | +0.0010 | +0.0004 | -0.029 | +0.032 | -0.029 |
| stream_clean defaults, mean model | 0.0132 | 0.0306 | +0.0013 | +0.0083 | -0.086 | +0.063 | -0.084 |
| + mask_dilation 4 | 0.0137 | 0.0300 | -0.0044 | +0.0056 | -0.110 | +0.064 | -0.109 |
| + min_period 1000 | 0.0143 | 0.0304 | +0.0012 | +0.0054 | -0.064 | +0.052 | -0.064 |
| + box 64, dilation 4 | 0.0126 | 0.0310 | -0.0004 | +0.0092 | -0.108 | +0.072 | -0.108 |
| row_median | 0.0167 | 0.0314 | +0.0150 | +0.0159 | -0.034 | +0.060 | -0.035 |
| row_median, anchor_mean | 0.0168 | 0.0303 | +0.0012 | +0.0022 | -0.047 | +0.046 | -0.049 |
| row_median, anchor_mean, dilation 2 | 0.0164 | 0.0299 | -0.0034 | -0.0023 | -0.052 | +0.042 | -0.054 |

(The figure titles quote smaller row-mean scatters because their
statistics mask also drops pixels more than 5 sigma from the sky, which
removes most wing pixels; with that mask the jwst step reads 0.0076 and
the stream clean 0.0135 on NRCA1, i.e. the stream clean's residual row
structure sits in the faint-sky pixels, which is the streaks.)

Reading: on this field the Wiener clean removes the most banding and is
unbiased in the mean, but it is a linear, non-robust estimator and the
unmasked wings of the saturated stars (Omega Cen is wings everywhere;
the chisel mask covers 30 percent of the pixels) are read as slow 1/f
and subtracted along the row, which is the negative tail and the dark
streaks next to bright stars. The jwst row median is robust to that
(its tails are 3 times smaller) at the cost of less banding removal.
Mask dilation does not help (the model is then built from fewer, fainter
pixels and the sky drops); a 10 ms cutoff (min_period 1000) trims the
tail by 25 percent. A new `anchor_mean` option (default True; a no-op
for the Wiener filter, whose gain is already 1 at zero frequency) sets
each channel's mean removed noise to the masked mean of the residual,
which makes `fit_method='row_median'` unbiased (+6 percent -> +0.5) with
the jwst-like robustness; it is the right choice for a field like this
one, the Wiener filter for darks and sparse fields. The real fix for
crowded fields is a scene model from the dithered mosaic (Open items),
which removes the wings before the filter sees them.

Regression with the masked-mean model, NRCA1 dark 20 groups
(`results/2026-10-21_darks_meanmodel`): pixel scatter 0.0196 -> 0.0195,
row-mean 0.0023 -> 0.0025, column-mean 0.0013 -> 0.0011, aperture
9/33/101 0.0038/0.0026/0.0022 -> 0.0039/0.0028/0.0026; the mean picks up
slightly more small-scale structure than the clipped median on a dark.
Injection (`results/2026-10-21_inject_meanmodel`): median bias 0.00 ->
0.00 percent, faintest bin -0.27 -> -0.23 percent, sky-gradient slope of
the residual -0.077 -> -0.017 and binned residual 0.0101 -> 0.0021 DN/s,
so the injected sky is preserved five times better. Net: a small loss at
large scales on darks, a large gain on sky and crowded fields; kept.

Second look at the same variants with the figure's statistic (background
pixels within 5 sigma of the sky; the all-background version above is
dominated by a few extreme unmasked pixels per row and is not useful),
NRCA1, full frame, DN/s, plus three new variants:

| variant | row-mean scatter | col-mean scatter | change mean | p5 | p95 |
|---|---|---|---|---|---|
| SIRS | 0.0210 | 0.0286 | 0 | 0 | 0 |
| jwst as run for 6154 (scalar median background, flat applied) | 0.0076 | 0.0282 | +0.0010 | -0.029 | +0.032 |
| stream_clean defaults (integration_model, box 32) | 0.0175 | 0.0284 | +0.0016 | -0.084 | +0.063 |
| + box 256 | 0.0133 | 0.0222 | -0.0009 | -0.100 | +0.074 |
| row_median, anchor_mean, box 32 model | 0.0137 | 0.0287 | +0.0012 | -0.047 | +0.046 |
| row_median, scalar median background | 0.0035 | 0.0023 | -0.0018 | -0.073 | +0.061 |
| wiener, scalar median background | 0.0023 | 0.0028 | -0.0017 | -0.080 | +0.066 |
| wiener, scalar median background, flat applied | 0.0029 | 0.0037 | -0.0017 | -0.082 | +0.068 |

So the box-32 integration model was the limiting choice on this field,
not the estimator: the model follows the banding (and the wings) at
scales above 32 pixels and the step hands that back, exactly as on the
dark. With a scalar background, which is what the jwst run uses too,
the Wiener clean reaches 0.0023 in row means and 0.0028 in column means
against 0.0076 and 0.0282 for the jwst run, with the mean sky changed
by -0.7 percent. The negative tail (streaks next to bright stars, p5
about -0.08) remains with every Wiener variant; it is local and does
not drive the row means. Applying the flat first does not help the
stream clean. Guidance: darks, no background; flat-sky fields, scalar
median; fields with real extended structure (nebulae, large galaxies,
strong sky gradients), integration_model, whose job is to protect that
structure (the injected 400-row sky sinusoid is destroyed by a scalar
background, kept by the box-32 model). The default stays
integration_model as the safe choice; the panel scripts take
`--background`. Field panels rerun with `--background median` for the
figures (`results/2026-10-21_field_panels_v3`).

Field figures with `--background median` (`results/2026-10-21_field_panels_v3`),
statistics over mask pixels within 5 sigma of the sky, DN/s:

| detector | chain | full row | full col | cutout row | cutout col | sky change | p5 / p95 |
|---|---|---|---|---|---|---|---|
| NRCA1 | SIRS, OLS_C | 0.0210 | 0.0286 | 0.0269 | 0.0191 | 0 | |
| NRCA1 | jwst as run, OLS_C | 0.0076 | 0.0282 | 0.0179 | 0.0195 | +0.4% | -0.028 / +0.032 |
| NRCA1 | stream_clean median bkg, OLS_C | 0.0023 | 0.0028 | 0.0052 | 0.0113 | -0.7% | -0.080 / +0.066 |
| NRCA1 | stream_clean median bkg, reset_rampfit | 0.0062 | 0.0035 | 0.0080 | 0.0116 | -0.5% | -0.088 / +0.078 |
| NRCALONG | SIRS, OLS_C | 0.0863 | 0.0999 | 0.0521 | 0.0342 | 0 | |
| NRCALONG | jwst as run, OLS_C | 0.0290 | 0.0999 | 0.0418 | 0.0343 | +1.5% | -0.074 / +0.111 |
| NRCALONG | stream_clean median bkg, OLS_C | 0.0063 | 0.0093 | 0.0147 | 0.0236 | -3.3% | -0.338 / +0.192 |
| NRCALONG | stream_clean median bkg, reset_rampfit | 0.0073 | 0.0102 | 0.0144 | 0.0240 | -3.5% | -0.340 / +0.195 |

The reset fit adds 1646 (NRCA1) and 6857 (NRCALONG) finite pixels over
OLS_C (saturated cores). Its row-mean scatter on NRCA1 is 0.0062 against
0.0023 for OLS_C on the same cleaned ramp: on 5 groups the pedestal
prior contributes row structure of its own; to look at. The NRCALONG
sky change of -3.3 percent comes from the cluster core on the left of
the frame (wings everywhere, little mask); the streak problem, not a
bias of the model. Write-up section "A crowded field: Omega Cen" added
with the four figures.

### 2026-10-21: the pedestal prior and the first read

James asked to look at why reset_rampfit showed more row structure
than OLS_C on the cleaned Omega Cen ramp (NRCA1 full frame 0.0062
against 0.0023 DN/s). Two routes were possible: the pedestal map prior
carrying its own structure, and the first read, which stream_clean
left untouched by default (`clean_first_group=False`) and which OLS_C
never uses while the reset fit uses group 0 minus the prior as a
difference with about a tenth of a group's weight.

The map (`pedestal_mean`, 10 integrations of the 9022 BRIGHT1 dark,
after superbias and SIRS): pixel robust sigma 13.9 DN, row-mean
scatter 3.5 DN against 0.3 expected from white noise, column-mean
scatter 1.5 DN; per channel 3.6 to 4.0 DN. Split into two halves of
five integrations (scratch `pedestal/split.py`): single-integration
row-mean scatter 5.6 to 6.3 DN; half means 3.8 and 4.0 DN, correlated
at 0.68 in their row means; the common part is about 3.2 DN and the
noise part 2.1 DN per half, 4.7 DN per single integration, 1.5 DN in
the ten-integration mean. So, as James said, the map does not carry
much 1/f of its own: its row structure is mostly a repeatable
difference between this epoch's reset level and the superbias
reference, present in every first read too. Per pixel: single
pedestal scatter 23.2 DN, ten-integration mean 14.0, implying about
12.6 DN of fixed per-pixel pattern and 19.6 DN of per-integration
scatter (sigma_b-like).

Field test on the saved NRCA1 jump model (scratch
`fielddiag/reset_rows.py`, median background), row-mean / column-mean
scatter of the sky pixels, DN/s:

| first read | prior | row | column |
|---|---|---|---|
| untouched | OLS_C (no reset) | 0.0023 | 0.0028 |
| untouched | map | 0.0062 | 0.0035 |
| untouched | scalar | 0.0066 | 0.0031 |
| cleaned against zero | map | 0.0044 | 0.0041 |
| cleaned against zero | scalar | 0.0023 | 0.0031 |
| cleaned against zero | map, high-passed | 0.0024 | 0.0031 |

Map and scalar priors give the same row structure with the first read
untouched, so the first read is the route, not the map. Cleaning the
first read with the scalar prior reaches OLS_C; with the raw map it
overshoots, because the map's real row pattern then enters with the
opposite sign; high-passing the map (per-channel row medians, even-odd
and column offsets removed, channel mean kept) restores OLS_C level
with the per-pixel prior intact. Bright-pixel rate ratio reset/OLS_C
1.0010 in all cases.

Changes: `clean_first_group` default True in StreamCleanStep (harmless
for OLS, needed for the reset fit); `prior_highpass` option (default
True) and `highpass_prior` in reset_rampfit; and, cleaner than
cleaning the first read against zero, a `reset_calib` option in
StreamCleanStep that cleans the first read against the reset prior
itself (`reset_prior_map`, shared by both steps: high-passed map plus
the channel offsets), so the filter only ever sees the reset frame's
and the first read's 1/f and never the real pattern or the channel
offsets. (Against zero, the filter's zero-frequency gain also removes
the reset level's channel mean from the first read, which the reset
step then adds back through its offsets: a 2b error of about 6 DN in
the first difference, 0.0007 DN/s in the rate, small but avoidable.)
Field panels rerun with it (`results/2026-10-21_field_panels_v4`).

Field panels v4 (`--background median`, stream_clean with `reset_calib`
so the first read is cleaned against the reset prior; the reset step
with the high-passed map prior), sky pixels, DN/s:

| detector | chain | full row | full col | cutout row | cutout col | sky change | finite |
|---|---|---|---|---|---|---|---|
| NRCA1 | SIRS, OLS_C | 0.0210 | 0.0286 | 0.0269 | 0.0191 | 0 | 4144553 |
| NRCA1 | jwst as run, OLS_C | 0.0076 | 0.0282 | 0.0179 | 0.0195 | +0.4% | 4144553 |
| NRCA1 | stream_clean, OLS_C | 0.0023 | 0.0027 | 0.0052 | 0.0114 | -0.7% | 4144553 |
| NRCA1 | stream_clean, reset_rampfit | 0.0024 | 0.0028 | 0.0052 | 0.0110 | -0.5% | 4146199 |
| NRCALONG | SIRS, OLS_C | 0.0863 | 0.0999 | 0.0521 | 0.0342 | 0 | 4093475 |
| NRCALONG | jwst as run, OLS_C | 0.0290 | 0.0999 | 0.0418 | 0.0343 | +1.5% | 4093475 |
| NRCALONG | stream_clean, OLS_C | 0.0063 | 0.0093 | 0.0146 | 0.0236 | -3.3% | 4093475 |
| NRCALONG | stream_clean, reset_rampfit | 0.0062 | 0.0090 | 0.0140 | 0.0235 | -3.5% | 4100332 |

The reset fit now matches OLS_C in row and column structure on both
detectors (NRCA1 0.0024 against 0.0023; NRCALONG 0.0062 against 0.0063)
while fitting 1646 and 6857 more pixels (saturated cores); bright
unsaturated pixels agree with OLS_C to 0.10 percent (NRCA1) and 0.05
percent (NRCALONG). The figures use these panels. 190 tests pass.

### 2026-10-21: superbias bookkeeping for the pedestal prior

James: superbiases nearer in time are what CRDS bestref provides, so
the step should depend on reftype superbias. Checked which files the
data got: NRCA1 calibration dark (9022, 2026-03-27), NRCA1 Omega Cen
(2025-03-28) and the 8758 dark (2025-11-04) all get
jwst_nircam_superbias_0454 (useafter 2025-01-29); NRCALONG dark and
field both get 0443; NIRISS darks 0257. So the existing priors are
consistent, by the context rather than by design. Implemented:
`write_calibration(..., superbias=...)` records the name the darks had
subtracted (`meta.ref_file.superbias.name`, written by the pipeline's
superbias step; `snowblind-reset-calib` refuses darks with different
files); ResetRampFitStep declares `reference_file_types = ["readnoise",
"gain", "superbias"]`, reads the data's name, logs a warning when it
differs from the current best reference, and `calibrated_pedestal_map`
corrects the map by (superbias_calibration - superbias_data) read from
the CRDS cache when the names differ, before the high-pass; the same
helper serves `reset_prior_map`, so stream_clean's first-read reference
gets the same correction. The three existing calibration files were
given their superbias names after the fact (provenance.superbias with a
note). Unit tests stay free of CRDS files (James: large, hard to
deploy): `superbias_difference` is tested with an injected file
resolver, the step with `override_superbias` pointing at a local file.
Real path verified on the NRCA1 field (same superbias on both sides, no
correction, 4146199 finite pixels). 195 tests pass.

### 2026-10-21: scene model from the dithered mosaic (in progress)

James chose the mosaic route next, with skymatch and tweakreg skipped.
Design: `scripts/run_mosaic_pass.py` runs, for every _uncal of one
detector, Detector1 to jump (SIRS), stream_clean (scalar median
background in pass 1, the scene models afterwards; first read cleaned
against the reset prior), reset_rampfit, saved as _rate; Image2 with
photom and resample skipped, so the cal files are flat-fielded DN/s;
Image3 on all exposures with tweakreg, skymatch and source_catalog
skipped and outlier_detection saving its intermediate products, which
include the median mosaic blotted back into each exposure's frame
(_blot.fits). `scripts/make_scene_models.py` multiplies each blot by
the flat the Image2 run used (so the scene is in the units of the
group data, DN/s before flat-fielding) and writes <root>_scene.fits;
pixels without coverage or with an unusable flat are NaN.
StreamCleanStep gained `scene_model`: the image times the group time
is the fixed background of every difference (the first read gets the
first-group time ratio as before, on top of the reset prior), the
chisel mask is built on the residual of the draft rate against the
scene, so it only covers where the model is wrong (saturated cores,
undersampled peaks, variable sources), and pixels without coverage are
excluded. Pass 1 is running on the four NRCA1 exposures of 6154
(`results/2026-10-21_mosaic_nrca1/pass1`); pass 2 uses its scenes.
Metrics to compare against the v4 panels: row and column scatter of
the sky pixels, sky change, the change-distribution tails near
sources (the streaks), and star photometry pass 2 against pass 1.

Pass 1 and pass 2 results (NRCA1, 4 exposures; `scripts/compare_mosaic_passes.py`,
`results/2026-10-21_mosaic_nrca1/pass2/compare.json`). Pass 1 (scalar
median background, first read against the reset prior) reproduces the
v4 panel numbers. Its blots: units DN/s (photom skipped), 3 to 4
percent of each frame without coverage, and on an exposure's sky
pixels the blot sits 5 percent above the exposure (median residual
-0.013 DN/s on 0.236; selection of the faintest pixels plus the
smoothing of the drizzle-and-blot), with a row-mean scatter of the
residual of 0.006 to 0.012 DN/s, two to three times the cleaned
exposure's own 0.003 to 0.005. Pass 2 (the blots as scene models,
chisel mask on the residual so that 96.6 percent of pixels count as
background against 70.8): on the v4 sky pixels of exposure 1, row-mean
scatter 0.0134 against 0.0024 in pass 1, column-mean 0.0166 against
0.0027, sky change +10.2 percent against -0.5, near-source 5th
percentile -0.070 against -0.099 (the one gain: the streaks are
shallower). Star photometry pass 2 over pass 1: 0.998 to 1.001 in four
flux bins, so no flux was moved, only the sky and its structure.
Mechanism, measured: the mean of (data - scene) over the pass-2 mask
is -0.055, -0.020, -0.018, -0.019 DN/s per channel (the blot is above
the data over the wings while the cores, where it is below, are
masked as model errors); the filter's unit gain at zero frequency
removes that and the sky rises by exactly those amounts (+0.053,
+0.015, +0.013, +0.014). The added row structure correlates at 0.49
with the row means of (scene - pass 1): the blot's error field is
imprinted. That error field is 0.014 DN/s rms in 16-pixel blocks on
the sky pixels, 6 percent of the sky, larger than the 1/f being
removed. On a field this crowded and undersampled a four-dither
drizzled median blotted back is not accurate enough at the slow scales
to serve as the subtracted scene; its errors dominate the result.
Variants under test on exposure 1: the scene with the conservative
(pass-1) star mask, and the scene re-anchored per channel to the data's
sky level.

Variants on exposure 1 (OLS_C, v4 sky pixels, DN/s):

| variant | row | col | sky change | p5 / p95 | near-source p5 |
|---|---|---|---|---|---|
| pass 1, scalar median background | 0.0024 | 0.0027 | -0.5% | -0.088 / +0.076 | -0.099 |
| scene, conservative (pass-1) star mask | 0.0111 | 0.0082 | +4.8% | -0.059 / +0.074 | -0.067 |
| scene re-anchored per channel, conservative mask | 0.0107 | 0.0071 | -1.8% | -0.082 / +0.061 | -0.091 |
| scene re-anchored, residual mask | 0.0130 | 0.0137 | +3.6% | -0.082 / +0.085 | -0.090 |

(Re-anchoring shifts: -0.046, -0.006, -0.006, -0.006 DN/s per channel;
channel 0 holds the cluster core.) Re-anchoring repairs the sky level
and the conservative mask trims the streaks a little, but the row and
column structure stays four times worse than the scalar background:
the blot's slow-scale error is imprinted whatever the mask or the
level. Conclusion for Omega Cen: the blotted four-dither median is not
an adequate scene model; the per-exposure scalar background with the
chisel mask remains the configuration (v4 panels). The scene_model
option, the mosaic pass script and the scene script stay in the tree
for fields where the blot error is small (sparse fields, smooth
extended emission, more dithers), which is also where the box-size
ambiguity bites; untested. What would make it work on a crowded
undersampled field is a scene model whose slow-scale error is below
the 1/f, which needs a PSF-based model of the stars rather than a
resampled image.

James asked whether the blot adds noise as a per-pixel background and
whether a smoother blot (kernel choice, or smoothing the blot before
use) would help. The jwst resample kernel is 'square' with pixfrac 1
and the blot is drizzle's polynomial interpolation of the mosaic, not a
sinc. Decomposition of the (re-anchored) scene's error against the
pass-1 rate on the sky pixels: row-mean scatter 0.0066 DN/s;
16-pixel block scatter 0.0103, and 0.0076 after removing the row
means. So a third of the slow error is row-aligned, which is the other
dithers' residual banding carried into the mosaic (a dither moves a
band to another row but keeps it a row), and the rest is the smoothing
of thousands of undersampled stars. The blot's pixel noise is not the
issue: only its low-pass part reaches the cleaned data, and that is
below 0.002 DN/s. Variants on exposure 1 (anchored scene, conservative
mask, OLS_C):

| scene variant | row | col | sky change | near-source p5 |
|---|---|---|---|---|
| anchored (reference) | 0.0107 | 0.0071 | -1.8% | -0.091 |
| Gaussian-smoothed, 4 px | 0.0219 | 0.0189 | +14.6% | -0.059 |
| Gaussian-smoothed, 16 px | 0.0459 | 0.0313 | +64.3% | +0.037 |
| per-channel row means removed | 0.0068 | 0.0071 | -1.6% | -0.088 |
| row means removed, then smoothed 4 px | 0.0135 | 0.0187 | +14.0% | -0.065 |
| pass 1, scalar median background | 0.0024 | 0.0027 | -0.5% | -0.099 |

Smoothing goes the wrong way, and strongly: on a crowded field it
spreads star light into the sky pixels, the model sits further above
the data there, and the filter's unit gain at zero frequency lifts the
sky by the difference (+15 percent at 4 pixels, +64 at 16). Removing
the row-aligned part of the model recovers 40 percent of the row
structure, confirming the banding carry-over, but the remainder is
still three times the scalar background. Conclusion unchanged: on this
field the subtracted scene must be either noise-free and structure-free
(scalar) or accurate to better than the 1/f at the slow scales, and a
resampled image of an undersampled crowded field is neither. A
smoother kernel would be the right lever only on a smooth scene
(extended emission, sparse field), where the smoothing error is small.

### 2026-10-22: reset calibration for all ten NIRCam detectors

`scripts/run_reset_calib_batch.sh`, one detector at a time (memory):
calibration from the 9022 BRIGHT1 10 x 10 dark of each detector
(`results/2026-10-22_reset_calib/<det>_reset_calib.asdf`, with the
superbias recorded and ASDF history entries), then the per-channel
first-read timing from the 6154 Omega Cen exposure 00004 of the same
detector (`exp_reset_field.py --use-pedestal-mean --write-timing`,
`results/2026-10-22_reset_field/<DET>/`). Summary in
`results/2026-10-22_reset_calib/summary.json`:

| detector | sigma_b (DN) | single-read noise (DN) | pedestal median (DN) | pedestal row scatter (DN) | t1_scale | per-channel t1_scale | reset offsets b (DN) | superbias | saturated-from-group-2 pixels fitted |
|---|---|---|---|---|---|---|---|---|---|
| NRCA1 | 16.83 | 5.19 | -0.03 | 3.5 | 0.904 | 0.897, 0.897, 0.910, 0.914 | -2.9, -2.0, -1.9, -5.6 | 0454 | 3530 of 3530 |
| NRCA2 | 15.82 | 5.70 | +0.01 | 4.4 | 0.930 | 0.933, 0.932, 0.931, 0.918 | -6.3, -7.2, -5.0, -6.3 | 0416 | 2775 of 2775 |
| NRCA3 | 14.97 | 4.55 | -0.16 | 2.5 | 0.883 | 0.880, 0.902, 0.895, 0.871 | -3.0, -3.7, -2.8, -3.8 | 0435 | 1879 of 1879 |
| NRCA4 | 16.08 | 4.99 | +0.01 | 4.4 | 0.908 | 0.906, 0.904, 0.903, 0.915 | -7.9, -6.4, -9.2, -6.1 | 0440 | 1456 of 1456 |
| NRCALONG | 17.64 | 4.78 | -0.07 | 3.2 | 0.943 | 0.946, 0.952, 0.936, 0.943 | -2.0, -2.2, +1.3, -0.1 | 0443 | 11034 of 11036 |
| NRCB1 | 20.80 | 5.62 | -2.44 | 6.0 | 0.914 | 0.916, 0.911, 0.932, 0.899 | +3.2, +4.1, +2.6, +1.7 | 0397 | 668 of 668 |
| NRCB2 | 16.36 | 5.04 | -0.17 | 3.0 | 0.880 | 0.896, 0.862, 0.904, 0.876 | +1.7, +2.0, +4.0, +3.2 | 0393 | 1308 of 1308 |
| NRCB3 | 17.15 | 5.60 | -0.16 | 3.0 | 0.882 | 0.891, 0.918, 0.895, 0.864 | +2.4, +2.9, +3.1, +2.9 | 0383 | 1667 of 1667 |
| NRCB4 | 17.42 | 4.86 | +0.30 | 3.1 | 0.881 | 0.906, 0.867, 0.871, 0.893 | -0.7, -2.2, -0.2, -1.0 | 0438 | 1916 of 1916 |
| NRCBLONG | 17.80 | 4.90 | -0.04 | 3.0 | 0.942 | 0.940, 0.936, 0.943, 0.953 | +1.5, +0.7, +2.4, +1.9 | 0434 | 8071 of 8074 |

NRCA1 reproduces the v3 file exactly. The first-read shortfall is 6 to
12 percent on every detector (t1_scale 0.88 to 0.94), largest on the
short-wave B module and smallest on the two long-wave detectors, with
per-channel spreads of 1 to 5 percent; the offsets b are a few DN,
negative on the A module and positive on B1 to B3. The reset fit
assigned a rate to every pixel saturated from group 2 on all detectors
and lost none relative to OLS_C. NRCB1 stands out: sigma_b 20.8 against
15 to 18 elsewhere, pedestal median -2.4 DN and row scatter 6 DN
against about 3: its superbias reference (0397) is likely from an
earlier epoch than the others; checked below.
Superbias epochs: all five A-module files have useafter 2025-01-29
(pedigree INFLIGHT 2025-01-29 to 2025-11-04) and all five B-module files
useafter 2025-02-09 (INFLIGHT 2025-02-09 to 2025-12-03), all delivered
2026-10-13. So NRCB1's larger reset scatter, -2.4 DN pedestal offset and
6 DN row pattern are not a superbias epoch effect; they are properties
of that detector relative to its superbias, and the sky-anchored
offsets absorb the level. Worth a look at B1 on a field before trusting
its map prior.

### 2026-10-22: NIRISS measurements on the darks

Calibration from both 9279 NISRAPID darks, first 20 of 120 groups, 10
integrations. The new superbias guard refused them at first: the two
exposures (observed months apart) had different superbias files
subtracted, 0252 and 0257, so the earlier two-dark file
(`nis_nisrapid_reset_calib_v2.asdf`, 24.3 DN) had mixed two references.
`snowblind-reset-calib` now expresses every dark relative to one
reference superbias, the latest delivery among the inputs, by adding
the difference of the two superbias images to the other dark's pedestal
(`reference_superbias`, `superbias_difference`, the same helper the step
uses), and records the reference in the provenance and history.
Result (`results/2026-10-22_reset_calib/nis_reset_calib.asdf`): sigma_b
24.97 DN, single-read noise 7.43 DN, pedestal median -0.36 DN, pedestal
row-mean / column-mean scatter 2.0 / 2.7 DN, superbias 0257. So the
increase from one exposure (22.0) to two is not a superbias artefact:
with the references aligned it is 25.0, a real exposure-to-exposure
scatter of the reset level, which the sky-anchored offset would absorb
on a field. First-read timing (k, b) still needs a NIRISS field;
t1_scale stays 1.0 in the file.

Dark cleaning metrics (`exp_darks_part_a.py`, jw09279002001, 20
groups, `results/2026-10-22_darks_nis/NIS/`; science frame, where the
fast readout axis runs along columns):

| variant | pixel | row-mean | column-mean | ap 9 / 33 / 101 |
|---|---|---|---|---|
| SIRS | 0.0300 | 0.0092 | 0.0083 | 0.0085 / 0.0057 / 0.0047 |
| SIRS + jwst median | 0.0303 | 0.0103 | 0.0065 | 0.0083 / 0.0058 / 0.0045 |
| SIRS + stream_clean box 32 | 0.0275 | 0.0043 | 0.0039 | 0.0066 / 0.0049 / 0.0044 |
| same, column_offsets off | 0.0278 | 0.0043 | 0.0039 | 0.0068 / 0.0050 / 0.0044 |

Rate panels (`make_rate_image_panels.py --background none`,
`results/2026-10-22_rate_images_nis/NIS/`), full frame, DN/s:

| chain | row-mean | column-mean | pixel | finite |
|---|---|---|---|---|
| median refpix, OLS_C | 0.0033 | 0.0103 | 0.0305 | 4152642 |
| SIRS, OLS_C | 0.0032 | 0.0083 | 0.0297 | 4152642 |
| SIRS + jwst median, no background, OLS_C | 0.0029 | 0.0043 | 0.0291 | 4152642 |
| SIRS + stream_clean, no background, OLS_C | 0.0004 | 0.0008 | 0.0267 | 4152642 |
| SIRS + stream_clean, reset_rampfit | 0.0005 | 0.0009 | 0.0266 | 4154975 |

The step works unchanged on the NIRISS geometry: the banding (along
science columns here) drops from 0.0083 to 0.0008 DN/s, ten times, and
the pixel scatter by 10 percent; the jwst median gets a factor 2. The
reset fit matches OLS_C in structure and fits 2333 more pixels. All
dark-based measurements are now saved under results/, so the dark
files can leave the disk; what still needs data is a NIRISS field.

### 2026-10-22: dark metrics regenerated with the current code

James asked whether `aperture_scatter.png` was stale: it read the v5
run (2026-10-21, before the even-odd and column-offset corrections, the
masked-mean model and the first-read cleaning). Rerun with the current
code, NRCA1 dark 8758, 20 groups
(`results/2026-10-22_darks_part_a/NRCA1/`), plus a new `sirs_wiener_none`
variant (no background, the right setting on a dark):

| variant | pixel | row-mean | column-mean | ap 9 / 33 / 101 |
|---|---|---|---|---|
| SIRS | 0.0205 | 0.0053 | 0.0020 | 0.0052 / 0.0037 / 0.0027 |
| SIRS + jwst median (model bkg) | 0.0213 | 0.0079 | 0.0021 | 0.0058 / 0.0046 / 0.0034 |
| SIRS + Wiener, box 32 (default) | 0.0195 | 0.0025 | 0.0011 | 0.0039 / 0.0028 / 0.0026 |
| SIRS + Wiener, box 256 | 0.0194 | 0.0013 | 0.0007 | 0.0030 / 0.0016 / 0.0014 |
| SIRS + Wiener, scalar median | 0.0194 | 0.0013 | 0.0007 | 0.0028 / 0.0010 / 0.0004 |
| SIRS + Wiener, no background | 0.0193 | 0.0005 | 0.0002 | 0.0027 / 0.0009 / 0.0003 |

Against v5 the default (box 32) is unchanged within 0.0002 at every
scale except 101 pixels (0.0024 -> 0.0026, the masked-mean model's
small cost noted on 2026-10-21); the column means are now measured and
halved by the column-offset correction; the no-background variant sits
at the white-noise floor at every scale (white prediction at 101
pixels 0.0002). The figures (`aperture_scatter.png`,
`channel_spectra.png`) now read this run, with the no-background curve
added.

### 2026-10-22: CEERS F444W MEDIUM8, NRCALONG exposure 00001 (program 1345, 2022-06-22, 9 groups)

`scripts/make_field_panels.py` now takes a list of background settings
(`--background median,box32,box256,none`) and `--apply-flat`;
`scripts/compare_backgrounds.py --flat` measures every chain on
flat-fielded rates (the CRDS flat of the exposure divided out) with a
large-scale change metric (scatter of 128-pixel block means of the
change against SIRS over the sky pixels, as a fraction of the sky).

Three things learnt, two of them about our own code.

1. Measure and compare after flat-fielding (James's point). The LW
rate image before the flat carries the detector's response, a 10
percent large-scale pattern that dominates any row or column statistic
and that a scalar background treats as slow structure. Pre-flat the
numbers were meaningless: the jwst step (which applies the flat before
fitting) seemed to do nothing (row-mean 0.0098 against 0.0097), and the
stream clean with a scalar median seemed to win (0.0007) while in
fact removing the response pattern (large-scale change 4.7 percent of
the sky; the full-frame figure showed it plainly). On flat-fielded
rates, sky 0.679 DN/s, full frame, DN/s:

| chain | row-mean | col-mean | sky change | large-scale change | p5 / p95 |
|---|---|---|---|---|---|
| SIRS, OLS_C | 0.0014 | 0.0016 | 0 | 0 | 0 |
| jwst median, flat applied | 0.0006 | 0.0016 | +0.03% | 0.11% | -0.002 / +0.002 |
| stream_clean, median, no flat | 0.0099 | 0.0221 | +0.40% | 4.74% | -0.039 / +0.048 |
| stream_clean, box32, no flat | 0.0017 | 0.0032 | +0.01% | 0.10% | -0.008 / +0.008 |
| stream_clean, box256, no flat | 0.0063 | 0.0052 | -0.13% | 1.35% | -0.020 / +0.019 |
| stream_clean, none, no flat | 0.0011 | 0.0016 | -100% | | sky removed |
| stream_clean, flat applied, median | 0.0005 | 0.0004 | -0.07% | 0.22% | -0.005 / +0.003 |
| stream_clean, flat applied, box32 | 0.0011 | 0.0015 | -0.03% | 0.04% | -0.003 / +0.003 |
| stream_clean, flat applied, box256 | 0.0008 | 0.0013 | -0.04% | 0.13% | -0.004 / +0.003 |
| stream_clean, flat applied, box512 | 0.0006 | 0.0007 | -0.06% | 0.20% | -0.005 / +0.003 |

With the flat applied the scalar median is the best setting again:
3 times below SIRS in row means and 4 times in column means, 3 times
below the jwst step in column means (which the row median cannot
touch), large-scale structure kept to 0.2 percent and the per-pixel
change within +-0.005 DN/s. The background that a sky field needs is
the sky times the detector response, so `apply_flat_field` must be on
for sky fields; the step had it as an option, off by default, and the
Omega Cen runs got away without it because stars dominated the sky.
`background_method=None` removes the sky entirely (the filter's unit
gain at zero frequency takes the channel mean): darks only. On this
exposure the 1/f left by SIRS is small to begin with (9 groups of 107
s average it down), so the gain over SIRS is a factor 3 to 4 rather
than the factor 10 of the darks.

2. The superbias correction of 2026-10-22 was wrong and is withdrawn.
The 2022 exposure had superbias 0220 subtracted, the calibration darks
0443, and the two references differ by 11700 DN (median; 745 DN pixel
scatter): the superbias tracks the reset level of its epoch. Adding the
difference to the pedestal map put the reset prior 11400 DN above the
data. The fit survived because the likelihood jump search rejected the
reset point for 99 percent of pixels (used fraction 0.0087), but the
432 pixels saturated from group 2, which have no other anchor, got
rates off by hundreds of DN/s. The residual against each epoch's own
superbias is the quantity that transfers; the step now uses the map
as it is and warns when the epochs differ (the per-pixel part of the
residual is untested across epochs). The same alignment was removed
from the calibration tool; for NIRISS, whose two darks had different
references (0252 and 0257, 18 DN apart), the unaligned two-dark file
of 2026-10-20 (sigma_b 24.3) is adopted as the calibration, since the
darks have left the disk. The step's messages now reach the stpipe
log (`get_stpipe_loggers` adds "snowblind"), so the warning is visible.

3. The first-read shortfall is a delay in seconds, not a fraction.
On MEDIUM8 (first-group time 48.3 s) the two-parameter fit gives
k = 1.009 (per channel 0.996 to 1.026) and b = -3.7 DN, against
k = 0.943 on BRIGHT1 (10.7 s). A fixed delay of (1 - 0.943) x 10.7 =
0.61 s predicts k = 0.987 at 48.3 s, within the MEDIUM8 errors.
The calibration now stores `t1_seconds`, the first-group time of the
timing exposure (10.73677 s for the 6154 BRIGHT1 files, backfilled
into all eleven files), and the step converts the scale to the
exposure's own first-group time (`rescale_t1`). The reset offset b is
the same few DN on both patterns.

Panels rerun with the flat applied in stream_clean
(`results/2026-10-22_ceers_panels_v3`, `--background median,box32,box256
--apply-flat`), measured on flat-fielded rates, sky 0.678 DN/s:

| chain | full row | full col | cutout row | cutout col | sky change | large-scale | p5 / p95 | finite |
|---|---|---|---|---|---|---|---|---|
| SIRS, OLS_C | 0.0013 | 0.0015 | 0.0014 | 0.0016 | 0 | 0 | 0 | 4146621 |
| jwst median, flat applied, OLS_C | 0.0006 | 0.0015 | 0.0013 | 0.0016 | +0.03% | 0.11% | -0.002 / +0.002 | 4146621 |
| stream_clean, flat, median, OLS_C | 0.0003 | 0.0003 | 0.0006 | 0.0010 | -0.06% | 0.22% | -0.005 / +0.003 | 4146621 |
| stream_clean, flat, box32, OLS_C | 0.0010 | 0.0014 | 0.0009 | 0.0011 | -0.03% | 0.04% | -0.003 / +0.003 | 4146621 |
| stream_clean, flat, box256, OLS_C | 0.0007 | 0.0011 | 0.0007 | 0.0011 | -0.04% | 0.13% | -0.004 / +0.003 | 4146621 |
| stream_clean, flat, median, reset_rampfit | 0.0011 | 0.0008 | 0.0014 | 0.0013 | +0.01% | 0.23% | -0.008 / +0.008 | 4147610 |

The scalar median with the flat applied is the configuration for this
field: 4 to 5 times below SIRS and 2 (rows) to 5 (columns) times below
the jwst step, with the large-scale structure kept to 0.2 percent. The
reset check with the corrected prior and the delay model
(`results/2026-10-22_ceers_reset_field_v2`): t1_scale carried to
0.9874, used-reset fraction 0.846 (from 0.0087), all 432 pixels
saturated from group 2 fitted, median rate 0.68993 against 0.69007
(LIKELY) and 0.6893 (OLS_C); the warning about the superbias epochs
and the timing-carry line now appear in the step log. Open point: on
this exposure the reset fit has more row and column structure than
OLS_C on the same cleaned ramp (0.0011 / 0.0008 against 0.0003 /
0.0003), which did not happen on Omega Cen; the suspect is the
pedestal map prior across superbias epochs, whose per-pixel residual
may carry row-coherent differences (the two superbias references
differ with 12.9 DN of row-mean scatter). Map prior against scalar
prior under test.

Order of operations with the flat (James): the background is measured
on the flat-fielded difference, where the sky is smooth and the
detector response is not, but what is subtracted from the raw
difference is that background times the flat, and the noise fit runs
in raw DN, where the 1/f noise is additive and does not pass through
the quantum efficiency. Until now the step did what the jwst step
does, divide the whole difference by the flat, fit and filter there,
and multiply back, which modulates the noise estimate by the flat (a
second-order error, invisible at the CEERS tails of +-0.005 DN/s, but
wrong in principle). `prepare_difference` now returns a raw-DN
residual and a raw-DN background (`background * flat`); the
integration model, fitted to the flat-fielded slope, is multiplied by
the flat at the same point; `clean_difference` no longer multiplies
the result by the flat. Pixels where the flat is unusable are treated
as missing. Two tests cover it. The CEERS panels are rerun with it.

Reset-fit row structure on CEERS, prior test (scratch
`ceers/prior_variants.py`, same cleaned ramp, flat-fielded sky pixels):
OLS_C 0.0002 / 0.0002; reset fit with the map prior 0.0012 / 0.0009,
with a scalar prior 0.0012 / 0.0009, with the map not high-passed
0.0012 / 0.0009. So the prior is not the source; the structure enters
through the first difference (group 0 minus prior) itself. On this
ramp the reset point has weight: sigma_b 17.6 DN against a per-group
scatter of about 9 DN (read noise over 8 frames plus the sky's Poisson
noise over 107 s), so about a quarter of a group at the largest lever
arm, and a row-coherent error of 2 DN in that difference is enough for
0.001 DN/s. Candidates: residual 1/f of the reset frame itself after
the first-read cleaning, or the 15 percent of pixels whose reset point
the jump search rejects, in row-correlated patches. At 0.2 percent of
the sky on sky pixels it is second order for this field, where the
reset point only matters for the 432 saturated cores; open item,
re-examine on a short ramp with a bright sky where the reset point
weighs more.

Panels rerun with the raw-DN flat formulation
(`results/2026-10-22_ceers_panels_v4`): identical to v3 within 0.0001
DN/s in every entry (median: rows 0.0003, columns 0.0002, large-scale
0.22 percent, tails -0.0047 / +0.0033), as expected for a second-order
change. The figures `field_rate_images_ceers_nrcalong{,_full}` read
v4, displayed and measured flat-fielded.

### 2026-10-22: the ZEROFRAME was not cleaned

James suggested the noisy reset frame against the 8-frame-averaged
first group as the source of the reset fit's row structure on CEERS.
The frame in question is the ZEROFRAME: for frame-averaged readouts
the pipeline carries the first frame of the first group separately,
and the likelihood fit (`use_zeroframe`, default on) peels it out of
the first group and uses it as its own resultant. The stream clean
only touched the group cube, so that frame kept the full 1/f banding
of a single read and the reset fit put it into the rate. BRIGHT1 has
no separate zeroframe (the first group is the first frame), which is
why Omega Cen never showed it. Test on the same cleaned CEERS ramp,
flat-fielded sky pixels, row / column scatter: OLS_C 0.0003 / 0.0002;
reset fit with the zeroframe 0.0012 / 0.0009, without it 0.0003 /
0.0002; with sigma_b inflated to 50 DN the same two numbers, so the
reset weight is irrelevant. Fix: stream_clean cleans the zeroframe as
it cleans the first group, as a difference against the reset prior,
with the background scaled to one frame time, keeping the pipeline's
zero marker for unusable zeroframe pixels.

Verified (`results/2026-10-22_ceers_panels_v5`, median background, flat
applied, zeroframe cleaned): reset fit row / column scatter 0.0003 /
0.0002, equal to OLS_C, with 4147611 finite pixels against 4146621;
sky change +0.01 percent, large-scale 0.23 percent, tails -0.0074 /
+0.0074 (wider than OLS_C's -0.0047 / +0.0033 because the reset point
adds the kTC-limited first difference to every pixel's fit, a per-pixel
white term, not structure). The CEERS figures read v5.

### 2026-10-22: all six CEERS F444W MEDIUM8 exposures

`results/2026-10-22_ceers_all/<exposure>/<DET>/`, scalar median, flat
applied, zeroframe cleaned, reset fit with the map prior; measured on
flat-fielded sky pixels, full frame, DN/s (row-mean / column-mean
scatter; sky change and large-scale change in percent of the sky):

| exposure | sky | SIRS | jwst | stream_clean, OLS_C | stream_clean, reset fit | reset: sky, LS, finite - OLS_C |
|---|---|---|---|---|---|---|
| NRCALONG 00001 | 0.678 | 0.0013 / 0.0015 | 0.0006 / 0.0015 | 0.0003 / 0.0002 | 0.0003 / 0.0002 | +0.01, 0.23, +990 |
| NRCALONG 00002 | 0.676 | 0.0012 / 0.0017 | 0.0006 / 0.0017 | 0.0003 / 0.0003 | 0.0003 / 0.0002 | +0.04, 0.29, +902 |
| NRCALONG 00003 | 0.679 | 0.0013 / 0.0016 | 0.0006 / 0.0017 | 0.0003 / 0.0002 | 0.0003 / 0.0002 | +0.02, 0.22, +1255 |
| NRCBLONG 00001 | 0.706 | 0.0021 / 0.0011 | 0.0007 / 0.0011 | 0.0003 / 0.0002 | 0.0003 / 0.0002 | -0.04, 0.29, +2133 |
| NRCBLONG 00002 | 0.707 | 0.0022 / 0.0015 | 0.0007 / 0.0015 | 0.0003 / 0.0002 | 0.0003 / 0.0002 | -0.24, 0.31, +1503 |
| NRCBLONG 00003 | 0.708 | 0.0018 / 0.0011 | 0.0006 / 0.0011 | 0.0003 / 0.0002 | 0.0003 / 0.0002 | -0.13, 0.26, +1462 |

The six are the same to the last digit: stream_clean 0.0003 / 0.0002
on every exposure of both detectors, against 0.0012 to 0.0022 /
0.0011 to 0.0017 for SIRS and 0.0006 to 0.0007 / unchanged for the
jwst median; the sky change is -0.06 percent on every one and the
large-scale change 0.21 to 0.30 percent; the per-pixel tails -0.005 /
+0.004. The reset fit matches OLS_C in structure on all six and fits
900 to 2100 more pixels; its sky change varies between -0.24 and +0.04
percent (the OLS_C clean is -0.06 on all), which is the reset-level
offset b of the B-module calibration (positive on NRCBLONG, +0.7 to
+2.4 DN) entering through the first difference; small, but worth
re-deriving b on this field rather than Omega Cen if the B module is
used in anger. NRCBLONG has more row banding after SIRS than NRCALONG
(0.0021 against 0.0013) and the same column structure. Figures:
`field_rate_images_ceers_{nrcalong,nrcblong}{,_full}` from exposure
00001. Remaining for CEERS: the photometry comparison against James's
iterated remstriping reduction, which needs his products.

### 2026-10-23: snowball expansion in the jump step for the dark figures

James saw snowball rings in the NIRISS dark figure and asked whether
`expand_large_events` was on. It was: the NIRISS jump parameter
reference (0075 in the run, 0087 in context 1598, same values for
darks and images) sets expand_large_events True, min_jump_area 15,
expand_factor 1.75, sat_required_snowball True; NIRCam's sets area 5,
factor 2. The NIRISS Detector1 parameter references (0004 darks, 0005
images) override nothing. The rings are the halo beyond the expanded
disc: in the annulus 2 to 25 pixels past the flagged edge of the three
largest events the rate sits 0.01 to 0.03 DN/s above the frame.
James's choice: area 8, factor at least 2; the SnowblindStep route is
not wanted for now (the helper's `snowblind` option now also switches
the jump expansion off, since Snowblind grows the raw regions and
would otherwise grow the circles again; `jump_params` and the scripts'
`--jump key=value` override the reference). Test (scratch
`snowballs/jump_test.py`, 20 groups, OLS_C; halo = median excess rate
in the annulus 2 to 30 px beyond each flagged region of 100 px or
more; "jump px" = JUMP_DET pixels in the rate DQ, which OLS_C still
fits from the other segments, so no pixel is lost):

| dark | setting | events >= 100 px | jump px | halo median (DN/s) | halo 90th pct |
|---|---|---|---|---|---|
| NIRISS | CRDS (area 15, factor 1.75, sat) | 15 | 50496 | +0.0059 | +0.0177 |
| NIRISS | area 8, factor 2 | 17 | 53804 | +0.0055 | +0.0131 |
| NIRISS | area 8, factor 2.5 | 21 | 61757 | +0.0035 | +0.0095 |
| NIRISS | area 8, factor 2, no sat required | 18 | 62199 | +0.0051 | +0.0123 |
| NIRISS | area 8, factor 3, no sat required | 39 | 106526 | +0.0015 | +0.0069 |
| NRCA1 | CRDS (area 5, factor 2, sat) | 10 | 71231 | +0.0008 | +0.0077 |
| NRCA1 | area 8, factor 2 | 10 | 70852 | +0.0008 | +0.0077 |
| NRCA1 | area 8, factor 2.5 | 11 | 73872 | +0.0006 | +0.0062 |
| NRCA1 | area 8, factor 2, no sat required | 16 | 90874 | +0.0008 | +0.0048 |
| NRCA1 | area 8, factor 3, no sat required | 47 | 155521 | +0.0011 | +0.0037 |

The halo falls off slowly, so the expansion factor is the lever and
the area threshold barely matters at 20 groups: on NIRISS factor 2
takes the median halo from 0.0059 to 0.0055, factor 2.5 to 0.0035
(90th percentile 0.0177 -> 0.0095) for 15 percent more flagged
pixels, and only factor 3 without the saturation requirement reaches
0.0015, at twice the flagged pixels (2.6 percent of the detector).
NRCA1 halos are 5 times smaller to begin with. Chosen for the figures:
area 8, factor 2.5, saturation required, on both darks
(`--jump min_jump_area=8,expand_factor=2.5`;
`results/2026-10-23_rate_images_{nis,nrca1}`). The remaining halo at
factor 2.5 is the tail of the charge profile; a profile-following
growth is what SnowblindStep does, kept for later.

James asked for more expansion on NIRISS, factor 3. Measured with the
saturation requirement kept (the earlier factor-3 row was without it):

| dark | setting | events >= 100 px | jump px | halo median (DN/s) | halo 90th pct |
|---|---|---|---|---|---|
| NIRISS | area 8, factor 3 | 21 | 71446 | +0.0030 | +0.0080 |
| NIRISS | area 8, factor 4 | 27 | 96081 | +0.0031 | +0.0076 |

Factor 3 takes the median halo from 0.0035 (factor 2.5) to 0.0030 and
the 90th percentile from 0.0095 to 0.0080 for 16 percent more flagged
pixels; factor 4 gains nothing further at 35 percent more, so the
halo beyond 3 times the core is at the measurement floor of this
annulus statistic. NIRISS figures now use area 8, factor 3
(`results/2026-10-23_rate_images_nis_f3`); NRCA1 stays at 2.5.

### 2026-10-23: the CEERS remstriping baseline

James's reduction recipe (`run_detector1.py`, `run_image2.py`,
`remstriping_search.py` in 5105_9263_nexus/reduction1): Detector1 with
the jump step's expansion off, rejection threshold 5, after-jump
flagging off, then SnowblindStep (min_radius 3, after_jumps 5),
JumpPlusStep, OLS_C; then `measure_striping` on the rate (flat applied
for the measurement, tiered source mask at Gaussian scales 25/15/5/2
px, Gaussian-fitted sky pedestal removed and restored, 2-sigma-clipped
median per amplifier row with a searched threshold for falling back to
the full-row median, then column medians), subtracted from the raw
rate; Image2 with photom, so the _cal files are MJy/sr.

Two checks on the script itself, on the CEERS NRCALONG exposure:

- Units. The pattern is measured on the flat-fielded image and
  subtracted from the raw one; the residual is the pattern times (1 -
  1/flat), which averages along a row to 1 to 3 percent of the row
  pattern (0.00003 to 0.00004 DN/s, pixel 0.0001): a formal
  inconsistency, not a contributor.
- Precision. The per-amplifier row median is estimated from the sky
  pixels of one rate image: 318 unmasked pixels per amp-row under the
  script's 27.7 percent mask, pixel scatter 0.021 DN/s flat-fielded,
  so a clipped median has an error of about 0.0015 DN/s per amp-row,
  while the striping it is measuring is 0.0015 to 0.0017 per amp-row
  (what the stream clean removes on the sky pixels). The pattern it
  subtracts has a row-to-row scatter of 0.0022 to 0.0025 per amp: half
  signal, half estimation noise. So on a sky-dominated LW exposure the
  method removes the striping and puts back as much noise, row by row.

Result (`measure_striping` run by us on our SIRS + OLS_C rate,
`sirs_remstriping_ols.npz` beside the panels; flat-fielded sky pixels):

| exposure | chain | full row | full col | cutout row | cutout col | sky | LS | p5 / p95 |
|---|---|---|---|---|---|---|---|---|
| NRCALONG 00001 | SIRS | 0.0013 | 0.0015 | 0.0014 | 0.0016 | 0 | 0 | 0 |
| | jwst median, flat | 0.0006 | 0.0015 | 0.0013 | 0.0016 | +0.03% | 0.11% | -0.0022 / +0.0022 |
| | remstriping | 0.0009 | 0.0008 | 0.0014 | 0.0013 | -0.00% | 0.18% | -0.0047 / +0.0044 |
| | stream_clean | 0.0003 | 0.0002 | 0.0006 | 0.0010 | -0.06% | 0.22% | -0.0047 / +0.0033 |
| NRCBLONG 00001 | SIRS | 0.0021 | 0.0011 | 0.0017 | 0.0013 | 0 | 0 | 0 |
| | jwst median, flat | 0.0007 | 0.0011 | 0.0013 | 0.0013 | +0.03% | 0.22% | -0.0029 / +0.0028 |
| | remstriping | 0.0010 | 0.0008 | 0.0017 | 0.0014 | +0.00% | 0.25% | -0.0055 / +0.0052 |
| | stream_clean | 0.0003 | 0.0002 | 0.0007 | 0.0010 | -0.06% | 0.28% | -0.0051 / +0.0038 |

Remstriping is the only method of the three that touches the column
structure (0.0015 -> 0.0008), and over the full frame it halves the
row structure, but in the 512-pixel cutout it changes nothing, and
its per-pixel change tails are twice the jwst step's: the estimation
noise above. The stream clean, which measures the noise spectrum over
~125 rows rather than one row at a time and works on the differences,
is 3 times below remstriping in rows and 4 times in columns. The
CEERS figures now show remstriping as the fourth panel in place of the
reset fit, whose numbers equal the OLS_C panel's there. Photometry
against James's own _cal products still to come, with SHALLOW4.

### 2026-10-23: NEXUS (programs 5105 and 9263): short ramps, a bright star, and James's remstriping products

Three pairs from James's reduction (jwst 2.0.1, context 1535; Detector1
with the jump expansion off, SnowblindStep, JumpPlusStep, OLS_C;
remstriping on the rate; Image2 with photom, so _cal in MJy/sr):
NRCB2 F200W MEDIUM8 6 groups with a bright star (jw05105014014_05101_00002),
NRCA1 F090W SHALLOW4 6 groups (jw05105014018_08101_00001), NRCBLONG F277W
MEDIUM8 7 groups (jw09263005003_03201_00001). Our chain under the
current context (median background, flat applied, zeroframe cleaned),
our own remstriping run on our SIRS rate, and James's _cal converted
back to raw DN/s (divided by PHOTMJSR, times the flat); all measured on
flat-fielded sky pixels (`results/2026-10-23_nexus/`):

| exposure (sky DN/s) | chain | full row | full col | cutout row | cutout col | sky | LS | p5 / p95 |
|---|---|---|---|---|---|---|---|---|
| NRCB2 (0.066) | SIRS | 0.0014 | 0.0010 | 0.0016 | 0.0019 | 0 | 0 | |
| | jwst median | 0.0006 | 0.0010 | 0.0011 | 0.0019 | +0.10% | 0.49% | -0.0018 / +0.0018 |
| | remstriping (ours / James) | 0.0004 / 0.0004 | 0.0004 / 0.0004 | 0.0009 | 0.0006 | -0.21 / +0.01% | 1.72% | -0.0039 / +0.0031 |
| | stream_clean | 0.0001 | 0.0001 | 0.0003 | 0.0005 | -0.13% | 1.75% | -0.0043 / +0.0033 |
| | stream_clean, reset fit | 0.0001 | 0.0001 | 0.0003 | 0.0005 | -0.50% | 1.75% | -0.0047 / +0.0034 |
| NRCA1 SHALLOW4 (0.045) | SIRS | 0.0031 | 0.0014 | 0.0034 | 0.0011 | 0 | 0 | |
| | jwst median | 0.0012 | 0.0014 | 0.0019 | 0.0011 | -0.04% | 2.76% | -0.0040 / +0.0040 |
| | remstriping (ours / James) | 0.0006 / 0.0006 | 0.0006 / 0.0006 | 0.0012 | 0.0009 | -0.10 / -0.18% | 4.66 / 4.45% | -0.0064 / +0.0063 |
| | stream_clean | 0.0003 | 0.0002 | 0.0005 | 0.0007 | -0.19% | 5.09% | -0.0066 / +0.0064 |
| | stream_clean, reset fit | 0.0003 | 0.0002 | 0.0005 | 0.0007 | +1.85% | 5.07% | -0.0064 / +0.0081 |
| | stream_clean, box 32 | 0.0016 | 0.0007 | 0.0017 | 0.0008 | -0.24% | 0.67% | -0.0055 / +0.0054 |
| | stream_clean, box 256 | 0.0010 | 0.0005 | 0.0005 | 0.0007 | -0.04% | 1.91% | -0.0060 / +0.0060 |
| NRCBLONG (0.184) | SIRS | 0.0017 | 0.0010 | 0.0015 | 0.0012 | 0 | 0 | |
| | jwst median | 0.0006 | 0.0010 | 0.0011 | 0.0012 | +0.05% | 0.39% | -0.0021 / +0.0023 |
| | remstriping (ours / James) | 0.0009 / 0.0009 | 0.0008 / 0.0008 | 0.0015 | 0.0010 | -0.04 / +0.08% | 0.60% | -0.0046 / +0.0043 |
| | stream_clean | 0.0002 | 0.0002 | 0.0004 | 0.0008 | -0.12% | 0.71% | -0.0050 / +0.0038 |
| | stream_clean, reset fit | 0.0002 | 0.0002 | 0.0004 | 0.0009 | +0.11% | 0.71% | -0.0063 / +0.0063 |

Our remstriping run reproduces James's _cal to the last digit in the
structure statistics (the sky offsets differ by the chains'
different superbias files and contexts), so the baseline is faithful.
The stream clean is 3 to 4 times below remstriping and 4 to 6 below
the jwst median on every exposure; on the SHALLOW4 frame the three
methods are at 0.0003, 0.0006 and 0.0012 in rows. On SHALLOW4 the
large-scale change (128-pixel blocks) is 4.5 to 5.1 percent of the
0.045 DN/s sky for remstriping and the stream clean alike, 2.8 for the
jwst median, and 0.7 with the box-32 model: slow structure that the
scalar backgrounds remove and the box model keeps, which from one
exposure cannot be told from sky; remstriping, which James trusts,
removes the same amount.

Photometry (`scripts/compare_field_photometry.py`: DAOStarFinder on the
SIRS rate, aperture radius 4 with an 8 to 12 annulus, ratios to SIRS by
flux quartile): every chain within 0.3 percent in every quartile on
all three exposures, except the stream clean's faintest quartile on
NRCA1 (-0.7 +- 0.4 percent, 4 to 24 DN/s) and NRCB2 (-0.3 percent),
the faint-source bias of the injection test; James's _cal shows
larger scatter from its different flagging (40000 fewer finite
pixels), not a bias.

The bright star (NRCB2, peak 25 DN/s = 378 x sky). Unmasked halo
excess over the sky and the change each chain makes there, as
fractions of the sky, azimuthal means over the chisel-mask pixels:

| annulus | halo excess | jwst | remstriping | stream_clean |
|---|---|---|---|---|
| 20-100 px | +0.197 | -0.009 | -0.049 | -0.147 |
| 100-200 px | +0.082 | -0.012 | -0.059 | -0.077 |
| 200-400 px | +0.028 | -0.004 | -0.024 | -0.028 |
| 400-800 px | 0.000 | 0.000 | -0.002 | 0.000 |

The stream clean removes 75 percent of the unmasked halo within 100 px
and all of it beyond; remstriping 25, 70 and 90 percent; the jwst
median little. This is the "dunking" James asked about, and it is the
linear estimator reading the smooth halo as slow 1/f: the chisel mask
with its 32-pixel tiled sky absorbs the halo into the sky and masks
only 25 percent of the pixels within 100 px (a 256-pixel sky box masks
95 percent there but still only 25 percent at 100-200 px, where the
excess is 8 percent of the sky and below the per-pixel threshold).
The fix is a large-kernel detection tier in the scene mask, as
remstriping has (Gaussian 25 then 15, 5, 2 px); under test. The
box-32 model protects the halo but hands back the slow 1/f.

Reset fit on SHALLOW4: the sky rises by +1.85 percent with the reset
fit against -0.19 with OLS_C on the same cleaned ramp, and 0.5 percent
on NRCB2. `exp_reset_field.py` on the exposure itself: the
sky-anchored reset offset b is +11.0 DN (per channel +6.3, +10.6,
+13.5, +13.4) against the calibration's -2.9 to -5.6 DN from Omega Cen
(March 2025) with the same superbias (0454): the reset level drifts by
10 to 15 DN over three months, and on a 0.045 DN/s sky with 27 s to
the first group that is a few percent of the first-read signal. So b
cannot be a calibration constant; the step should anchor it to each
exposure's own sky pixels (median over faint, clean pixels of the first
resultant minus the prior map minus the slope from the later groups
times its time), per channel. Implemented next. The first-read scale
k is also inconsistent across patterns (0.904 at 10.7 s BRIGHT1, 0.923
at 26.8 s SHALLOW4, 1.009 at 48.3 s MEDIUM8: neither a fixed fraction
nor a fixed delay), and the SHALLOW4 fit gave identical k for three
channels, a sign of too few bright bins on a faint field; k stays an
open calibration question, with the delay model kept as the better of
the two simple ones.

### 2026-10-23: the reset offset anchored to each exposure's sky

Implemented in ResetRampFitStep (`anchor_offset_to_sky`, default True;
`anchor_max_signal` 50 DN, `anchor_min_pixels` 1000 with the file's
offset as the fallback): per channel, the median over faint unflagged
pixels of the first resultant (the zeroframe when used, else group 0)
minus the prior map minus the later-group slope times
(t_first - (1 - k) t1). On NEXUS the anchored offsets against the
calibration's: NRCA1 +11.9, +12.6, +12.6, +8.9 DN (file -2.9 to -5.6),
NRCB2 +0.2 to +2.5 (file +1.7 to +4.0), NRCBLONG +4.3 to +6.0 (file
+0.7 to +2.4). After anchoring the first-difference residual on the
SHALLOW4 exposure is centred (median -0.001 DN per channel, 16/84 at
+-1.18 DN, chi-square 1.40, reset point used for 96.3 percent of
pixels), and the reset fit agrees with stcal's LIKELY fit without the
reset point to 0.1 percent at every brightness:

| pixels (NRCA1 SHALLOW4, uncleaned ramp) | n | LIKELY / OLS_C | reset / OLS_C | reset / LIKELY |
|---|---|---|---|---|
| sky (within 5 sigma) | 4117849 | 0.9946 | 0.9932 | 0.9988 |
| faint sources (5 to 50 sigma) | 24430 | 0.9982 | 0.9975 | 0.9992 |
| brighter (50 to 500 sigma) | 2788 | 0.9991 | 0.9990 | 0.9999 |
| bright (500 to 5000 sigma) | 350 | 0.9987 | 0.9983 | 0.9996 |

So the residual -0.6 to -0.8 percent of the reset fit against OLS_C on
the sky of this exposure is not the reset point: it is stcal's
likelihood fit itself, which sits 0.5 percent below OLS_C on faint sky
pixels of a 6-group ramp (0.1 percent on the 0.68 DN/s CEERS sky, 0.1
to 0.2 on Omega Cen). Noted for the stcal observations: on a faint sky
the Poisson-weighted likelihood fit and OLS_C differ by half a percent,
and which is right needs a known-rate test (a flat-field lamp or an
injection). The per-exposure first-read scale from bright pixels
(`star_anchored_scales`, `anchor_timing_to_stars`, default off) is
implemented and tested on a synthetic ramp but not needed by this
evidence; measured on NEXUS below.

Star-anchored first-read scale on NEXUS (`anchor_timing_to_stars`):
on NRCB2 every channel fell back to the carried calibration value
(too few bright unsaturated pixels for three bins of 20), on NRCA1
three of four did and the fourth fitted 0.951 against 0.966; the
rates on brighter pixels are 0.999 of OLS_C either way. The option
stays off by default: on typical fields there is no lever arm for k,
and the carried value with the sky-anchored offset already puts the
reset fit within 0.1 percent of the likelihood fit.

Bright-star halo masking, second try (scratch `nexus/halo_tier2.py`):
a coarse chisel pass (Gaussian FWHM 12 px, erode 4, tiled sky of 256
px, growth capped at 4) with the detection quantile raised from 0.5 to
0.9 masks 9.9 percent of the NRCB2 frame against 7.1 for the fine mask
alone, and within 100 px of the star 99 percent of pixels (fine: 75),
at 100-200 px 33 percent (fine: 19), at 200-400 px 13 (fine: 9).
Growing every masked region of 500 px or more by its own radius masks
51 percent of the frame: unusable. The coarse tier at quantile 0.9 is
implemented as a second tier of the chisel mask; its cost on a field
without bright stars (CEERS) and its effect on the halo removal are
measured next.

Two-tier scene mask (coarse tier: FWHM 12, quantile 0.9, erode 4, sky
box 256, growth 4; `chisel_mask(..., coarse_fwhm=12)`, on by default):
`results/2026-10-23_coarse_mask/`. Mask fractions: NRCB2 9.9 percent
(fine alone 7.1), CEERS NRCALONG 12.0 (9.1), NRCA1 7.6. Sky statistics
of the stream clean unchanged on all three to the last digit (CEERS
0.0003 / 0.0003, LS 0.20 percent, tails -0.0044 / +0.0032 against
-0.0047 / +0.0033); faintest-quartile photometry improved on NRCB2
(0.9973 -> 0.9992) and NRCA1 (0.9926 -> 0.9940), unchanged at 1.000 on
CEERS. The bright star, change against SIRS as a fraction of the sky,
now over ALL pixels of each annulus (masked ones included, which is
what the science image sees):

| annulus | halo excess (unmasked) | stream, one tier | stream, two tiers | remstriping |
|---|---|---|---|---|
| 20-100 px | +0.197 / +0.130 | -0.144 | -0.084 | -0.046 |
| 100-200 px | +0.082 / +0.074 | -0.082 | -0.070 | -0.058 |
| 200-400 px | +0.028 / +0.025 | -0.030 | -0.029 | -0.026 |

The coarse tier halves the removal within 100 px (99 percent of those
pixels are now masked and the estimate there comes from outside) but
beyond 100 px it changes nothing: with a third of the annulus masked
the filter estimates the slow component from the rest, which still
carries the halo, and subtracts it under the masked pixels too. An 8
percent excess over the sky that is smooth over hundreds of pixels is
what slow 1/f looks like to any one-exposure method, and remstriping
removes it as well (0.058 of 0.082 at 100-200 px). Conclusion: the
tier stays on (helps the core and faint photometry, costs 3 percent
of pixels and no sky statistic); the extended halo of a bright star is
removed by the stream clean and by remstriping alike, and keeping it
needs either the box background model (which protects structure above
its box at the cost of the slow 1/f) or a scene model from more than
one exposure. The worst 30-degree sector-mean change near the NRCB2
star went from -0.21 to -0.13 times the sky.

### 2026-10-23: column-aligned diffraction spikes of saturated stars

James: the per-column offset correction darkens the columns through the
bright stars (NRCB2 F200W, one star; NRCBLONG F277W, two), because the
vertical diffraction spike runs on past the chisel mask at a few columns
wide and a fraction of the sky, and a per-column mean over 2040 rows
cannot clip it. Measured on NRCB2 (star at column 835): the spike is a
triple ridge about 15 px wide (peaks at -6, 0 and +6 columns) at 0.3 of
the sky 400 rows from the core and 0.1 at 700 rows, where the chisel
mask has stopped; the change against SIRS in the spike columns (835 +-
3, background pixels of the chisel mask) was 0.024 of the sky below the
neighbouring columns (15-40 px away) for the stream clean, against
0.008 above for remstriping and 0.003 above for the jwst median.

Three detectors were tried. A walk outward from the core along the
local column peak lost the spike. A bar filter at the six spike angles
(48 x 5 px, normalized masked convolution, detection above 2 percent
of the sky and 3 sigma, restricted to strips through the stars) masked
10-20 percent of the frame: the star's halo and any smooth background
excess are also "detected" by a bar, the connected detections bleed
into them, and the saturated-core selection (contiguous non-finite
pixels) picked bad-pixel clusters at the frame edges as stars, with
strips across the whole frame. Flanking bars 7 px out cancelled the
15-px-wide spike along with the halo. James: only the column-aligned
spike matters, and this is a star problem; stars and galaxies have
different profiles and mask shapes.

What is in `scene_mask.py` now: `saturated_stars` finds stars by their
mask shape, a masked region of at least 10000 px (the reference border
set aside so that regions at the edge are not joined to it) that
contains non-finite pixels and fills at most 0.4 of its bounding box
(a star mask with spike arms fills 0.15-0.3; galaxy masks 0.5-0.8),
centred on the largest cluster of saturated pixels. On the four frames
this gives NRCB2 one star, NRCBLONG two, CEERS none, NRCA1 SHALLOW4
two (the two bright stars at the left of the frame), exactly the
saturated stars. `spike_mask` then walks each star's column up and
down in 64-row blocks: the masked mean over the rows of 81 columns
about the current centre, smoothed over 3 columns, less the median of
the columns more than 22 from the centre (which removes the halo and
the local background), and columns within 12 of the centre above 2
percent of the sky and 3 sigma are masked for that block with one
column either side; their centroid is the next block's centre, blocks
inside the star's own mask are passed over, and the walk ends after
three blocks without a detection. `StreamCleanStep(mask_spikes=True)`
applies it after the chisel mask; it is on by default. Mask fractions:
NRCB2 9.90 -> 10.17 percent, NRCBLONG 13.99 -> 14.12, NRCA1 7.56 ->
7.59, CEERS unchanged. On NRCB2 the walk covers 800 rows above and
500 below the star, to where the block excess is 1 percent of the sky.

End to end (`results/2026-10-23_spike_mask/`, median background, flat
applied, the same chain as the coarse-mask runs; the remstriping
products copied over, as they do not depend on the mask), the change
against SIRS in the spike columns relative to the neighbouring
columns, in units of the sky, measured on the background pixels of the
new mask (spike wings excluded) for every chain:

| frame, column | stream, coarse mask | stream, spike mask | remstriping | jwst median |
|---|---|---|---|---|
| NRCB2, 835 | -0.016 | +0.016 | +0.011 | +0.005 |
| NRCBLONG, 1495 | -0.001 | +0.002 | -0.006 | +0.001 |
| NRCBLONG, 558 | -0.009 | -0.003 | -0.004 | +0.001 |

(On the chisel mask's background pixels, wings included, NRCB2 goes
from -0.024 to +0.016.) The darkening is gone and the stream clean
sits where remstriping sits; the small positive residual on NRCB2 is
the halo that the neighbouring columns lose and the masked columns do
not. NRCBLONG's F277W spikes outside the chisel mask are 3-4 percent
of the sky over 400 rows, which is why the darkening there was small
to begin with. Sky statistics unchanged to the third digit on both
frames (NRCB2 row/column scatter 0.00014 / 0.00010 DN/s, NRCBLONG
0.00023 / 0.00021); NRCB2 faintest-quartile photometry 0.9992 ->
1.0002 (remstriping 1.0019, jwst 1.0000), the other quartiles within
0.0003 of one; NRCBLONG photometry unchanged. The worst 30-degree
sector-mean change near the NRCB2 star is -0.13 of the sky as before:
that is the halo, which the spike mask does not address. Figures
regenerated from the new runs; unit tests cover the star selection
(a spike-shaped mask around NaNs is a star, a filled square around
NaNs is not), the walk beyond the arm of a star-shaped mask, and the
refusal of an unsaturated source.

### 2026-10-26: defaults, the packaged calibration, the pipeline recipe, and a NIRISS field

James asked where testing stood across data types, for an example of
running the two steps through the existing pipeline infrastructure
with their defaults, and for the defaults to be the most generally
usable. Tested so far: NIRCam darks (all ten detectors' reset
calibrations), NIRISS darks, and three NIRCam fields (Omega Cen, CEERS
F444W, NEXUS with a bright star). Missing: a NIRISS field, which the
NIRISS first-read timing also needed. Six NIRISS F200W exposures in
`~/data/jwst_niriss_test/` (all `NIS_IMAGE`, CLEAR + F200W: the direct
images of WFSS programs, nothing dispersed) cover NIS and NISRAPID, 7
to 20 groups, one with 6 integrations.

Defaults changed (`StreamCleanStep`): `background_method='median'`
(was `integration_model`, the box model the measurements had not
recommended since Omega Cen), `apply_flat_field=True` (an exposure
without a flat reference, such as a dark, is cleaned in raw DN with a
log line). `ResetRampFitStep` with neither `reset_calib` nor `sigma_b`
now uses a table packaged with snowblind,
`src/snowblind/data/reset_calib_defaults.json`, written by
`scripts/package_reset_defaults.py` from the calibration directory:
per detector the median sigma_b, t1_scale and reset offset (detector-wide
and per channel), t1_seconds and the provenance; the pedestal maps stay
in the ASDF files (`use_pedestal_mean` and `per_pixel_sigma_b` need the
file). With `save_results` the step writes `_rate` and `_rateints`
itself, since stpipe would give a two-model result indexed names.

Pipeline recipe (README and write-up "Running the steps"), verified end
to end on `jw03383060001_07201_00002_nis` (results/2026-10-26_niriss_strun_v2):

    strun calwebb_detector1 jw..._uncal.fits --steps.refpix.refpix_algorithm=sirs \
        --steps.jump.post_hooks="snowblind.StreamCleanStep", --steps.jump.save_results=True \
        --steps.clean_flicker_noise.skip=True
    strun reset_rampfit jw..._jump.fits --output_dir=reset

Two things the first attempt taught: a one-element `post_hooks` needs
the trailing comma (a bare string is "of the wrong type"), and the
NIRISS `pars-detector1pipeline` reference turns `clean_flicker_noise`
ON, so without the skip the pipeline cleaned our cleaned ramp a second
time with its median (the saved `_jump` is taken before that step, so
the reset products were unaffected). stpipe post-hook results replace
the step result, so the saved `_jump` is the cleaned ramp. A hook's own
log messages did not appear in the pipeline log: stpipe configures
handlers for the loggers the TOP-LEVEL class names, and jwst's pipeline
does not name ours. `snowblind.step.Step.run` now lends the `snowblind`
logger the handlers (and level) of the `stpipe` logger for the duration
of the run; standalone it is a no-op (the handlers are there already).
Two tests in test_step.py.

Spike mask on NIRISS: `saturated_stars` first took the thin network of
bad pixels that the chisel mask joins across the frame (a 26k px region
filling 1 percent of its box, with an 8761 px NaN cluster spanning
2040 x 2040) and two masked bad-column strips as stars. A star's core
must now be compact: the largest saturated cluster at least 16 px and
filling at least 0.3 of its own bounding box (`min_core`, `core_fill`).
Stars found: NRCB2 1, NRCBLONG 2, CEERS 0, NRCA1 2 (unchanged), NIRISS
jw03807 1 (the 1500 DN/s star), jw01571 0.

NIRISS first-read timing (`scripts/exp_reset_field.py`): jw04681
(NISRAPID, 20 groups) has too few bright pixels (3229 in 50-800 DN,
per-channel fits 0.45-0.86: noise). jw03807 (NISRAPID, 8 groups, 118
sources) gives the two-parameter fit k = 0.950, b = -2.3 DN (bright),
faint-pixel b = +5.4 DN, 4 bins, 4373 pixels; per channel 1.00, 0.95,
0.98, 0.98 from 2-3 bins each. Written into the NIRISS calibration
(`update_timing`, detector-wide k for every channel, t1_seconds 10.737,
history entry) and repackaged. A 0.05 shortfall at t1 = 10.7 s is a
0.53 s delay, close to NIRCam's. The NIS exposure (t1 = 26.8 s) gets
k = 0.980 from the delay model; its own fit says 0.98-1.06 per channel.
The sky-anchored offsets come out +5.4 to +6.8 DN on every NIRISS field
exposure against the darks' 0: the epoch drift again.

NIRISS field results (`results/2026-10-26_niriss_field_v2/`, median
background, flat applied, calibration with the new timing; 
`compare_backgrounds --flat`, `compare_field_photometry`):

| exposure | chain | row-mean | col-mean | sky % | LS % |
|---|---|---|---|---|---|
| jw03807 NISRAPID 8 gr, sky 0.58 | SIRS | 0.0165 | 0.0384 | | |
| | jwst median | 0.0147 | 0.0117 | -0.00 | 3.09 |
| | stream clean | 0.0022 | 0.0039 | -0.17 | 3.78 |
| | stream clean + reset fit | 0.0021 | 0.0037 | -0.59 | 3.68 |
| jw01571 NIS 7 gr, sky 1.38 | SIRS | 0.0045 | 0.0066 | | |
| | jwst median | 0.0046 | 0.0024 | +0.04 | 0.12 |
| | stream clean | 0.0006 | 0.0009 | -0.07 | 0.40 |
| | stream clean + reset fit | 0.0006 | 0.0009 | -0.06 | 0.41 |

NIRISS carries much stronger column structure than NIRCam (SIRS
column-mean scatter 0.038 DN/s on jw03807, visible as vertical bands in
the figure); the stream clean removes it to 0.0039 (jwst median
0.0117). Photometry on jw03807, complete apertures, 75 sources: every
chain within 0.5 percent of SIRS in every quartile (reset fit 0.9966,
1.0020, 1.0004, 1.0002).

Photometry on jw01571 (NIS) looked alarming at first: the reset fit's
quartiles 1.14-1.61 with scatter 0.3-0.7. Three causes, in order found:
(1) `compare_field_photometry` summed apertures with NaN pixels as
zero, so for a saturating star whose core OLS_C leaves without a rate
and the reset fit fills (2197 recovered pixels on this exposure, core
values 170-465 DN/s consistent with each pixel's first difference) the
"ratio" measured the missing core; the script now takes complete
apertures only. (2) With that, 16 sources remained, the brightest, and
their quartiles were still 1.09-1.61: small numbers. (3) The controlled
comparison (`results/2026-10-26_niriss_recovered_jw01571/chain_matrix.npz`:
the same ramp fitted raw and cleaned with OLS_C, stcal LIKELY and the
reset fit, 115 sources complete in all six) gives, relative to raw
OLS_C: cleaned OLS 1.001, LIKELY 1.008-1.009, reset fit 1.013-1.014
(mad 0.02-0.03), the same on the raw and the cleaned ramp. So the
reset fit sits 1.3 percent above OLS_C here, 0.8 of it the likelihood
fitter itself and 0.5 the reset point. Per pixel at 5-50 DN/s, LIKELY
and the reset fit have an upper tail of +5 to +7 percent at the 84th
percentile against OLS_C, present for unflagged pixels too (n = 347),
and the brightest pixels (50-200 DN/s, 3-5 of 7 groups flagged by the
jump step and charge migration, values in the 2 and 4 flags) differ by
up to 17 percent either way: on ramps where most groups are flagged
OLS_C and the likelihood fit weigh the few remaining differences
differently and neither has a claim to the truth. This is the
LIKELY-vs-OLS_C observation of the CEERS section again, larger on NIS
7-group data with charge-migration flags, and one for the stcal
developers; the reset point adds half a percent to it. On the NISRAPID
exposure the same comparison is within 0.2 percent.

Open: a dispersed exposure (its smooth dispersed background would need
the box model or a scene model; the direct images here are not that),
and the extended-emission field. (NIRISS never has a zeroframe, James
notes: NISRAPID is one frame per group and NIS averages four frames
without a separate first frame; the zeroframe is a NIRCam product for
NFRAMES > 1.)

### 2026-10-26 (later): the occulting spots, the flat's DQ, spikes on both axes, ASDF defaults

James, from the NIRISS figure: the four occulting spots are not dealt
with, the largest raises the stream clean's estimate around it; are
we using the DO_NOT_USE flags from dq_init at all? And: NIRISS has its
PSF spikes along the 1/f direction, not the column correction, unlike
NIRCam. Also: package the defaults as ASDF rather than JSON, like a
future CRDS reference. And a request for the total noise against
number of groups on linear axes, as the jdocs detector pages do it.

DQ use, as it was: the draft rate's DO_NOT_USE pixels become NaN and
are masked (and the pixeldq in the extra passes), and each group
difference drops DO_NOT_USE, JUMP_DET and SATURATED groups through the
jwst `_mask_unusable` helper. The spots, measured on jw03807: two dark
blobs of 581 and 320 px at (1885,1060) and (1862,749) at 0.07 and 0.04
DN/s against a 0.58 sky, with DQ 'none' in the mask reference, the
PIXELDQ and the rate DQ (no NON_SCIENCE anywhere in the frame), and 7
and 0 percent of them in the scene mask. Before writing a detector for
dark regions (a sign-flipped chisel pass; written, measured, dropped),
James asked for the other DQ to be checked: the FLAT reference marks
them DO_NOT_USE with a flat value of 0.0. The jwst `read_flat_file`
helper we use replaces zeros and NaNs in the flat by a smoothed local
value on purpose, so that dividing by the flat is harmless there, and
our `flat_ok` test never saw them. Effect, change against SIRS around
the big spot in units of the sky: +0.085 (0-8 px), +0.074 (8-15),
+0.040 (15-25), -0.007 (25-40), -0.025 (40-80, the general level); the
stream filter read the 88 percent deficit as negative slow noise and
raised the cleaned values at and around the spot.

Fix: `unusable_in_flat(model, flat_filename)` reads the flat's own DQ
and data and the step keeps DO_NOT_USE, zero and NaN flat pixels out
of the background mask (first pass and extra passes; the saved mask
shows it). Rerun (`results/2026-10-26_niriss_field_v3/`): the spots
are masked entirely, the 8-15 px ring goes from +0.074 to +0.004 and
the rings beyond sit at the general level (-0.03 to -0.04); sky
statistics and photometry unchanged to the last digit. Tests: the
helper on a flat with a DO_NOT_USE block and a zero block, and the
step with an override flat writing a mask that excludes the block.
Side effect of the flat default: `get_reference_file` RAISES
CrdsLookupError on an exposure type without a flat (a dark), in the
prefetch too, which took every dark experiment down; the step now
fetches the flat in `process`, treats a CRDS miss as no flat, and sets
`prefetch_references = False`.

Spikes on both axes: `spike_mask` walks each star's row as well as its
column (the same walker on transposed views; `axes` option). For
NIRCam the column spike is the one that biases the per-column offsets;
for NIRISS, which reads out along science columns, the column spike
lies along the stream (where the Wiener estimate would dig a local
trough through it) and the row spike is the one the column offsets see.
Cost on NIRCam frames: none measurable (no horizontal spikes).

The defaults are now one ASDF file per detector,
`src/snowblind/data/reset_calib_<detector>.asdf` (James: so that one
detector can be updated alone, and NIRSpec is two more files), each
with a history entry naming its source calibration file; the JSON and
the single table are gone.

Memory: running two 100-group dark experiments and the NIRISS panel
chain at once exhausted the laptop's 18 GB and crashed it. One heavy
job at a time from now on; the dark runs for the linear noise figure
(`results/2026-10-26_darks_vs_ngroups/`, NIS and NRCA1, n = 3 to 100,
variants sirs, sirs_jwst_median, sirs_wiener_none) run sequentially.

Noise against ramp length, both detectors, one integration, no
background model (the dark setting), pixel scatter of the OLS_C rate in
DN/s (`results/2026-10-26_darks_vs_ngroups/n*/{NIS,NRCA1}/summary.json`):

| groups | NIS SIRS | NIS jwst median | NIS stream | NRCA1 SIRS | NRCA1 jwst median | NRCA1 stream |
|---|---|---|---|---|---|---|
| 3 | 0.4655 | 0.4417 | 0.4134 | 0.3486 | 0.3382 | 0.3284 |
| 5 | 0.2120 | 0.2018 | 0.1878 | 0.1578 | 0.1535 | 0.1481 |
| 10 | 0.0781 | 0.0753 | 0.0687 | 0.0558 | 0.0550 | 0.0526 |
| 20 | 0.0298 | 0.0303 | 0.0270 | 0.0205 | 0.0213 | 0.0193 |
| 50 | 0.0106 | 0.0121 | 0.0099 | 0.0059 | 0.0074 | 0.0056 |
| 100 | 0.0060 | 0.0074 | 0.0058 | 0.0027 | 0.0042 | 0.0025 |

The same picture on NIRISS as on NIRCam: the stream clean is below SIRS
at every length (11 to 12 percent at 3 groups, 3 to 7 percent at 100),
and the jwst median helps at short ramps and HURTS beyond 20 groups (24
percent above SIRS at 100 groups on NIS, 55 on NRCA1), the random walk
of per-difference medians along the ramp. `noise_vs_groups_linear.png`
(scripts/make_figures.py `noise_vs_groups_linear`) shows it as total
noise per integration in DN (rate scatter times integration time,
10.74 s per group on both) on linear axes, the form of the jdocs
detector pages, one panel per detector.

### 2026-10-26: one pipeline run, hooks as classes, citation, and the WFSS plan

James asked whether both steps can be hooks of one Detector1 run, noted
that step classes can be passed as hooks, that the GitHub "cite this
repository" link from CITATION.cff is not working, and that the NIRISS
WFSS modes have a ``wfssbkg`` background reference the WFSS
``BackgroundStep`` scales and subtracts, which the stream clean will
need for dispersed data.

Hooks: stpipe runs post-hooks on a step's result and takes a hook's
return value as the new result, so `StreamCleanStep` (ramp in, ramp
out) hooks after jump; `ResetRampFitStep` returns (rate, rateints) and
cannot be a hook of `calwebb_detector1`, whose later steps and
`setup_output` expect a ramp, and a pre-hook of ramp_fit has the same
problem. `_hooks.hook_from_string` accepts a dotted name, a Step
subclass, a Step instance or a function, so classes work in Python.
What does give one run: `snowblind.pipeline.Detector1Pipeline`
(``strun snowblind_detector1``), a subclass of the jwst pipeline with
`step_defs["ramp_fit"] = ResetRampFitStep`, `StreamCleanStep` appended
to the jump step's ``post_hooks`` and ``clean_flicker_noise.skip``
forced in ``__init__`` (so CRDS cannot turn the jwst step back on), and
`get_config_reftype` returning ``pars-detector1pipeline`` so that the
jwst pipeline's CRDS parameters apply to it (a subclass would otherwise
get none, as pars are looked up by class name). Registered as a
pipeline in `_get_steps`. Verified with strun on jw03383060001 NIRISS
(results/2026-10-26_niriss_strun_v3): one command, `_rate` and
`_rateints` with reset_rampfit COMPLETE. The rate products lacked the
`stream_clean` record, though the ramp had it: jwst's
`create_image_model` builds the rate with `update(input_model)`, which
keeps only the step records in the core schema, and `stream_clean`
(like `snowblind` and `jump_plus`) is not in it; non-schema records
also get no `S_` FITS keyword and live in the ASDF metadata only. The
reset step now carries every cal_step record of the ramp that the
products lack, so the one-command route records the cleaning; the
stock-pipeline route (hook plus jwst ramp_fit) cannot, except on a
saved `_jump`. README, changelog and write-up use the pipeline as the
recipe; the hook-only form and the standalone steps stay documented.

James then asked whether the stock pipeline could do it with hooks
alone (a pre-hook of ramp_fit with ramp_fit skipped, or both steps as
post-hooks of jump), and for more flexibility than the subclass. From
stpipe's `Step.run`: pre-hooks run BEFORE the skip test and their return
value replaces the step's input; a skipped step stamps `cal_step.<alias>
= SKIPPED` on that input and returns it; post-hooks run AFTER, skipped
or not, and their return value becomes the result. jwst's `RampFitStep`
returns ``(rate, rateints)``, which is why `ResetRampFitStep` does the
same and fits the slot; Detector1 unpacks the pair and runs gain_scale
on each. Hence: two post-hooks of jump with ramp_fit skipped hands the
pair down the pipeline and gain_scale fails on it; a pre-hook of a
skipped ramp_fit runs, but the skipped step stamps SKIPPED on the rate
image and `setup_output` names it `_ramp`; a POST-hook of the skipped
ramp_fit that returns the rate image alone, with the ramp_fit record set
COMPLETE on it (which the step does), gives a proper `_rate` through
gain_scale and `setup_output`, and the hook's own `save_results` writes
the `_rateints`. That is the new `as_hook` option of `ResetRampFitStep`
and the all-hooks recipe in the README. The pipeline class gets two
switches, ``--stream_clean`` and ``--reset_fit``, so either jwst step
can be kept.

### 2026-10-26 (night): second review round, the docs build, and a first look at WFSS

`/code-review high` over the day's modules found ten things; all fixed:
(1) the pipeline failed on NIRCam, since the CRDS `pars-detector1pipeline`
reference carries `steps.ramp_fit.suppress_one_group` and stpipe validates
that section against `ResetRampFitStep.spec` (NIRISS had no such entry,
so the earlier verification passed); the step's spec now accepts the jwst
ramp_fit parameters (`algorithm`, `int_name`, `opt_name`,
`suppress_one_group`, `firstgroup`, `lastgroup`, `maximum_cores`), none
of which it uses, `firstgroup`/`lastgroup` reported when set; this also
lets `--reset_fit=False` take stock ramp_fit parameters. (2) A
`scene_model` is in raw DN/s but was handed to `prepare_difference` as a
fixed background, which the flat path multiplies by the flat, and
subtracted raw from the flat-fielded draft rate in the chisel mask; the
step now divides the scene by the flat where a flat is in use and passes
the raw scene for the raw extra-pass draft. (3) `binary_dilation` with
`iterations=erode + opening` at zero means "until nothing changes" in
scipy; guarded. (4) A `t1_scale` override dropped the file's per-channel
offsets in `_reset_prior` while `reset_prior_map` kept them, so the
first-read prior of the stream clean and the fit's prior differed;
offsets now stay, the override applies to every channel and is not
rescaled (it is for this exposure). (5) `calibrate` took `nframes` from
the last dark; darks with different frames per group now raise. (6) The
mask-shape skip ran after the flat AND and would crash first; check
moved up. (7) `write_resetfit` recorded the scalar scale and the
offset-free level; it now stores the per-channel scales and offsets and
folds the mean offset into the level median. (8) A `post_hooks` string
was split into characters; wrapped. (9) `_first_resultant` ran twice
when both anchors were on; computed once and passed. (10) A logger test
depended on test order; fixed. Verified on NIRCam with
`strun snowblind_detector1` on the workshop NRCA1 BRIGHT1 exposure
(results/2026-10-26_nrca1_strun_pipeline).

Docs: `docs/conf.py` and `docs/index.rst` were still the project
template (`my_package`, an automodapi of a module that does not exist),
so the build failed; both rewritten, and three simple tables in the
write-up had labels running over their first column; the build is clean
with 20 figures.

WFSS first look (`~/data/jwst_niriss_test/wfss/`: GR150C 1x8 and 5x8
groups NIS, GR150R 6x8 and 1x19; the GR150C single-integration exposure
jw03383014 through `make_field_panels` with median and box256, flat
applied: `results/2026-10-26_wfss/`): sky 0.75 DN/s, 11.2 percent masked
(0 stars). SIRS column-mean scatter 0.0188 DN/s (the dispersion runs
along columns for GR150C), rows 0.0044; jwst median 0.0206 / 0.0043;
stream clean median 0.0007 / 0.0021; box256 0.0195 / 0.0029. But the
large-scale change is 2.86 percent of the sky for the median background
(jwst 0.31, box256 1.30) with 5th/95th percentiles of the per-pixel
change at -0.042 / +0.046: the dispersed background, which varies
smoothly along the dispersion by more than the 1/f does, is read by the
Wiener filter as slow noise under a scalar background, and the box
model keeps the structure but gives back the column banding. Exactly the
case for the ``wfssbkg`` reference as the background model (previous
entry); the data are now on disk to build it against. The workshop set
(`~/data/jwst_workshop_1f/`: NRCA1 BRIGHT1 5 groups, NRCA3 MEDIUM8 5
groups, NIS NISRAPID 5 groups, and NRCB2 BRIGHT1 3 groups on NGC 1566
with extended emission) is the next test set; the NRCB2 exposure is the
extended-emission case the plan has wanted.

### 2026-10-26 (late): the workshop set, crowded fields and extended emission

All four non-NIRSpec workshop exposures through `make_field_panels`
(defaults: median background, flat applied; NGC 1566 also with the box
models), `compare_backgrounds --flat` and `compare_field_photometry`
(`results/2026-10-26_workshop/`, NGC 1566 rerun with box32 first in
`results/2026-10-26_workshop_ngc1566/`):

| exposure | masked | SIRS rows / cols | jwst median | stream clean (median) | sky change | photometry |
|---|---|---|---|---|---|---|
| NRCA1 F070W BRIGHT1 5 gr, LMC, 2472 sources | 19.5% | 0.0195 / 0.0085 | 0.0066 / 0.0085 | 0.0020 / 0.0014 | -1.0% (jwst +0.0) | within 0.15% |
| NRCA3 F150W MEDIUM8 5 gr, sparse | 8.5% | 0.0015 / 0.0011 | 0.0006 / 0.0011 | 0.0002 / 0.0001 | -0.2% | within 0.3% |
| NIS F200W NISRAPID 5 gr, LMC, 7876 sources | 26.1% | 0.058 / 0.097 | 0.058 / 0.056 | 0.0099 / 0.0101 | -2.7% (jwst -13.9%) | within 0.2% |
| NRCB2 F200W BRIGHT1 3 gr, NGC 1566 | 11.9% | 0.142 / 0.544 | 0.122 / 0.542 | 0.031 / 0.015 | -8.0%, LS 53% | see below |

The two LMC fields are the crowding limit: a quarter of the frame
masked, the unmasked pixels still carrying star wings, so the tiled sky
on 32 px tiles ranges 0.6-1.4 of the median; the stream clean lowers the
sky by 1 (NRCam) and 2.7 percent (NIRISS) and the large-scale change is
8-16 percent, while photometry is unchanged to 0.2 percent. The jwst
median lowers the NIRISS LMC sky by 14 percent: its clipped-median
background on a crowded field. The sparse MEDIUM8 field is the usual
picture, 7x on rows and 10x on columns below SIRS.

NGC 1566 fills the left half of the frame; the chisel mask covers the
core (12 percent of the frame) and the disk stays unmasked, the 32 px
tiled sky running from 0.18 to 3.4 of the median. Under the scalar
median the stream clean removes the disk's gradient (change map: -30
percent on the left, +30 on the right; LS 53 percent of the sky, sky
-8 percent), and the jwst row medians do the same in horizontal
stripes of +-30 percent. In the galaxy-free right third (columns
1500-2040, masked sky pixels; the galaxy's outer light still gives SIRS
0.084 / 0.067 DN/s there):

| chain | rows | cols | median change |
|---|---|---|---|
| SIRS | 0.084 | 0.067 | 0 |
| jwst median | 0.063 | 0.069 | -0.031 sky |
| stream box32 | 0.061 | 0.069 | +0.001 sky |
| stream box128 | 0.036 | 0.063 | +0.008 sky |
| stream box256 | 0.034 | 0.015 | +0.040 sky |
| stream median | 0.009 | 0.003 | +0.574 sky |

The box model keeps the galaxy and gives back the 1/f slower than its
box: box 32 is unbiased (+0.1 percent) and removes 28 percent of the
row structure in the sky region, box 128 removes 57 percent for 0.8
percent of bias, box 256 removes 60 percent of rows and most of the
columns for 4 percent. Photometry of the 13 bright sources: box 32 and
box 256 at -0.3 to -0.5 percent, jwst -0.1, the reset fit +0.3 to +0.9
(a 3-group BRIGHT1 ramp leans on the first read; NRCB2's k is 0.88, the
lowest of the ten, and the direction is consistent with k slightly too
low; noted, not pursued). Conclusion as before, now measured: in one
exposure, extended emission and slow 1/f are the same thing to any
estimator; the box model (``integration_model``, box 32) is the safe
default for such a field and a scene model from the other dithers the
way to do better. New: the step measures the structure of the unmasked
background (`background_structure`: the 32 px tiled sky's 1st-99th
percentile range over its median) when the background is the scalar
median, and warns above 0.5 (NGC 1566 3.2, WFSS 0.7, the LMC fields
0.6-0.8; CEERS, NEXUS and the MEDIUM8 field 0.1-0.15), naming the box
model. The figure set gets the NGC 1566 panels with the box-32 clean,
and the field figures now label the stream clean's background from
`panels.json` rather than assuming the median.

### 2026-10-26 (late): pruning the API

James asked which options were dead ends. Removed from `StreamCleanStep`:
`background_method='model'` (the per-difference box-spline, the random
walk of the first finding), `clip_sigma` and `iterations` (per-difference
clipping and re-estimation, the same), `fit_method='boxcar'` with
`window_length` (a sanity-check estimator, never in the write-up),
`scene_mask='clip'` with `n_sigma` (the jwst mask, kept only for the
injection table's clip rows; the jwst step itself is the living
comparison), `psd_prior` and `protect_period` (implemented, never
measured), `mask_science_regions` (a NIRSpec/MIRI helper option), and
`mask_passes` after one measurement: on CEERS NRCALONG a second pass
changes row and column means by 0.1 percent and the sky by 0.006 percent,
with slightly wider tails (`results/2026-10-26_mask_passes/`). From
`ResetRampFitStep`: `per_pixel_sigma_b` (never measured; every result
uses the scalar median). The experiment scripts lose the matching
variants (`*_clipped`, `sirs_boxcar_box32`, `*_nsig1`, `*_clipmask`,
`--per-pixel`); the write-up tables keep their rows with a note. Kept
with a measurement behind them: `exposure_model`, `scene_model`,
`use_pedestal_mean` and `prior_highpass`, `anchor_timing_to_stars`,
`reset_offset`, `rejection_threshold`, and the defaults.

### 2026-10-26 (night): the WFSS background reference as the background model

James: subtract a scaled CRDS background from each difference before the
1/f removal, fit once for all differences so that nothing random-walks.
The reftype is ``bkg`` (`jwst_niriss_bkg_0027.fits` for GR150C F200W,
`_0029` for GR150R), a normalised image (median 1.00, range 0.07-1.55)
in raw detector units, which the pipeline's WFSS background step scales
to the data with an error-weighted mean ratio and 1 percent clipping
behind a catalog-based source mask. Here: one scale per exposure, the
median of the ratio of the ramp's least-squares slope image to the
reference over the chisel mask where the reference is above 0.2 of its
median (0.758 DN/s on GR150C, 0.535 on GR150R); the scaled reference
then goes through the fixed-background path like a scene model, scaled
to each difference by its time only.

Trial through the `scene_model` option (`results/2026-10-26_wfss_reference/`),
GR150C jw03383014 (NIS, 8 groups) and GR150R jw04681316 (NIS, 19 groups).
Direct check first: the 64 px smoothed sky over the mask runs from 0.35
to 1.06 of the median before and -0.02 to +0.04 after subtracting the
scaled reference. A trap in the metrics: on a dispersed exposure the
row- and column-mean scatter of the rate is the dispersed background
itself (it varies along the dispersion), so the median background's
"0.0008" column scatter on GR150C is the background flattened away and
the reference run's "0.024" is the background kept. Measured properly,
on the background-subtracted flat-fielded rate (rate minus scale x bkg)
over the mask:

| grism | chain | rows | cols | residual median | 5 / 95 pct |
|---|---|---|---|---|---|
| GR150C | SIRS | 0.0059 | 0.0063 | -0.0007 | -0.101 / +0.107 |
| GR150C | jwst median | 0.0065 | 0.0172 | +0.0027 | -0.107 / +0.130 |
| GR150C | stream, median bkg | 0.0071 | 0.0239 | -0.0087 | -0.121 / +0.261 |
| GR150C | stream, box 256 | 0.0050 | 0.0060 | +0.0024 | -0.107 / +0.208 |
| GR150C | stream, reference | 0.0004 | 0.0005 | -0.0013 | -0.099 / +0.103 |
| GR150R | SIRS | 0.0021 | 0.0067 | -0.0008 | -0.079 / +0.093 |
| GR150R | jwst median | 0.0021 | 0.0041 | +0.0011 | -0.077 / +0.089 |
| GR150R | stream, median bkg | 0.0250 | 0.0077 | -0.0208 | -0.119 / +0.396 |
| GR150R | stream, box 256 | 0.0053 | 0.0078 | -0.0009 | -0.096 / +0.282 |
| GR150R | stream, reference | 0.0004 | 0.0003 | -0.0014 | -0.076 / +0.080 |

(rows and cols in DN/s, the rest in units of the sky; sky 0.749 and
0.529 DN/s.) With the reference the banding drops 12-15x on GR150C and
5-20x on GR150R, the residual median is within 0.14 percent of the sky
and the 5/95 percentiles are SIRS's own, i.e. nothing at large scales
is touched; the median background flattens the dispersed sky (the
warning fires, 0.38 and 0.32 to 1.05), the box model gains little, and
the jwst median imprints column structure on GR150C. Implemented as
`background_method='crds_background'` (first named `reference`; James:
that is anodyne for a per-filter, per-grism background, and it should be
the default on WFSS, where it now is): ``bkg`` joins the step's reference
types, fetched in `process`; the scale is fitted on the step's own
least-squares slope image in raw DN/s (no flat needed, and CRDS-free in
the tests) over the mask, the scaled reference becomes the scene model
(divided by the flat where one is in use), the chisel mask is rebuilt
on the residual, and the structure warning names the option for
dispersed exposures. `make_field_panels` takes `crds_background` as a
background setting.

Through the step itself (`results/2026-10-26_wfss_panels/`, median and
reference backgrounds, flat applied, with the reset fit on the reference
chain): scales 0.7584 (GR150C) and 0.5342 DN/s (GR150R), masks 12.1 and
17.6 percent; banding of the background-subtracted rate, rows / cols in
DN/s: GR150C SIRS 0.0059 / 0.0064, stream reference 0.0004 / 0.0005,
reference + reset fit 0.0006 / 0.0007, stream median 0.0071 / 0.0240;
GR150R SIRS 0.0021 / 0.0067, reference 0.0004 / 0.0003, with the reset
fit 0.0005 / 0.0003, median 0.0250 / 0.0077. Residual medians within
0.2 percent of the sky for the reference chains. The two multi-integration
WFSS files (5 and 6 integrations) are the next check, for one scale per
exposure against integrations. Figures `field_rate_images_wfss_gr150c`
and `_gr150r` added to the set. Environment note: a non-editable
snowblind 1.0.1 was installed into site-packages at 17:03 (not by this
session), shadowing the editable tree; tests and scripts run with
`PYTHONPATH=src` until the editable install is restored.

### 2026-10-27: skip the grisms that disperse along the stream

James: GR150R causes real problems for the Wiener filter, dispersed
spectra sit right in its frequency range and pile up with the real
spectral variations on the same spatial scales; refuse to run on NIRISS
GRxxxR, or at least warn loudly. The geometry: NIRISS reads along science
columns and GR150R disperses along detector rows, the same axis, so a
trace is a 100-200 clock feature in the stream with continuum shape and
lines at the 30-300 px scales the gain passes; masked traces are bridged,
faint traces and wings under the chisel threshold lose spectral shape to
the noise estimate. No sky statistic sees it, which is why the GR150R
numbers looked as good as GR150C's. GR150C crosses the stream in a few
clocks (safe). NIRCam reads along rows: GRISMR is the stream-parallel
case there, GRISMC safe. Implemented as `dispersion_along_stream(model)`
from the filter and pupil keywords (works before any WCS) and a skip in
`StreamCleanStep.process` with a warning that explains why, overridden
by `allow_dispersion_along_stream=True`. The pipeline class inherits it
through the hook. Three tests. The GR150R rows and figure stay as the
record; an injection test of faint traces along the stream (synthetic
traces of known shape in the GR150R ramp, recovered flux and shape
along the trace, against the same traces across the stream in GR150C)
is the way to size the bias and to decide whether a longer cutoff
period could make those exposures safe.

Citation: README gets a "Citing snowblind" section at the top with the
concept DOI and a BibTeX entry; CITATION.cff's author becomes a
family-names/given-names person entry (GitHub's own examples use that
form) and the cff-from-621 static block in pyproject matches.

WFSS, for later: a dispersed NIRISS exposure has a smooth background
(dispersed zodiacal light and stray light) that the scalar median cannot
model and the box model would only partly keep out of the noise
estimate. The pipeline's WFSS `BackgroundStep` scales the ``wfssbkg``
reference image to the exposure and subtracts it. For the stream clean
the same image is the right background model: add a
``background_method='reference'`` that fetches ``wfssbkg`` through
CRDS (``reference_file_types``), fits one scale per group difference to
the masked, flat-fielded difference (the sky grows linearly along the
ramp, so one scale per difference is the whole model) and subtracts
scale x reference x flat from the raw difference before the noise fit,
exactly as the median background is handled now; the scene mask then
sees sources (the dispersed traces, extended along the dispersion
axis) against a flat residual. Needs a dispersed exposure to test; the
six files on disk are the direct images. The same mechanism serves
NIRCam WFSS (``wfssbkg`` exists for it too).

### 2026-10-27: NIRCam WFSS, and the per-column offsets on dispersed exposures

James: no injection test (it will fail for NIRISS and worse for NIRCam,
whose traces are longer); we just do not correct when the dispersion is
parallel to the stream. Check NIRCam WFSS, and verify that the
column-level correction orthogonal to the stream is safe for all the WFSS
data even when it aligns with the dispersion.

Data (~/data/nircam_wfss/): program 5398 F444W GRISMC (NRCALONG,
SHALLOW2, 5 groups, a deep field, 11 percent masked) and GRISMR (MEDIUM8,
7 groups), program 1076 F444W GRISMC of P330-E (BRIGHT1, 5 groups). CRDS
has `bkg` references for both long-wavelength detectors and both grisms.
The GRISMR exposure is skipped (SKIPPED, data untouched).

The column-level correction is not safe as it was. Metric: per stream
(row), the median correction in trace pixels minus that in the sky pixels
of the same row; and the summed flux of the longest traces (object pixels
of the mask, local sky from a box around) before and after. On 5398
GRISMC with `column_offsets` on: excess -0.019 DN/s (a third of a pixel's
noise, 0.061), the eight longest traces lose 3 to 29 percent; on P330-E
-0.011 and 4 to 13 percent on the fainter traces. Variants (all with the
CRDS background): row_median fit -0.018 (so not the Wiener filter),
clean_first_group off -0.019, mask_dilation 6 -0.015, column_offsets off
+0.003. The cause: each column's masked mean carries the outskirts of its
detected traces and the whole of its undetected ones, 1000-plus pixels
long, and the offset removed takes that out of every trace in the column.
Emulated estimators on the rate image (the correction is linear, so the
column means of the rate reproduce it) with a hold-out (offsets from even
rows, banding on odd rows): masked median -0.005, dilation 12/24/48
across the dispersion -0.007/-0.007/-0.006 (plateau: undetected traces),
high-pass of the offsets across 17 columns 0.000 but the removable
column structure is largely at scales above 17 columns (0.0071 of 0.0109
left), so the high-pass gives the banding back. Nothing separates a
smooth per-column pattern from dispersed outskirts; the choice is off.
What it costs on 5398: column-median scatter 0.0128 instead of 0.0027
(SIRS 0.0225), the trace excess +0.003 (0.08 of a pixel's noise), the
longest traces -4 to +4 percent (jwst median -5 to +6: the measurement's
own scatter). Implemented: `column_offsets` forced False on WFSS
exposures in `process`, logged, spec comment, two tests.

NIRISS GR150C with the same metric: -0.0045 with the offsets on (the
per-row offsets there), -0.0036 off; 21 long traces change by a median
-0.1 percent (quartiles -2.5, +0.4). Dilation on this field: masked
fraction 0.12 -> 0.32 (4) -> 0.61 (8) -> 0.95 (16), excess -0.0029,
-0.0019, -0.0004; not a route. The GR150C row banding with the offsets
off is 0.0035 (0.0010 on; SIRS 0.0059), columns 0.0010.

P330-E also shows a 0.2 DN/s square over the top 400 rows (scattered
light the reference lacks); the mask catches a tenth of it, the stream
clean removes 0.06 DN/s of it, the jwst median adds 0.16. Extended
emission beyond the reference is the `scene_model` case.

Figures: FIELD_PANEL_SETS gains NIRCam GRISMC
(results/2026-10-27_nircam_wfss_panels_v2) and GR150C moves to the rerun
(results/2026-10-27_wfss_panels_v2). Write-up, README, CHANGES updated;
the 1.0.0 changelog entry no longer claims the WFSS work (released
2026-10-26 without it); the skip warning quotes the NIRCam trace length;
the remaining in-body imports in the tests moved to the module tops.

Also seen on 5398 GRISMC: the reference spans 0.26 to 1.01 along the
dispersion (rows) and the single scale leaves a residual whose row
medians run -0.04 to +0.14 DN/s (sky 0.58); the filter removes that
gradient as slow noise (per-pixel change p5 -0.145 DN/s). A scale plus
offset fit (a 0.41, b +0.15 DN/s) halves the residual span, so part of
the sky is an undispersed additive component; not implemented, since the
filter takes the residual either way and the trace excess (+0.003) shows
extraction is unaffected. P330-E's residual is a shape mismatch (the
offset fit does not change it). NIRISS GR150C: reference 0.70 to 1.06,
residual 0.01 DN/s.

### 2026-10-27 (later): the residuals James saw in the GRISMC figure

James: odd residuals aligned in the cross-dispersion direction, the
correction lowers the background there; does not look like the Wiener
filter. Also: the figure is not flat-fielded, cross-hatching visible.
Three more causes, all fixed:

1. The CRDS `bkg` references differ in units. NIRCam's (bkg_0032) has no
   pixel pattern (correlation of its pixel-scale structure with the
   flat's 0.000); NIRISS's (bkg_0031) carries the flat (0.956). The step
   took both as raw. On NIRCam that left sky x (flat - 1) in the residual
   and the flat's large-scale structure went out as slow noise: the dark
   patches correlate 0.72 with -scene x (flat - 1), and the largest one
   (rows 1463-1542, cols 0-345, -0.04 DN/s) sits where the flat is 1.05
   against a row median of 0.99. jwst's spec2 subtracts the bkg before
   flat_field with no flat handling either. New: `reference_carries_flat`
   (correlation > 0.5) and `reference_in_raw_units` (multiply by the flat
   otherwise), used in `_fit_reference` and by the panel script for the
   saved scene; five tests. The cross-hatching was the same thing: the
   figure divides (rate - scene) by the flat.
2. The scene mask lost the brightest trace. `remove_stripes` took
   per-column medians of the draft rate; the trace at col 1516 covers
   1190 of 2048 rows, median 6.5 DN/s, so it vanished from the detection
   image except where brighter than that: masked 55 percent of its
   length, one-sided (the mask started at the peak). Its unmasked flux at
   a fixed clock every row gave a sinusoid at the row period in the
   Wiener estimate: a bowl of -0.05 to +0.02 DN/s across channel 2 in
   rows 1321-1392 (the row_median variant shows the same channel offset,
   so not the Wiener filter as such). New: `remove_stripes(axes, clip)`
   clips |x| > 3 MAD before the medians; `chisel_mask(stripe_axes)`;
   the step uses only the stream axis on WFSS (from fastaxis). Trace now
   fully masked; mask 10.2 -> 17.5 percent of the deep field; imaging
   masks change by 0.5 percent of pixels (NEXUS, NIRISS F200W), NGC 1566
   by 3.7 (more disk masked; the right-third banding and bias numbers
   are unchanged to the last digit: box32 rows 0.054, cols 0.067, +0.1
   percent). Four tests; the noise-free stripe test gets clip=None.
3. The residual pattern at the row frequency. Mean correction against
   clock phase within a channel row, over the frame: fundamental 0.002-
   0.005 DN/s on GRISMC against 0.0002-0.0005 on NEXUS imaging. The
   unmasked wings and faint traces along columns repeat every row. New:
   `notch_period` (Gaussian notch, relative sigma 0.02) applied in
   `channel_gain` at `channel_width + LINE_OVERHEAD` (524 clocks) on WFSS
   exposures; fundamental -> 0.0001-0.0009, per-stream trace excess
   -0.0014 -> -0.0008, column banding 0.0093 -> 0.0104 (the notch leaves
   that sliver). Per-object within-row scatter (0.08 of the objects'
   brightness) is unchanged by the notch: it is the real within-row 1/f
   that the row median cannot see, not a bias.

Final runs (results/2026-10-27_nircam_wfss_panels_v5,
2026-10-27_wfss_panels_v5): 5398 trace excess -0.0008 +- 0.0033 DN/s,
P330-E -0.0037 +- 0.0105, NIRISS GR150C -0.0012 +- 0.0038 (was -0.0036);
banding rows/cols 5398 0.0146/0.0145 -> 0.0011/0.0104, P330-E
0.0275/0.0153 -> 0.0022/0.0067, NIRISS 0.0055/0.0060 -> 0.0040/0.0010.
The P330-E square at the top of the frame persists after the flat fix
(real scattered light); the dispersion-direction gradient of the
one-scale reference (-0.03..+0.13 DN/s) too. Figures regenerated;
write-up, README, CHANGES updated.

James asked how the convention is detected and whether it is defined
somewhere. It is not: no BUNIT in either SCI extension, the NIRCam
history says "stack of many science observations ... scaled such that
level is unity in a fiducial region", the NIRISS one "CV3 data ...
normalized_background" files; the jwst background_subtraction docs and
the BKG reference page describe format and scaling only; JDox (NIRCam
WFSS backgrounds) says "individually normalized, resulting in
dimensionless images" and defers to Ryan et al. 2025 (in review, not
public); Pirzkal, Boyer & Ryan 2026 (AJ 171, 232, arXiv 2602.13918)
does not describe the sky images at all. calwebb_spec2 runs bkg_subtract
before flat_field for WFSS and background_sub_wfss has no flat handling,
so the pipeline assumes raw units for both and the NIRCam cal products
keep the flat's pattern times the sky. Against the CRDS flats the
correlation is 0.82-0.97 for the four NIRISS files and -0.02..+0.01 for
the six NIRCam files (2018 Hilbert and 2025 Ryan alike); against an
unrelated ground flat NIRISS drops to 0.3, so the test needs the
matching flat, which the step uses. Added: `reference_flat_correlation`,
the value logged, a warning for 0.3-0.7 or NaN, and the spec option
`reference_units = auto | raw | flat_fielded`. Issue text for jwst
drafted for James to post.

James asked for a two-panel figure of the units mismatch made with the
standard pipeline. `scripts/make_wfss_bkg_units_figure.py`: Detector1
defaults, assign_wcs, then `subtract_wfss_bkg` (the background step's
own WFSS function; the step itself refuses to run without a source
catalog even with a user mask) with the stream clean's mask, once with
jwst_nircam_bkg_0032 as delivered and once with it multiplied by
flat_0761, both divided by the flat afterwards as flat_field does.
Factors 0.5764 and 0.5786; sky residual row/column-mean scatter
0.0199/0.0201 -> 0.0147/0.0145 DN/s, pixel 0.071 -> 0.064. Figure
docs/figures/wfss_bkg_units.jpg, in the write-up and for the issue.

James: the figure does not look right, why a big top-to-bottom gradient
after the pipeline's scaled subtraction? Measured: the ratio of the
sky to the scaled reference along rows is 2.36 at row 16, 1.61 at 128,
1.30 at 256, 0.94 at 512 and 0.98-1.02 above 1024 (reference 0032 falls
to 0.20 of its plateau at the bottom edge, the sky only to 0.48); the
P330-E exposure (2022) gives 1.99, 1.29, 1.25, 0.90, 0.98-1.04; CEERS
F444W imaging has a flat sky in every row (0.68 DN/s), so it is not the
detector. 93 percent of the smoothed residual variance is a function of
row; a*ref + b leaves p5/p95 -0.046/+0.038, adding a linear ramp does
not help. So the reference's falloff along the dispersion is too steep
in the bottom quarter, in every exposure in hand, independent of the
flat question. JDox NIRCam WFSS known issues: single master background
per detector/filter/grism, assumed constant in time and spectrum,
"can be a limiting factor", NC-WFSS02 over-subtraction, workaround a
local background subtraction on the extracted 2-D spectra. The figure
now removes the row profile from both panels (the same in both) and
quotes the 2-D structure on 33 px (0.0200 -> 0.0126), the column-mean
scatter (0.0159 -> 0.0113) and the pixel scatter (0.055 -> 0.048);
stretch +-0.06. Write-up and issue text carry both findings.

James: the default INFO log (noise spectra alpha/W/corner, sigma_b and
t1_scale arrays per channel) is not useful to an end user. Reworded: the
WFSS switches are one sentence; the mask is one line with the source
fraction and the spike-masked stars; "Cleaning <file> with the Wiener
filter along the readout stream"; a new line per integration with the
rms of the removed 1/f noise in DN per group difference over the
background pixels; the reference units as a sentence; the reset fit as
"the reset level enters as an extra resultant with a scatter of X DN,
and the first read weighs k of its nominal time", the anchoring as
ranges across channels against the calibration's. Spectra, correlation,
per-channel arrays, flat DQ fraction, chisel passes all at DEBUG.

### 2026-10-27 (afternoon): the flattened NIRISS sky, and reset_fit=False

Found that Detector1Pipeline.call(..., reset_fit=False) raised
"Extra value 'reset_calib' in root": Step.call builds the steps.ramp_fit
section from the class spec (ResetRampFitStep's), and the jwst RampFitStep
rejects the extra keys. Fixed in __init__ by dropping the reset-only
parameters (own spec minus the jwst spec minus the generic Step spec)
from the section when the jwst step takes the slot; test added.

There's a "RuntimeWarning: Mean of empty slice" at stream_clean.py:398
on every snowblind_detector1 run of the WFSS file.  Check whether it
matters. It is ours (the draft slope image that scales the CRDS
reference, np.nanmean over integrations), harmless: pixels with no
usable group in any integration became NaN, which the scale fit already
skips. Replaced by `mean_slope_per_group`, which sums the finite slopes
and divides by their count under errstate, NaN where the count is zero;
test with the warning turned into an error.

### 2026-10-27 (afternoon): CI failed on CRDS

Run 40515705526 on the fork: 18 tests died with "CRDS server connection
and cache load FAILED" (crds calls sys.exit when the server is
unreachable and the cache has no config). The workflow did set
CRDS_PATH and CRDS_SERVER_URL; the STScI server was simply unreachable
from the runner at 13:52 UTC. Reproduced locally with
CRDS_SERVER_URL=https://127.0.0.1:9 and an empty cache: 50 tests fail,
every one that goes through Step.call (the step-parameter reference
lookup) and every jwst step that used a reference file, override or not
(finalize_result asks for the CRDS context). Fix, keeping the rule that
unit tests are CRDS-free: conftest.py has an autouse fixture that makes
get_config_from_reference return an empty ConfigObj on every class of
the snowblind Step's MRO and stubs stpipe.crds_client.get_context_used;
the reset-fit tests' reference_files helper also writes a SuperBiasModel
and every call passes override_superbias, since the step looks the
superbias up for its warning. 286 pass with the server unreachable
(and 12 s instead of 17). ci.yml no longer sets the CRDS environment,
so a test that touches CRDS fails deterministically there.

### 2026-10-28: jump detection before and after the stream clean

James: the stream clean runs after the jump step and the reset fit
replaces ramp_fit; cleaning the 1/f first should let jumps be found at a
lower threshold without flagging real data. Three questions. (1) Can the
clean run before the jump step, how sensitive is it to jumps and to
snowballs, and would a jump step both before and after make sense? (2)
After cleaning, how low can the jump threshold go? Today 5-6 sigma is
needed so that rows of 1/f noise are not flagged. (3) Does the iterative
likelihood jump search of fitramp (Brandt) find jumps better than
JumpStep? His jump settings: JumpStep at 5 sigma, flag_4_neighbors with
min_jump_to_flag_neighbors 20, after_jump_flag_time1 0,
expand_large_events False, and post_hooks
SnowblindStep(min_radius=3, after_jumps=5) and JumpPlusStep, which take
over the large-event expansion and the after-jump flagging.

Facts from the code. fitramp's search is in stcal already
(likely_fit.mask_jumps, nearly line for line: leave out one difference
at 4.5 sigma, or two adjacent ones at the same chance probability), runs
in every LIKELY fit and in our fork (mask_jumps_reset), so
ResetRampFitStep already searches the cleaned ramp; jwst passes no
threshold. stcal drops both differences next to any flagged group. With
fewer than 10 usable differences (min_diffs_single_pass) JumpStep's test
works on the absolute values of the differences (iterative_jump), which
raises its effective threshold by about 0.7 sigma on faint pixels and
makes its flags lopsided (below). JumpStep.call(skip=True) does not skip:
stpipe sets skip to False for a step run on its own. flag_large_events
rebuilds its regions from all JUMP pixels, so a second JumpStep with the
expansion on would grow the first one's discs again.

The experiment: scripts/exp_jump_order.py and
scripts/exp_jump_order_pixels.py, results in
results/2026-10-28_jump_order{,_rates,_second_pass,_wide}. Each exposure
goes through Detector1 up to the dark step (SIRS refpix, no jump step)
and gets one of three sets of jump flags before the clean: James's
settings at 5 sigma, the same at 7 sigma (three- and four-group
thresholds at 7 too), or none. Exposures: CEERS F444W NRCALONG (MEDIUM8,
9 groups) and NRCA3 (MEDIUM8, 5 groups); program 1074 NRCA1 (BRIGHT1, 5
groups; crowded, few cosmic rays, uninformative); NIRISS program 3383 (NIS,
8 groups) and 4681 (NISRAPID, 20 groups); the darks 8758 NRCA1 and
NRCALONG (RAPID) and 9279 NIS (NISRAPID), first 30 groups. Four
measurements:

- Striping left in the rate: the cleaned ramp fitted by plain least
  squares; on background pixels (flat-fielded on sky) the mean of each
  readout line within each channel; the rms of those means in DN/s.
- Row crowding of jump flags: per group difference and readout line
  within a channel, the flags on background pixels not next to a
  first-pass flag; a line with 5 or more is crowded (chance alone gives
  about 0.03 per line at 4 sigma). Reported as the share of flags that
  sit in crowded lines.
- The sign test: cosmic rays only add charge, so a real jump is an upward
  difference while noise goes both ways. Downward flags on background
  count a search's false flags, upward minus downward its real jumps.
  Valid only where the false flags are symmetric, which JumpStep's are not
  on ramps of fewer than 10 differences (below).
- Rate outliers: in ResetRampFitStep's rate, (rate - median of the 5x5
  box) / ERR above +5 (up) or below -5 (down), on background pixels with
  no flag, with the flagged fraction (pixels with JUMP_DET in the rate DQ)
  as the cost. It has a flaw, found later (below).

(1) The clean without jump flags. Line-mean rms in DN/s:

| exposure | 7 sigma | 5 sigma | no flags |
|---|---|---|---|
| CEERS NRCALONG | 0.00056 | 0.00056 | 0.00078 |
| CEERS NRCA3 | 0.00032 | 0.00033 | 0.00041 |
| 1074 NRCA1 | 0.00334 | 0.00335 | 0.00363 |
| NIRISS 3383 | 0.00164 | 0.00168 | 0.00245 |
| NIRISS 4681 | 0.00368 | 0.00370 | 0.00411 |
| dark NRCA1 | 0.00032 | 0.00032 | 0.00488 |
| dark NIS | 0.00076 | 0.00080 | 0.00972 |
| dark NRCALONG | 0.00050 | 0.00051 | 0.00827 |

Without flags the clean is worse everywhere, by 9 to 49 percent on sky
and 13 to 17 times on the darks: small cosmic rays outside the scene mask
leak into the linear estimators (the draft-rate mask catches only the
bright ones). Column means behave the same way. 7 sigma cleans as well
as 5. A comparison of the readout lines through large events against
other lines of the same channel showed no effect either way, so it is
the many small hits that matter, not the snowballs. So the clean needs a
jump step first, and 7 sigma is enough for it.

(2) How low after cleaning. Row crowding of JumpStep's flags on the raw
ramps: CEERS NRCA3 59 percent of flags in crowded lines at 3.5 sigma, 29
at 4, 9 at 4.5, 2.6 at 5; NIRISS 4681 2 percent even at 5 sigma; the
NRCALONG dark 16 percent at 3.5. After cleaning 0.1 percent or less at
4.5 sigma and above everywhere (CEERS NRCA3 3.8 percent at 4, 32 at 3.5).
The clean removes the row flagging James sees. What limits the
threshold then is the plain false-flag rate. Downward JumpStep flags on
the cleaned darks at 4.5 sigma against the Gaussian count: NRCA1 698 vs
397, NIS 646 vs 391, NRCALONG 5835 vs 394. The NRCALONG tail is on
ordinary pixels (one event each, normal read noise; TELEGRAPH pixels are
enriched 58 times but are only 14 percent of them), so the floor is
each detector's own tail, not 1/f. Cleaning halves to quarters the false
flags at 3.5-4.5 sigma on the SW and NIRISS darks.

Faint jumps left after a 7 sigma first pass (sign test, away from the
first-pass flags): none on the NRCA1 and NIS darks (up within the noise
of down at every threshold from 3.5 to 5 sigma). On the NRCALONG dark
JumpStep has about 950 more up than down at 4-4.5 sigma and the
likelihood search 3600 more down than up: non-Gaussian both ways,
unresolved. On sky a few hundred to a thousand at 4.5 sigma (CEERS
NRCALONG 1113 with JumpStep, 998 with the likelihood search; NIRISS 3383
787 and 159). Many more are gained next to first-pass flags (CEERS
NRCALONG 1953-2595 at 4.5).

(3) JumpStep's test against the likelihood search on the same cleaned
ramps. The likelihood search makes more false flags at the same nominal
threshold (NRCA1 dark at 4.5 sigma 1588 down vs 698; NRCALONG 26134 vs
5835), and on the darks more down than up late in the ramp (curvature?).
JumpStep on short ramps cannot be scored by the sign test: on CEERS NRCA3
(4 differences) at 3.5 sigma it flags 56768 up and 61 down, against 3457
down expected from Gaussian noise, while the likelihood search flags
6463 up and 3728 down. On a pixel with sky signal the absolute value
folds downward excursions back up, so few are flagged; and its upward
flags far outnumber the faint real jumps the likelihood search sees (net
2735), so most of them are noise. An earlier reading on CEERS NRCALONG,
that JumpStep found 3-4 times more real jumps per flag, came from this
and is withdrawn.

End to end with ResetRampFitStep. Runs: the jump step at 5 sigma
(James's) or 7, then the clean, then ResetRampFitStep at its default 4.5
sigma or at 4; also 5 sigma with use_zeroframe False, and 7 sigma with a
second JumpStep at 4 sigma after the clean (James's settings, new flags
carried one group as JumpPlusStep does). Up/down rate outliers and
percent of pixels flagged:

| exposure | 5, fit 4.5 | 5, fit 4 | 7, fit 4.5 | 7, fit 4 | 7, JumpStep 4, fit 4.5 |
|---|---|---|---|---|---|
| CEERS NRCALONG | 764/106, 13.94 | 700/100, 14.19 | 766/106, 13.81 | 701/100, 14.06 | 757/104, 13.85 |
| CEERS NRCA3 | 582/278, 6.65 | 517/243, 6.71 | 631/281, 6.36 | 535/246, 6.44 | 488/271, 6.99 |
| NIRISS 3383 | 2341/392, 3.78 | 2323/391, 3.85 | 2342/391, 3.64 | 2322/390, 3.72 | 2336/391, 3.72 |
| NIRISS 4681 | 1900/370, 2.10 | 1890/364, 2.18 | 1900/370, 2.06 | 1889/365, 2.14 | 1896/371, 2.11 |
| dark NRCA1 | 62/480, 2.45 | 43/398, 2.58 | 63/479, 2.39 | 43/397, 2.52 | 61/470, 2.54 |
| dark NIS | 368/906, 3.65 | 266/732, 3.75 | 373/906, 3.53 | 265/731, 3.63 | 366/899, 3.59 |
| dark NRCALONG | 902/12435, 4.28 | 623/9751, 5.06 | 923/12576, 4.16 | 632/9847, 4.96 | 873/12176, 4.74 |

The flaw in the outlier count: it leaves out the pixels the fit flagged,
so a run that flags more has fewer pixels counted, and a pixel flagged
but not fixed drops out instead of counting as a failure. That favours
the runs that flag more: the fit at 4 sigma (CEERS NRCALONG up 764 ->
700; NRCALONG dark 902 -> 623 with down 12435 -> 9751, which cosmic rays
do not explain) and the second JumpStep at 4 sigma (CEERS NRCA3 631 ->
488 with 0.63 percent more pixels flagged, where JumpStep's upward flags
on 4 differences are mostly noise). Both are withdrawn as
recommendations until the count is redone over all pixels the jump step
left unflagged, compared at matched flagged fractions. On the darks down
outliers outnumber up, so most outliers there are not cosmic rays at
all. What stands: 7 sigma before the clean matches 5 sigma, with 0.06-0.29
percent fewer pixels flagged and the same outliers to a few percent;
the exception is CEERS NRCA3 (631 vs 582), which this count cannot
settle.

The zeroframe. With use_zeroframe (the default) the fit's log reported
many more pixels with new jumps on the MEDIUM8 exposures (CEERS NRCALONG
35540 vs 8887 without the zeroframe, NRCA3 16896 vs 2809), while the rate
hardly changes (up outliers 764 vs 787, 582 vs 567; the same scatter)
and the pixels newly flagged that the jump step had left clean are
about the same (9585 vs 7721, 2108 vs 2629). So the extra pixels are
ones the jump step had flagged in another group. Likely, not checked: a
cosmic ray during group 1, which the jump step can only flag at group 2,
and which the fit also sees between the zeroframe and the rest of
group 1. The exposures without a zeroframe (RAPID, one-frame NIRISS) are
identical with and without.

Decision (James): JumpStep at 7 sigma with his other settings, then the
stream clean, then ResetRampFitStep at its default 4.5 sigma.
snowblind_detector1 now sets this up. JUMP_DEFAULTS in pipeline.py (the
rejection, three- and four-group thresholds 7; flag_4_neighbors;
min_jump_to_flag_neighbors 20; after_jump_flag_time1 0;
expand_large_events False) go over the CRDS parameter references in
Detector1Pipeline.get_config_from_reference, and into the steps mapping
in __init__ for runs without CRDS; stpipe merges the caller's parameters
on top, so --steps.jump.rejection_threshold=5 still wins. Checked with
real CRDS on NIRISS 3383, whose pars-jumpstep has 4 sigma, the expansion
on, flag_4_neighbors off and 90 s of after-jump flagging. While
expand_large_events is off, SnowblindStep(min_radius=3, after_jumps=5)
and JumpPlusStep go into post_hooks unless listed, ahead of the stream
clean (so a caller's stream clean hook string gets them put in front);
expand_large_events=True gives back the jwst snowball flagging without
them. reset_fit=False warns that the jump step is still at 7 sigma. Ten
tests, three updated. Not covered by the data: ramps of 3 and 4 groups,
where the three- and four-group thresholds apply.

The log line. It said "pixels that the jump step had not flagged" but
counted every pixel with a newly dropped difference. James: give both.
It now reads "found jumps in N pixels the jump step had not flagged (x
percent), and more groups to exclude in M pixels it had flagged";
log_new_jumps takes the input JUMP mask. Four tests.

### 2026-10-28: bimodal snowball rates on the 8758 NRCA1 dark (a step at group 59)

James ran the standard pipeline on jw08758003001_05201_00001_nrca1 (2
integrations of 150 groups; SIRS, jump at 7 sigma, expand_factor 3). In
the rate the flagged snowball circles are bimodal: those flagged in
integration 1 come out high, those in integration 2 low
(~/data/jwst_darks/snowballs.reg), each flat across its circle, so not
persistence. Also: a snowball at (y, x) (1862, 1966) seems to persist
into integration 2, and (702, 1984) and (414, 1973) through all groups
and integrations.

Cause: a common-mode step of -8.9 DN in every active pixel during the
readout of group 59 of integration 1. The top half of the frame steps in
the difference of groups 58-59, the bottom half in 59-60, the same in all
channels. The reference pixels do not show it, so SIRS cannot remove it;
it is not in the dark reference (0617), since integration 2 has no step.
Integration 1's dark rate is -0.0062 DN/s, integration 2's +0.0033. A
snowball flag splits a pixel's ramp, the split drops the step from that
pixel's fit, and split segments get less weight in the exposure rate
(least-squares weights go as n(n^2 - 1)). A pixel split in integration
1 loses the step and comes out high. One split in integration 2 has no
step to lose, but its integration 2 gets less weight, so its rate leans
toward integration 1's lower one. Predicted offsets of a split pixel
against an unsplit one, from the median ramps of the two integrations:
+0.0009 to +0.0044 DN/s for a split in integration 1, -0.0010 to -0.0028
in integration 2. Observed in James's circles: +0.0019 to +0.0041 and
-0.0013 to -0.0032. So it is neither the non-linearity nor the fit's
weighting as such, but the step.

The persistent snowballs: (702, 1984) and (414, 1973) are static clusters
of about 400 pixels flagged WARM and DO_NOT_USE in the mask reference,
with no JUMP or SATURATED flag in any group. (1862, 1966) is a warm
patch, high in both integrations (+0.0039 and +0.0034 DN/s) and partly
WARM in the mask.

Does the stream clean repair the step? First 75 groups of integration 1,
James's jump settings at 7 sigma with SnowblindStep and JumpPlusStep,
then the clean. Measured: the median group-to-group change of unflagged
pixels at 58-59 and 59-60, and the slope of ramps split at group 60
minus the unsplit slope (the snowball bias):

| clean | step at 58-59 / 59-60 (DN) | split minus unsplit (DN/s) |
|---|---|---|
| none | -3.45 / -5.06 | +0.00865 |
| background None (darks) | +0.02 / +0.01 | -0.00001 |
| integration_model (sky imaging) | -0.11 / -0.13 | -0.00001 |
| median background | -3.50 / -4.97 | +0.00857 |

The clean removes what differs between readout lines and keeps, as
background, whatever its background model holds. With no background
the step goes entirely. With integration_model the model is one box
model per integration made from the least-squares slope, so it cannot
follow a sudden change: the step lands in what the filter removes, and
every difference comes out at the model's level (about -0.13 DN per
group, all differences alike). The bias between split and unsplit pixels
is gone; what is left is the step's share of the least-squares slope,
the same in every pixel, so a uniform rate offset of about -0.012 DN/s
for this 75-group integration that the sky subtraction takes later. With
the median background (not used now) each difference's median takes in
its share of the step as sky, so the full step stays in every pixel and
the snowball bias with it. An offset confined to part of the frame (some
channels or rows) would be removed except for the share the median takes
in. Scripts: scratchpad step_test.py, step_test_median.py and
step_test_intmodel.py of the session (not kept in the repo).

How common such steps are is unknown. A scan of other darks for the
median group-to-group change of unflagged pixels, ordered by readout
time, would show them cheaply.

### 2026-10-28 (evening): the outlier count redone

The count of the first 2026-10-28 entry left out the pixels each run
flagged and judged each pixel by its own ERR, so a run that flags more
had fewer pixels counted and could hide a pixel it flagged but did not
fix. Redone (`common_outliers` in scripts/exp_jump_order.py, results in
results/2026-10-28_jump_order_recount): every run is judged on the same
pixels, the background pixels the 7 sigma jump step left without any
flag (3.2 to 3.9 million per exposure), with the same noise scale, the
ERR of the run with the 7 sigma jump step and the fit at 4.5. Pixels a
run flagged stay in. z is the rate minus the median of its 5x5
neighbourhood over that scale. Pixel by pixel between two runs: an
outlier (|z| > 5) in one run with |z| < 3 in the other is fixed by the
other; the gap keeps pixels near 5 that move by a little from counting.
Two runs added: the fit's search at 5 sigma, and in effect off (1000
sigma). New `--fits-only` option: no clean without flags, no clean
comparisons, no sweep. It was needed because the machine was paging (8.7
GB of swap used, mostly by other apps). NIRISS 4681 took 81 minutes
instead of 10, and the NRCA1 dark was stopped after 41 minutes and rerun
in 9 with the option.

Outliers fixed / new, each comparison against the run named second
(the run with a 5 sigma jump step fits at 4.5; the second JumpStep at 4
sigma runs after the 7 sigma one and the clean):

| exposure | search 4.5 vs off | search 5 vs off | search 4 vs 4.5 | jump step 5 vs 7 | second jump step 4 vs none | zeroframe off vs on |
|---|---|---|---|---|---|---|
| CEERS NRCALONG, 9 groups | 548 / 11 | 543 / 8 | 1 / 7 | 3 / 9 | 7 / 18 | 3 / 5 |
| CEERS NRCA3, 5 groups | 703 / 215 | 673 / 140 | 89 / 411 | 84 / 1888 | 152 / 5621 | 101 / 108 |
| 1074 NRCA1, 5 groups | 779 / 3 | 777 / 3 | 20 / 96 | 13 / 31 | 38 / 6 | 0 / 0 |
| NIRISS 3383, 8 groups | 744 / 61 | 737 / 54 | 13 / 63 | 4 / 10 | 11 / 66 | 0 / 0 |
| NIRISS 4681, 20 groups | 1229 / 29 | 1225 / 23 | 12 / 71 | 2 / 2 | 4 / 7 | 0 / 0 |
| dark NRCA1, 30 groups | 889 / 128 | 875 / 92 | 102 / 1805 | 1 / 13 | 6 / 147 | 0 / 0 |
| dark NIS, 30 groups | 5400 / 389 | 5356 / 324 | 314 / 1013 | 12 / 19 | 8 / 43 | 0 / 0 |
| dark NRCALONG, 30 groups | 6264 / 1371 | 5922 / 902 | 2722 / 7918 | 122 / 228 | 215 / 1111 | 0 / 0 |

Up / down outliers per run, all after the 7 sigma jump step and the
clean unless named:

| exposure | fit search off | 5 | 4.5 | 4 | jump step 5, fit 4.5 | second jump step 4, fit 4.5 |
|---|---|---|---|---|---|---|
| CEERS NRCALONG | 2001 / 157 | 990 / 143 | 955 / 147 | 924 / 154 | 941 / 150 | 924 / 157 |
| CEERS NRCA3 | 1413 / 337 | 754 / 417 | 744 / 480 | 762 / 811 | 855 / 2190 | 806 / 5978 |
| 1074 NRCA1 | 7886 / 45 | 7125 / 13 | 7120 / 13 | 7133 / 74 | 7124 / 29 | 7071 / 18 |
| NIRISS 3383 | 3153 / 438 | 2428 / 427 | 2418 / 435 | 2429 / 477 | 2410 / 445 | 2400 / 518 |
| NIRISS 4681 | 3156 / 395 | 1920 / 395 | 1918 / 400 | 1950 / 441 | 1917 / 402 | 1918 / 406 |
| dark NRCA1 | 732 / 708 | 87 / 573 | 87 / 605 | 902 / 1550 | 91 / 615 | 158 / 676 |
| dark NIS | 5689 / 1117 | 537 / 1201 | 523 / 1241 | 913 / 1631 | 521 / 1249 | 549 / 1250 |
| dark NRCALONG | 4093 / 15806 | 1316 / 13797 | 1276 / 14171 | 5519 / 16016 | 1410 / 14149 | 1993 / 14661 |

What it shows:

1. The fit's own search is the main gain after a 7 sigma jump step: it
   fixes 550 to 1230 outliers per sky exposure and 900 to 6300 per dark
   (cosmic rays the jump step left), and makes new ones: a third as many
   on CEERS NRCA3, a fifth on the NRCALONG dark, a few percent elsewhere.
2. The search at 5 sigma does as well as at 4.5: it fixes slightly fewer
   and makes fewer new ones. Net (fixed minus new) it is within 2 on the
   four other sky exposures, about 20 ahead on the NRCA1 and NIS darks,
   and ahead on CEERS NRCA3 (533 vs 488) and the NRCALONG dark (5020 vs
   4893).
3. The search at 4 sigma is worse on every exposure: more new outliers
   than fixed, up to 102 / 1805 on the NRCA1 dark. A false jump splits a
   ramp into two segments with one slope, and where read noise dominates
   (darks, faint sky) a split in the middle nearly doubles the pixel's
   noise: on the dark pixels the 4 sigma fit newly flags, its ERR is 1.5
   to 1.9 times larger (median 1.54 on NRCA1) and the rate moves by a
   median 4.5 of the reference ERR. The pixel's own ERR grows with it,
   which is why the first count could not see this. A combination
   weighted by ERR would soften the cost, but those pixels are noisier.
4. The 7 sigma jump step is as good as 5 sigma or better everywhere. On
   CEERS NRCA3 (4 differences), 5 sigma makes 1888 new outliers, almost
   all downward in pixels it flagged: on short frame-averaged ramps the
   extra flags are upward noise, and dropping an upward difference pulls
   the rate down.
5. A second JumpStep at 4 sigma after the clean is worse everywhere but
   1074 (bright stars, 38 / 6): 152 / 5621 on CEERS NRCA3, 215 / 1111 on
   the NRCALONG dark.
6. Leaving out the zeroframe makes no difference (101 / 108 on CEERS
   NRCA3, 3 / 5 on CEERS NRCALONG).

So the snowblind_detector1 defaults stand: JumpStep at 7 sigma, the
stream clean, the fit's search at 4.5. Raising the search to 5 would be
at least as good on this set. It is not changed: the gain is small and
higher values (5.5, 6) are untested. Caveats: the count compares
neighbours on the final rate, so a pixel the 7 sigma jump step flagged is
not judged at all. It also charges extra noise as failure at the fixed
scale, which is the cost a single-exposure user sees.

### 2026-10-31: zeroframe pixels that the superbias subtraction sets to 0

The full NEXUS reduction (programs 5105 and 9263) showed up to 1 percent
of the pixels of each rate product flagged SATURATED at random on the
sky. On NRCA1 jw05105001001_02101_00001 (F090W, SHALLOW4) the rate DQ had
37,846 SATURATED pixels, 34,245 of them isolated single pixels at sky
level.

The zeroframe has no DQ array, so the pipeline marks an unusable
zeroframe pixel by setting it to 0 (the saturation step does this), and
the jwst superbias step keeps the zeros it is given (`bias_sub.py` finds
them before subtracting). A pixel whose raw zeroframe equals the
superbias exactly also comes out at 0, though: the raw data are whole DN
and 52 percent of the values of the NRCA1 superbias (reference 0289) are
whole numbers, which made 35,308 new zeros among the science pixels of
that exposure, against 1,925 real ones. The later steps take the new
zeros for unusable. Refpix restores zeros after its correction, so they
miss a correction of -4.8 DN (median; rms 9.8 DN); linearity and the
stream clean leave them at 0.

Every NIRCam superbias reference delivered in 2025 and 2026 (0289 and
0363 to 0454, all ten detectors, made from the cycle 2 to 4 darks) holds
only whole and half-integer values, half of each; the January 2024
delivery (0220 to 0246) is continuous and made no new zeros. Over the
whole NEXUS set (6,470 integrations in 6,390 files from September 2024
to August 2026, all ten detectors, twenty superbias references; table
`5105_9263_nexus/zeroframe_zeros.ecsv` on the NEXUS disk) the new zeros
were 0.46 percent of the pixels, per-integration median 0.51 percent,
from 0 to 1.06 percent. The whole-number half of the superbias times the
density of raw zeroframe minus superbias at 0 predicts each
integration's fraction to 2 percent (median ratio 1.002), so it falls as
the bias drifts away from the superbias: NRCA3 has sat 50 to 100 DN
below its superbias since mid-2025 and has 0.01 percent, and the bias of
NRCA1, NRCALONG, NRCB1 to NRCB4 and NRCBLONG moved by 27 to 140 DN
between the February and March 2026 visits (same references), which
took NRCB4 from 0.66 percent to none.

What the zeros did to the fits, on the NRCA1 exposure:

- The reset fit gave a zero zeroframe SATURATED in the zeroframe plane
  (change 3 of `likely_fit_reset.py`), and that plane reaches the rate
  DQ. It also dropped their zeroframe and the reset point and took a zero
  first frame out of group 1, which raised their rates by 0.0046 DN/s
  (0.3 of the rate uncertainty) with an ERR 7 percent higher than their
  neighbours'.
- Stock OLS_C (the default; NIRCam has no CRDS ramp-fit parameter
  reference) uses the zeroframe only for pixels without a good group and
  skips zeros: 11 pixels lose a rate.
- Stock stcal LIKELY uses the 0 as data: rate error rms 0.39 of ERR,
  tails to 8.6 of ERR, and a false JUMP_DET in 10.5 percent of them.

Fix: `snowblind/superbias.py` `SuperBiasStep`, a subclass of the jwst
step in the `superbias` slot of `snowblind_detector1`, notes the zeros it
is given and sets any other zeroframe pixel left at 0 to the smallest
normal float32 (1.2e-38 DN), which is 0 DN to every later step but not
the marker. The reset fit now gives a zero zeroframe DO_NOT_USE in that
plane and fits those pixels again without the zeroframe (change 7). On
the same exposure only the 1,925 real zeros remain, SATURATED covers
0.05 percent of the pixels instead of 0.90 (all in blocks of the
saturation step's growth), the ERR ratio at the former zeros is 0.999
(was 1.069), and the rate bias is gone. The 6,424 NEXUS rate files of the
2026-10-30 run carry the old flags and need Detector1 again before
persistence flagging. A jwst issue on the superbias step was filed.

### 2026-11-01: the field results moved here from docs/

`docs/flicker_and_reset.rst` and the README are now written for users of
any JWST dataset: what the steps do, how to run them, how to choose the
settings, how they work and where they are limited, with no program or
field names. The write-up as it stood in `docs/` at commit 62f6cbe is
kept below, converted to Markdown, so that the field-by-field
measurements and their figures stay on record: the calibration darks
(programs 8758, 9022 and 9279), Omega Cen (6154), CEERS (1345), NEXUS
(5105 and 9263), the NIRISS fields and WFSS exposures, the NIRCam GRISMC
exposures (5398 and P330-E from 1076), and the 1/f workshop set. The
figures are in `docs/figures/`; the docs now use only the dark and
injection ones (channel_spectra, dark_rate_images, noise_vs_groups,
flux_bias, first_read). Section numbers in the text below ("section 3",
"section 4") are those of the old page: why difference cleaning adds
noise, and the faint-source section with the scene mask.

Draft for discussion with the jwst developers. Measurements made with jwst
e15d3e779, stcal 1.19.0 and the snowblind working tree on 2026-10-20 and 21.
All numbers come from the dated entries in
`PLAN-1f-cleaning-and-reset-anchored-rampfit.md`, section D; the scripts
under `scripts/` regenerate them.

#### Summary

Two steps were built and measured on real NIRCam data:

- `StreamCleanStep` removes 1/f noise from group differences by filtering
  each output channel's readout-ordered sample stream with a Wiener gain
  derived from the exposure's own masked background. It reuses the scene
  mask and background machinery of `jwst.clean_flicker_noise`.
- `ResetRampFitStep` runs the stcal likelihood fitter with the
  superbias-subtracted reset level prepended as an extra resultant, so that
  pixels with a single unsaturated group get a rate.

Four results matter for the pipeline:

1. Cleaning group differences and rebuilding the groups turns any part of
   the noise estimate that is not a fixed linear function of the data,
   applied identically to every difference, into a random walk along the
   ramp. The true 1/f offsets do no such thing: the offset of group j
   enters the differences j and j+1 with opposite signs, so in the sum of
   differences that makes a group only the first and last offsets survive.
   The slope fit therefore amplifies the walk about 27 times in variance
   relative to the true offsets. On a NIRCam dark the jwst median step and
   every other nonlinear variant raise the rate noise above SIRS alone. A
   strictly linear clean lowers it.
2. The Wiener filter with a cutoff at periods below 3 ms removes the
   row banding and all slower 1/f power from the group differences, lowers
   the pixel and aperture scatter of the rate image below SIRS alone at
   every scale, and with a NoiseChisel-style scene mask biases the faintest
   injected sources by 0.3 percent, the same as the row median. Without
   the cutoff it gains a further 11 percent in pixel rate scatter and loses
   12 percent of the flux of the faintest sources.
3. The reset-level scatter is 16.8 DN on NRCA1 (36 e, the size of kTC
   noise), 17.6 DN on NRCALONG and 25 DN on NIRISS. With it as a prior,
   every pixel that saturates after group 1 on a bright field gets a rate,
   and on NRCA1 the recovered cores of 30 saturated stars agree with the
   curve of growth of 223 unsaturated stars to 2 +- 1 percent.
4. The first read holds about 9 percent less signal than the slope
   predicts, plus a few DN, on NRCA1. The fitter models this as a delayed
   reset. Without the correction the reset-anchored fit shifts the median
   rate of all pixels by 1 percent.

#### Running the steps

The defaults are set for sky-limited imaging, which is what the
measurements below cover: the background of each group difference is a
scalar median measured on the flat-fielded difference, sources come from
the two-tier scene mask with the column spikes of saturated stars
followed, and the reset fit takes the detector's reset scatter and
first-read timing from a file packaged with snowblind, one per detector (ten NIRCam
detectors and NIRISS, measured as described in the reset section) with
the reset offset anchored to each exposure's own sky. The
`snowblind_detector1` pipeline is `calwebb_detector1` with its steps,
parameters and CRDS parameter references, except that `StreamCleanStep`
runs as a post-hook of the jump step (the pipeline's own
`clean_flicker_noise` step is skipped, since some parameter references
turn it on) and `ResetRampFitStep` takes the `ramp_fit` slot, so one
run gives the rate products of the cleaned ramp and the reset-anchored
fit:

```
strun snowblind_detector1 jw..._uncal.fits --steps.refpix.refpix_algorithm=sirs
```

The jump step runs at 7 sigma (the three- and four-group thresholds
too), flags the four neighbours of jumps above 20 sigma, and leaves the
snowball expansion and the flagging after jumps to
`SnowblindStep(min_radius=3, after_jumps=5)` and `JumpPlusStep` as its
first post-hooks: at 7 sigma it flags what would disturb the stream
clean, and the reset fit's own jump search (4.5 sigma) finds the fainter
jumps on the cleaned ramp. A clean with no jump flags leaves more
striping in the readout-line means of the rate than one after a 7 sigma
jump step (10 to 50 percent more on sky fields, 13 to 17 times on darks),
and 7 sigma cleans as well as 5 sigma. These
settings take precedence over the CRDS parameter references and give way
to the caller's (`--steps.jump.rejection_threshold=5`);
`--steps.jump.expand_large_events=True` gives back the jwst snowball
flagging in place of the two hooks.

Either substitution switches off (`--stream_clean=False`,
`--reset_fit=False`). The stock pipeline does the same with hooks
alone: the stream clean as a post-hook of the jump step
(`--steps.jump.post_hooks="snowblind.StreamCleanStep",` with
`--steps.clean_flicker_noise.skip=True`) and the reset fit as a
post-hook of a skipped `ramp_fit` in its `as_hook` form, which
returns the rate image alone, since stpipe runs a step's post-hooks even
when the step is skipped and takes a hook's return value as the data
that goes on; in Python a hook can be the step class itself.

A snowblind step run as a hook inside a jwst pipeline sends its log
messages to the pipeline's handlers, so they appear in the pipeline log
like jwst's own. The scene mask also leaves out every pixel the DQ marks
DO_NOT_USE, in the mask reference, the flat reference and the group
flags, and follows the spikes of saturated stars along both image
axes, since NIRCam reads out along science rows and NIRISS along
science columns. On darks `background_method=None` is right (the
median removes the dark current's own level); for extended emission
larger than a few hundred pixels `integration_model` keeps it out of
the noise estimate at the cost of the slowest 1/f; for a dispersed
(grism) exposure the step defaults to `background_method='crds_background'`,
which scales the CRDS background reference once to the exposure and
subtracts it from every difference (see the WFSS section).

#### Readout geometry

In the detector frame rows are the slow axis, the four outputs are
512-column blocks and adjacent outputs read in opposite directions. Each
output reads one pixel every 10 microseconds (the pixel time, the unit of
the stream), and 12 pixel times of overhead separate the end of one row
from the start of the next, so a row takes 524 pixel times, 5.24 ms. Times
below are given in milliseconds, with the pixel count where it helps. The
direction of channel 0 was checked on darks by comparing both directions
through the correlation of samples that straddle a row boundary at lags of
0.13 to 1 ms (13 to 100 pixel times), after removing per-row means. Channel
0 reads toward increasing detector column and the
directions alternate, on NRCA1 (2.5 to 6.7 sigma), NRCB3 (7 to 10 sigma),
NRCALONG (three channels at 2.4 to 5.6 sigma, one undetermined) and NIRISS
(3 to 11 sigma, the transposed geometry).

After the per-row mean is removed, the within-row autocorrelation of a
RAPID dark's group differences is 0.011 to 0.037 at lag 2, 0.004 to 0.007
at lags 3 to 8, 0.002 at lag 13 and zero beyond lag 100. Inter-pixel
capacitance gives 0.16 to 0.32 at lag 1. Most of the 1/f power is at
periods of a row and longer.

#### Noise spectra

Per channel on NRCA1 the stream spectrum of a group difference fits a power
law with slope 0.89 to 0.91 above a white level of 32 to 37 DN^2 per bin,
with the power law crossing the white level at periods of 1.0 to 1.3 ms,
and a second rise toward the shortest periods (114 to 184 at a period of
0.1 ms against 58 to 67 at 0.5 ms) that is not 1/f.

Figure `docs/figures/channel_spectra.png`: What the cleaner removes, channel 0 of the NRCA1 dark. Each curve is the noise power in a group difference against the period along the readout stream, on log axes, median over the 19 differences and over logarithmic period bins; values below 0.1 are drawn at 0.1. Before cleaning (blue) the power is flat at the read-noise level of 41 DN^2 at periods below 0.3 ms and rises as 1/f toward long periods, by a factor 30 at one row (5.24 ms) and a factor 1000 at a second. After the clean (orange) the power at every period longer than the 3 ms cutoff is gone, two to four orders of magnitude below the read-noise level: the row banding and everything slower is removed, together with the read noise's own share of those periods, which is what a Wiener gain near one does. Below the cutoff the two curves coincide by construction. Beyond 100 ms the estimate is the poorly measured end of a 0.66 s autocorrelation and the binning hides the pickup lines at 2, 4, 5 and 10 pixel periods that the raw spectrum shows.

#### Why difference cleaning can add noise

Cleaned group j is input group j minus the sum of the noise estimates of
differences 1 to j. If every estimate were a mean over the same pixels,
that sum would collapse to the mean of group j minus the mean of group 0,
as the true offsets do, and nothing would accumulate. For the per-row
median the part of the estimate that does not cancel this way is the
difference between a median and a mean, 0.25 DN per row per difference on
this dark, which predicts 0.0047 DN/s added in quadrature to the rate; the
measured change is 0.0046. A per-difference
Background2D model adds a larger such term and per-difference sigma
clipping a third, because the mask and so the filter change from one
difference to the next.

Table 1, NRCA1 dark jw08758003001, first integration, 20 groups, rate
images from OLS_C. Robust sigmas in DN/s over 98.8 percent of the pixels;
the row-mean sigma is the scatter of per-row means of the rate image, and
S is the mean stream spectrum in the band (white level 44).

| Variant | rate sigma | row-mean sigma | slow band S | row to 50 S | ap 9 | ap 33 | ap 101 |
|---|---|---|---|---|---|---|---|
| white prediction | 0.0188 | 0.0008 | 0 | 0 | 0.0023 | 0.0006 | 0.0002 |
| SIRS, no cleaning | 0.0205 | 0.0053 | 1204 | 121 | 0.0052 | 0.0037 | 0.0027 |
| SIRS + jwst clean_flicker median | 0.0213 | 0.0079 | 403 | 122 | 0.0058 | 0.0046 | 0.0034 |
| SIRS + row median, clipped | 0.0217 | 0.0090 | 397 | 123 | 0.0061 | 0.0048 | 0.0038 |
| SIRS + Wiener, clipped | 0.0209 | 0.0060 | 213 | 114 | 0.0059 | 0.0049 | 0.0042 |
| SIRS + row median, linear, box 32 | 0.0210 | 0.0073 | 222 | 123 | 0.0045 | 0.0027 | 0.0022 |
| SIRS + Wiener, linear, box 32 | 0.0197 | 0.0026 | 2.5 | 111 | 0.0039 | 0.0028 | 0.0024 |
| SIRS + Wiener, linear, box 128 | 0.0196 | 0.0018 | 2.4 | 111 | 0.0034 | 0.0019 | 0.0015 |
| SIRS + Wiener, linear, box 256 | 0.0195 | 0.0012 | 2.4 | 111 | 0.0031 | 0.0015 | 0.0012 |
| SIRS + Wiener, linear, median bkg | 0.0196 | 0.0016 | 7.3 | 112 | 0.0030 | 0.0011 | 0.0006 |

The clipped variants, the per-difference model background and the boxcar
estimator were removed from the step after these measurements; the rows
stay as the record of why.

"Linear" means one background model per integration, fitted to the
least-squares slope of the ramp and subtracted from every difference, no
per-difference clipping and one iteration; all Wiener rows use the 3 ms
cutoff (300 pixel times, about half a row) that is now the default. The
clean with the default 32-pixel box is better than SIRS alone at every
scale: pixel scatter down 4 percent, row banding down 2 times, aperture
scatter down 25, 24 and 11 percent at 9, 33 and 101 pixels. Larger boxes
and the scalar median go further on a dark, because there is no sky to
protect.

The rows above were measured before the even-odd column correction was
added. Each amplifier of these detectors gives its even and odd columns a
slightly different offset in every read (0.02 to 0.05 DN per group on
NRCA1, with a different value and sign per amplifier, and the pattern of
one integration predicts only 8 percent of the next), so it is not a fixed
pattern that the superbias removes and it survives into the rate image as
faint vertical structure. The step now measures the offset per amplifier
and per difference from the masked science pixels and subtracts it before
the stream filter. It is one number per amplifier per difference, so it is
linear and does not random-walk. With it on, the box 32 row reads 0.0196
and 0.0023 DN/s for the pixel and row-mean scatter, the aperture scatter is
0.0038, 0.0026 and 0.0022, and the scatter of the column means of the rate
image drops from 0.0020 to 0.0013 DN/s.

What remained after that was a second effect, which the CEERS team's
striping removal (Bagley et al. 2023) also treats: an offset per column
that wanders during the integration. On two integrations of the NRCA1
dark the column means of the cleaned rate scatter 3 times more than white
noise allows, the pattern of one integration predicts only 6 percent of
the next (so it is not dark current, superbias or dark reference
residual), it is constant along the whole column within an integration
(column means from the four row quarters correlate at 0.65 to 0.80, as
high as their own noise permits), white across column index and
independent per amplifier. Per-read column jitter would leave only 0.0004
DN/s in a 20-group rate; the observed excess needs about 0.26 DN of
per-column drift over the 215 s integration. It is not a readout-stream
process, so the stream filter leaves it alone: a column offset is a
once-per-row periodic signal, white across columns, inside the 3 ms band
the filter reserves for the scene. The step now removes each column's
masked mean from each difference, per amplifier, after the background
model and before the filter (`column_offsets`, default on). CEERS
subtract a column median from the rate image once; inside the ramp a
median would random-walk, while a mean is a fixed linear operation, and
since the slope is itself a fixed linear combination of the differences
the result equals subtracting the column mean of the rate image and
cannot add noise. Row banding does not bias the column means, because it
adds the same value to every column of a row. On the dark the
column-to-column part of the column means falls from 0.0010 to 0.0002
DN/s (below the white level of 0.0004, as a mean removal should), and
what is left is the background model's structure at scales above 64
columns, which on a dark is large-scale 1/f noise the model has fitted
and the step keeps by design, just as it does for rows. On the injection
the faintest-bin bias is unchanged within its noise (-0.27 +- 2.1 to
-0.20 +- 2.4 percent) and the sky gradient is untouched. The cost on real
data is the same unmasked-wings problem as the row cleaning, spread
along a whole column, and it has not been measured on a dense field.

Where the background model comes from matters. At scales above the box
the filter removes everything from each difference and the model is added
back, so the cleaned rate's structure there is whatever the model holds. A
model fitted to the mean group difference, which is the last group minus
the first over the elapsed time, carries 3.7 times the variance of the
least-squares slope for white noise and more for 1/f; built that way the
clean was worse than SIRS at 33 to 101 pixels (0.0046 and 0.0042 against
0.0037 and 0.0027), and doubling the integrations barely helped. Fitting
the model to the slope removed the penalty. The jwst step's per-difference
model has the same weakness in a different form, one noisy estimate per
difference.

Figure `docs/figures/aperture_scatter.png`: Scatter of aperture means against aperture size on the NRCA1 dark for SIRS alone, the jwst median step, and the linear Wiener clean with a 32-pixel and a 256-pixel background box and with no background, with the white-noise prediction. With no background to protect, which is the right setting on a dark, the clean reaches the white-noise floor at every aperture size (0.0003 DN/s at 101 pixels against 0.0002 predicted); the box model keeps whatever structure it holds above its scale, by design.

On NRCALONG (same dark program, same script, run with the earlier
mean-difference model) the pattern repeats in weaker form: SIRS alone
0.0199 DN/s pixel scatter and 0.0052 row-mean scatter, the jwst median
0.0202 and 0.0062, the Wiener clean with the cutoff 0.0196 and 0.0047 with
the slow band removed entirely; the large-scale numbers of that run carry
the mean-difference penalty described above and are to be regenerated.

#### Faint sources and the sky

Table 2, 1500 Gaussian sources (sigma 1.3 px, 2 to 2000 DN/s) and a sky of
0.3 + 0.2 x/2048 + 0.1 sin(2 pi y/400) DN/s added to the same dark's jump
product. Relative bias of r = 3 px aperture flux against the uncleaned fit
of the same injected data; "slope vs sky" is the fraction of the sky
structure removed.

| Variant | median bias | 2 to 6 | 6 to 20 | 20 to 63 | slope vs sky |
|---|---|---|---|---|---|
| jwst clean_flicker_noise median | -0.0001 |  |  |  | -0.126 |
| row median, linear, box 32, clip mask | -0.0001 | -0.0078 | -0.0006 | -0.0003 | -0.088 |
| row median, linear, box 32, chisel | -0.0000 | -0.0027 | +0.0002 | -0.0001 | -0.076 |
| Wiener, box 32, no cutoff, clip mask | -0.0021 | -0.124 | -0.0245 | -0.0070 | -0.090 |
| Wiener, box 32, 3 ms cutoff, clip mask | -0.0001 | -0.0107 | -0.0016 | -0.0005 | -0.092 |
| Wiener, box 32, 3 ms cutoff, chisel | -0.0000 | -0.0026 | +0.0006 | +0.0002 | -0.077 |
| Wiener, box 128 | -0.0023 |  |  |  | -0.733 |
| Wiener, box 256 | -0.0022 |  |  |  | -0.693 |

(The sigma-clip mask rows come from a step option since removed; the
jwst step in the field figures is that mask's living comparison.)

With the chisel mask the Wiener clean and the row median are equivalent in
photometry down to the faintest injected sources, 0.3 percent at 2 to 6
DN/s and nothing above, and the sky loss drops for both because the
background model is fitted with fewer source pixels in it.

On NRCALONG the same injection gives a median bias of 0.0000 for the jwst
median and the row median and -0.0001 for the Wiener with the cutoff, with
-0.0046 at the faintest bin, and 14, 10 and 10 percent of the sky removed.

A low-pass filter removes part of any source the scene mask misses, so
the mask matters. The jwst step's mask is a per-pixel sigma clip of the
draft rate image: a pixel is a source if it sits more than 2 sigma above
the background, with no segmentation and no growth into the wings, so it
misses sources below about 20 DN/s here and the wings of every bright
source. `StreamCleanStep` now builds its mask along the lines of
NoiseChisel instead (Akhlaghi and Ichikawa 2015, with Gnuastro's default
parameters): the draft rate is sky-subtracted with a tiled sky model and convolved with a
2-pixel Gaussian, pixels above the median of the result are candidates
(NoiseChisel uses the 0.3 quantile; on these undersampled images the
convolved noise then percolates into clusters that survive the erosion,
so the median is used), two erosions and one opening remove what noise
alone produces, each surviving blob must exceed
the 99th percentile of the signal-to-noise of the pseudo-detections that
the same procedure finds in the sign-flipped image, and the survivors are
grown back over the eroded border and into pixels above the 0.9 quantile.
On synthetic images it masks 99.9 percent of a bright source's flux
where the clip masks the core only, still catches a source with a 3-sigma
peak, and marks 1.5 percent of a pure-noise image with a sky gradient and
a sinusoid on it.
A blanket dilation of the clip mask by 3 pixels is not a substitute: it
brings the faintest-bin bias from -1.1 to -0.4 percent but doubles the sky
structure removed, from 9 to 20 percent, because the background model
loses too many pixels.
The cutoff makes the filter blind to compact sources and costs the pixel
rate gain, which lived in the same band. Any background model smoother
than the sky structure lets the cleaner remove that structure: within one
exposure, 1/f banding and sky at periods of a few hundred rows are
degenerate and the model scale picks the compromise.

Figure `docs/figures/flux_bias.png`: Relative photometric bias of injected sources against their flux, all with the 3 ms cutoff. With the jwst-style clip mask the Wiener clean loses 1 percent of the faintest sources; the NoiseChisel-style mask brings both the Wiener clean and the row median to 0.3 percent. The Wiener clean without the cutoff, not plotted, loses 12 percent there.

#### Reset-anchored fitting

The pedestal of the OLS_C fit on dark integrations gives the reset scatter,
after removing the first group's own read noise:

| Detector | data | pedestal scatter | read noise | sigma_b |
|---|---|---|---|---|
| NRCA1 | 9022 BRIGHT1, 10 integrations | 17.6 DN | 5.2 DN | 16.8 DN |
| NRCALONG | 9022 BRIGHT1, 10 integrations | 18.3 DN | 4.8 DN | 17.6 DN |
| NIRISS | 9279 NISRAPID, 10 integrations | 25.5 DN | 7.4 DN | 25.0 DN |

The NIRISS value comes from two exposures of five integrations, observed
months apart and processed with different superbias references, which
the calibration expresses against one of them before combining; it
includes any change of the reset level between the exposures, which the
sky-anchored reset offset absorbs on a science exposure. One exposure
alone gives 22.0 DN. On NRCA1 the value is 36 e at a gain of 2.05, the size of kTC noise;
the per-channel medians agree to 0.5 percent, and the per-pixel pedestal mean
has a scatter of 13.9 DN of which 12.7 DN is systematic superbias error.
The prior's weight: in a BRIGHT1 5-group ramp the reset difference carries
about 1/23 of the weight of an ordinary difference.

On the program 6154 NRCA1 exposure (BRIGHT1, 5 groups), 3530 pixels
saturate from group 2 on; none gets a rate from OLS_C or LIKELY, all get
one from the reset fit. On the NRCALONG exposure of the same visit 11034 of
11036 such pixels get one. Pixels with exactly two usable differences need a
guard: with two differences the one-omit jump search is degenerate, and
without the guard it dropped the real difference.

The first read, measured as d_0 / a_rest on 242k pixels with a first
difference of 50 to 2000 DN, holds R1 = b + k a t1 with k = 0.904 and
b = -3.5 DN detector-wide on NRCA1 (per channel k = 0.897 to 0.914, b =
-2.2 to -6.0 DN) and k = 0.943, b = -2.6 DN on NRCALONG (855k pixels). The
fitter models k as a reset delayed by (1 - k) t1, which leaves
every later interval unchanged; scaling the first group's read time instead
also stretches the first ordinary interval and biased the median rate by
0.9 percent. The chi-square of the reset difference is 1.5 over all clean
pixels and 2.0 over bright ones; the all-pixel excess matches the
superbias error map, of which the dark-epoch map recovers part, and the
bright excess is not channel timing.

Figure `docs/figures/first_read.png`: Ratio of the reset difference to the rate from the other differences against the first group difference, NRCA1, with the two-parameter model. The form is the same on every detector measured; only the numbers move: k from 0.88 to 0.95 at a first-group time of 10.7 s (a delay of 0.5 to 1.3 s; the ten NIRCam detectors 0.88 to 0.94, NIRISS 0.95), the four channels of a detector within 0.03 of each other, and the offset b between -9 and +4 DN. The values live in the packaged calibration of each detector.

Curve-of-growth test on NRCA1: 30 isolated stars with cores saturating
from group 2, fully recovered by the reset fit, against the core-to-ring
ratio of 223 isolated unsaturated stars (10.22, known to 0.6 percent).
Recovered core flux over prediction minus one: +0.024 (n 22) for peaks of
2000 to 4000 DN/s, -0.066 (n 7) for 4000 to 8000, +0.016 +- 0.013
overall. On NRCALONG the field is too crowded at the long-wave PSF for a
clean reference sample and the test is inconclusive at the 5 to 10 percent
level.

The offset b must be taken from the sky pixels, which set the zero point:
with the bright-pixel regression intercept the NRCALONG median rate was
0.5 percent above LIKELY, with the sky value it is within 0.2 percent
(NRCA1 within 0.1).

The first read needs cleaning too. `stream_clean` works on group
differences, and for OLS_C the first read's own banding is common to
every rebuilt group and drops out of the slope; the reset-anchored fit
uses group 0 minus the prior as a difference in its own right, with
about a tenth of a group's weight, so that banding enters the rate. On
the Omega Cen NRCA1 exposure it raised the row-mean scatter of the sky
pixels from 0.0023 DN/s (OLS_C) to 0.0062, with the map prior and with
a scalar prior alike, so the map was not the source. The map does carry
row structure, 3.5 DN rms on NRCA1 against 0.3 from white noise, but
two halves of the calibration dark correlate at 0.68 in their row
means: about 3.2 DN of it is a repeatable difference between this
epoch's reset level and the superbias reference, present in every first
read as well, and only 1.5 DN is averaged-down 1/f. The step therefore
now cleans the first read by default, and when given the calibration
file it cleans it against the reset prior itself (the pedestal map with
its row, even-odd and column structure removed, plus the channel
offsets; the same map the reset step anchors to), so the filter sees
only the 1/f of the reset frame and the first read. With that the reset
fit's row-mean scatter returns to the OLS_C value (0.0024) with the
per-pixel prior intact, and bright-pixel rates agree with OLS_C to 0.1
percent. Cleaning the first read against zero with the raw map instead
leaves 0.0044, because the map's real row pattern then enters with the
opposite sign.

The map is a residual relative to a superbias. The darks that calibrate
it have had their CRDS superbias subtracted, so the map is the mean
reset level relative to that file, and it is the right prior only for a
science exposure that had the same file subtracted. The calibration now
records the superbias name the darks were processed with, the reset
step declares `superbias` among its reference types so that CRDS
delivers the best reference for the data, and both steps read the name
the superbias step wrote into the data. When the two names differ the
prior is corrected by the difference of the two reference images; when
the best reference differs from the one the data had subtracted the
step says so. For the Omega Cen exposures and the calibration darks the
context served the same files (NRCA1 0454, NRCALONG 0443) for
observations a year apart, so no correction applied here. The
repeatable 3 DN row pattern in the map is then the drift of the reset
level since that superbias epoch; a superbias nearer in time would
shrink it, and the bookkeeping makes that automatic.

#### Comparison with the JWST documentation on 1/f noise

The known-issues page on 1/f noise
(https://jwst-docs.stsci.edu/known-issues/1-f-noise) describes the effect
as striping along the fast-read direction from reference-voltage
fluctuations of the readout amplifiers, with a different pattern per
amplifier, and presents `clean_flicker_noise` as the mitigation, off by
default. Two of its figures have direct counterparts here.

Its Figure 1 shows a NIRCam dark before and after `clean_flicker_noise`.
The two figures below are the same comparison on the NRCA1 dark of this
work, with the median reference-pixel correction (the refpix step's
own default, and the pipeline default before SIRS), SIRS (which the
current NIRCam Detector1 parameter reference selects), SIRS plus the
jwst step (median per channel), and SIRS plus `stream_clean`. Only the
cleaning differs between the panels: every one is fitted with OLS_C over
20 groups and shown at the same stretch, and both cleaning steps run with
no background subtraction, which is the right setting for a dark. All
Detector1 steps up to and including jump run with the pipeline's CRDS
parameters, except that the jump step's snowball expansion runs with a
minimum jump area of 8 pixels and an expansion factor of 2.5 on NRCA1
and 3 on NIRISS in place of the reference values (area 5 and factor 2
for NIRCam, 15 and 1.75 for NIRISS): the halo of a snowball falls off
slowly and the reference expansion leaves 0.01 to 0.03 DN/s of it in
the rate image around the largest events, visible at the stretch of
these figures; the larger factors remove about half of that for 15 to
40 percent more flagged pixels, which OLS_C still fits from the other
segments, and a factor of 4 gains nothing further. The remaining ring around
the largest events in every panel is the tail of that halo, the same in
all four since they share the jump product. A
background model exists to protect the sky from the cleaning, and a dark
has none; a smooth model fitted across the whole detector also ignores
the real discontinuities in the 1/f noise at the amplifier boundaries and
hands part of that noise back (see below). The first figure is the
central 512 by 512 pixels, rows and columns 768 to 1279, which straddle
the boundary between the two middle amplifiers; the second is the full
frame. The step's `fft` method (NSClean) is not shown because the step
only allows it for NIRSpec exposure types and skips a NIRCam image; the
median method is what the pipeline offers NIRCam.

Figure `docs/figures/dark_rate_images.png`: The central 512 by 512 pixels of the NRCA1 dark rate image, one image pixel per detector pixel, stretch +- 2.5 pixel sigma about the median; unusable pixels are drawn at the median and hot pixels clip to the top of the scale. The row-mean and column-mean scatter in each title are over this region. SIRS alone reduces the row-mean scatter of the median correction by a fifth. The jwst step removes the banding from every group difference, yet the rate image keeps as much of it as SIRS (0.0051 against 0.0048 DN/s): over 20 groups the non-cancelling part of the per-difference median accumulates into as much row structure as it removed, which is the random walk of section 3; with the step's default background model it is 0.0069. The snowblind clean leaves the read noise (0.0008 in row means, 0.0010 in column means); with its default background model it is 0.0016 and 0.0023, the difference being 1/f noise above the box scale that the model hands back. The even-odd and per-column offset corrections take the column-to-column structure out; the dark column visible left of centre in the SIRS and jwst panels is a per-integration column offset, not a detector feature.

Figure `docs/figures/dark_rate_images_full.jpg`: The same four rate images over the full 2048 by 2048 frame, binned 2 by 2 for display, same stretch; the scatters in the titles are over the whole unbinned science area. Over the full frame the jwst step reaches 0.0036 DN/s in row-mean scatter against 0.0047 for SIRS, so the central cutout is a harder region for it than the average; with its default per-difference background model it is 0.0047, and that model shows as rectangular patches, each difference's `Background2D` being a different noisy tiling of the dark that accumulates along the ramp like the medians. The snowblind clean with no background reaches the white-noise floor: 0.0004 in row means and 0.0002 in column means, against 0.0004 and 0.0004 expected from the pixel scatter alone, and 0.0003 in 101-pixel block means against 0.0002. With its default background model the same clean gives 0.0022 and 0.0010, and smooth gradients remain across the frame. Those gradients are 1/f noise at scales above the background box: the model, fitted to this integration's own rate image, contains them and the step reproduces them. Measured in 64 by 64 pixel block means on two integrations, that large-scale component has 0.0024 DN/s of scatter against 0.0003 from white noise, correlates at 0.88 with the same component of the SIRS rate, where the banding hides it, and at -0.07 with the next integration, so it is not detector structure. On a dark the model has nothing to protect and should be off; on a science exposure it keeps the sky, and separating sky from slow 1/f needs a model with more information than one exposure (see Open items).

The same comparison on a NIRISS dark (program 9279, NISRAPID, first 20
of 120 groups) shows the step on the other readout geometry: on NIRISS
the fast readout axis runs along science columns, so the banding is
vertical and the amplifiers are horizontal bands.

Figure `docs/figures/dark_rate_images_nis.png`: NIRISS dark, central 512 by 512 pixels, same layout and statistics as the NRCA1 figure. Over the full frame the column-mean scatter, which is the banding direction here, goes from 0.0103 DN/s (median refpix) and 0.0083 (SIRS) to 0.0043 with the jwst median and 0.0008 with the stream clean, and the pixel scatter from 0.0297 to 0.0267.

Figure `docs/figures/dark_rate_images_full_nis.jpg`: The same on the full NIRISS frame, binned 2 by 2 for display. The rectangular patches in the first two panels are large-scale structure of this dark that the median reference-pixel correction and SIRS leave in place; with no background to protect, the stream clean removes them with the banding.

Its Figure 4 plots the 1/f noise in NIRCam dark data against the number
of groups in the ramp, with and without `clean_flicker_noise`, and the
caption notes a factor of 10 improvement for small group counts and worse
results for very large group counts. Section 3 is the explanation of that
turnover: the per-difference estimate carries a non-cancelling error that
accumulates along the ramp, and the slope fit weights that accumulation
about 27 times more heavily than the true offsets, with the ratio growing
with the number of groups. For two or three groups there is little to
accumulate and the median wins by a wide margin; for twenty it loses. The
figure below repeats the measurement with the same three variants on this
dark for 3 to 100 groups.

Figure `docs/figures/noise_vs_groups.png`: Row-mean scatter and pixel scatter of the OLS_C rate image against the number of groups fitted, NRCA1 dark, for SIRS alone, SIRS plus the jwst median step (its default configuration) and SIRS plus the stream clean. The jwst step halves the banding at 3 groups, crosses SIRS at about 12 and is 5 times worse at 100 groups, where it also raises the pixel scatter by 55 percent, the turnover the documentation reports. The stream clean stays below SIRS by a factor 1.8 to 2.3 at every length.

Figure `docs/figures/noise_vs_groups_linear.png`: The same measurement as total noise per integration in DN (rate scatter times integration time) on linear axes, as the detector performance pages show it, on the NRCA1 dark and on a NIRISS dark (NISRAPID, one frame per group on both). The stream clean is below SIRS at every ramp length on both detectors, by 11 to 12 percent at 3 groups and 3 to 7 percent at 100; the jwst median is below SIRS for short ramps and above it beyond 20 groups, by 24 percent on NIRISS and 55 on NRCA1 at 100 groups, which is the random walk of per-difference medians along the ramp.

#### A crowded field: Omega Cen

The same comparison on a science exposure, program 6154's Omega Cen
field (NRCA1 F115W and NRCALONG F277W, BRIGHT1, 5 groups, a different
dither from the one the reset timing was calibrated on). The jwst panel
runs `clean_flicker_noise` the way the program's own reduction did:
median method over full rows, flat applied before fitting, science
regions masked, scalar median background. The snowblind panels run
`stream_clean` with the same kind of background, a scalar median, its
first read cleaned against the reset prior, and otherwise its defaults.
To be explicit about the two things the step keeps apart: the
background that is subtracted before filtering and added back is here a
single number per difference (`background_method='median'`), which
was the best choice on this field, ahead of the fitted box model and of
a scene model from the mosaic (both below); the pixels that feed the
noise fit are still chosen by the NoiseChisel-style scene mask of
section 4, which uses its own tiled sky estimate only to find sources
and never subtracts it from the data. All panels start from the same
SIRS-corrected ramp and the first three are fitted with OLS_C; the
fourth is fitted with `reset_rampfit`. The statistics in the titles
are over the background pixels of that scene mask that lie within 5
sigma of the sky, which is where faint sources are found.

Figure `docs/figures/field_rate_images_nrca1.png`: Omega Cen on NRCA1, central 512 by 512 pixels, stretch +- 2.5 sigma of the background pixels about the sky. Over the full frame the row-mean scatter goes from 0.0210 DN/s (SIRS) to 0.0076 with the jwst step and 0.0023 with the stream clean, and the column-mean scatter from 0.0286 to 0.0282 and 0.0028: the row median does nothing for the per-column offsets, the stream clean removes them. The stream clean changes the mean sky by -0.7 percent (jwst +0.4) but its pixel-level changes have long tails (5th and 95th percentiles -0.080 and +0.066 DN/s against -0.028 and +0.032 for the jwst step), visible as short dark streaks along rows next to the brightest stars: the unmasked wings of saturated stars are read as slow 1/f by a linear estimator and subtracted along the row, where a median ignores them. With `reset_rampfit` the row-mean and column-mean scatter are 0.0024 and 0.0028, the OLS_C values, and bright unsaturated pixels agree with OLS_C to 0.1 percent; its benefit is the 1646 saturated cores it fits and OLS_C does not.

Figure `docs/figures/field_rate_images_nrca1_full.jpg`: The same on the full NRCA1 frame, binned 2 by 2 for display.

Figure `docs/figures/field_rate_images_nrcalong.png`: Omega Cen on NRCALONG. Full-frame row-mean scatter 0.0863 (SIRS), 0.0290 (jwst), 0.0063 (stream clean) and 0.0062 (stream clean with `reset_rampfit`), column-mean 0.0999, 0.0999, 0.0093 and 0.0090 DN/s. The sky changes by +1.5 percent with the jwst step and -3.3 percent with the stream clean, the latter driven by the cluster core on the left of the frame, where the mask covers little and the wings dominate; the tails there reach -0.34 DN/s. The reset fit adds 6857 saturated cores.

Figure `docs/figures/field_rate_images_nrcalong_full.jpg`: The same on the full NRCALONG frame, binned 2 by 2 for display.

Two things were learnt on this field. First, the default background
model of `stream_clean`, the 32-pixel-box model fitted to the slope
image, is the wrong choice for a flat-sky star field: it follows the
banding and the wings at scales above the box and the step hands them
back, so with it the stream clean reached only 0.0175 DN/s in row-mean
scatter against 0.0023 with a scalar background. The model's job is to
protect real extended structure, which the injection test has (a sky
sinusoid of 400-row period that a scalar background would remove); a
flat-sky field has none. The guidance is therefore: no background on
darks, a scalar median on flat-sky fields, the box model where there is
extended emission or a strong gradient to keep. Second, the first
version of that model, taken from the jwst helper, was a sigma-clipped
median per box that also discards any box more than 10 percent masked;
on this field it sat 15 to 20 percent below the masked mean and, because
the filter's gain is 1 at zero frequency, the step removed the
difference and lowered the sky by 16 percent on NRCA1 and 34 percent on
NRCALONG. The model is now the plain masked mean per box
(`masked_mean_background`), which is unbiased by construction; on the
injection it preserves the sky gradient five times better and on the
dark it costs 0.0004 DN/s at the 101-pixel scale.

What remains open is the streaks: a linear estimator cannot ignore
unmasked wings, and on a field this crowded the scene mask cannot cover
them all. The `row_median` estimator with the new `anchor_mean`
option (which gives a median-based estimate the masked mean of the
residual, so it cannot shift the sky) has tails 40 percent shorter at
0.0035 DN/s row-mean scatter, still well below the jwst step. A scene
model from the dithered mosaic was tried and does not help on this
field (Open items): the blot's own errors are larger than the noise.

#### A deep field: CEERS

The same comparison on a CEERS exposure (program 1345, NRCALONG F444W,
MEDIUM8, 9 groups of 107 s, observed 2022-06-22): a sparse field of
galaxies on a bright, smooth sky, the common JWST case. Three things
differ from the dark and Omega Cen runs. The images are compared after
flat-fielding, because the long-wave rate image before the flat carries
the detector's response, a 10 percent large-scale pattern that
dominates any row or column statistic. The step applies the flat
before measuring its background and subtracts that background times
the flat from the raw difference, since the background of a sky field
is the sky times the detector response, while the 1/f noise is additive
in raw DN and is filtered there. And the background is the scalar
median: with the flat applied the sky is smooth, and the box models
hand back the slow 1/f for no gain.

Figure `docs/figures/field_rate_images_ceers_nrcalong.png`: CEERS on NRCALONG, flat-fielded, central 512 by 512 pixels, stretch +- 2.5 sigma of the sky pixels about the sky; the fourth panel is the CEERS team's remstriping (Bagley et al. 2023, the iterated-threshold version) run on the same SIRS rate image. Over the full frame the row-mean scatter of the sky pixels goes from 0.0013 DN/s (SIRS) to 0.0006 with the jwst step, 0.0009 with remstriping and 0.0003 with the stream clean, the column-mean scatter from 0.0015 to 0.0015, 0.0008 and 0.0002: the pipeline's row median cannot see the per-column offsets, remstriping's column pass halves them, the stream clean removes them. Remstriping's per-amplifier row medians come from about 300 sky pixels each in one rate image, so their estimation error, 0.0015 DN/s, is the size of the striping they measure; the pattern it subtracts is half noise, which is why it leaves the central cutout unchanged and doubles the per-pixel change tails. The sky changes by -0.06 percent, its large-scale structure by 0.2 percent, and the per-pixel change lies within -0.005 and +0.003 DN/s. On a 9-group ramp of 107 s groups the 1/f left by SIRS is small to begin with, so the gain is a factor 4 to 5 rather than the factor 10 of the darks.

Figure `docs/figures/field_rate_images_ceers_nrcalong_full.jpg`: The same on the full frame, binned 2 by 2 for display.

The other five exposures of the visit, two more dithers on NRCALONG and
three on NRCBLONG, give the same numbers to the last digit: 0.0003 and
0.0002 DN/s for the stream clean on every one, against 0.0012 to 0.0022
and 0.0011 to 0.0017 for SIRS, with the sky changed by -0.06 percent
and its large-scale structure by 0.2 to 0.3 percent on each. NRCBLONG
has more row banding after SIRS than NRCALONG and the same column
structure.

Figure `docs/figures/field_rate_images_ceers_nrcblong.png`: CEERS on NRCBLONG, exposure 00001, same layout and statistics.

The exposure predates the calibration darks by three years and had a
different superbias subtracted (reference 0220 against 0443, 11700 DN
apart). The reset step reports the mismatch and uses the pedestal map
as the residual against the exposure's own superbias, which is the
quantity that transfers between epochs; and it carries the first-read
timing from the calibration's 10.7 s first-group time to this
pattern's 48.3 s as a fixed delay of 0.6 s, which the two-parameter fit
on this exposure confirms (k of 1.009 against 0.987 predicted, within
its errors). With that, the reset point is used for 85 percent of
pixels, all 432 pixels saturated from group 2 get a rate, and the
median rate agrees with the likelihood fit to 0.02 percent.

This exposure also found a frame the step had not been cleaning. For
frame-averaged readouts the pipeline carries the first frame of the
first group separately, the ZEROFRAME, and the likelihood fit uses it
as a resultant of its own; BRIGHT1 has no separate zeroframe, so Omega
Cen never exercised it. Left uncleaned it carried the full 1/f banding
of a single read into the reset fit, which showed 0.0011 DN/s of row
structure against 0.0002 for OLS_C on the same cleaned ramp. The step
now cleans the zeroframe as it cleans the first read, as a difference
against the reset prior with the background scaled to one frame time,
and the reset fit matches OLS_C on all six exposures (0.0003 and 0.0002
DN/s) while fitting 900 to 2100 more pixels each.

#### Short ramps and a bright star: NEXUS

Three exposures from the NEXUS programs (5105 and 9263), reduced by the
same hands as the CEERS data with the team's remstriping recipe, so
that the comparison is against a baseline with a known pedigree:
NRCB2 F200W MEDIUM8 with six groups and a bright star (peak 378 times
the sky), NRCA1 F090W SHALLOW4 with six groups of 54 s on a 0.045 DN/s
sky, and NRCBLONG F277W MEDIUM8 with seven groups. The baseline
`_cal` files were converted back to raw DN/s through their
photometric conversion and the flat; remstriping was also run by us on
our own SIRS rate, and reproduces the baseline's structure statistics
to the last digit. All chains are measured on the same flat-fielded
sky pixels.

Figure `docs/figures/field_rate_images_nexus_nrca1.png`: NEXUS on NRCA1, SHALLOW4, flat-fielded, central 512 by 512 pixels. Over the full frame the row-mean scatter of the sky pixels goes from 0.0031 DN/s (SIRS) to 0.0012 with the jwst median, 0.0006 with remstriping and 0.0003 with the stream clean; the column-mean scatter from 0.0014 to 0.0014, 0.0006 and 0.0002. On the MEDIUM8 frames the same ordering holds with the stream clean at 0.0001 to 0.0002 against 0.0004 to 0.0009 for remstriping. Aperture photometry of the sources found on the SIRS rate agrees between every chain to 0.3 percent in every flux quartile, except the stream clean's faintest quartile at -0.7 percent, the faint-source bias of the injection test.

Figure `docs/figures/field_rate_images_nexus_nrcb2_full.jpg`: NEXUS on NRCB2 with the bright star, full frame binned 2 by 2. The star's halo and diffraction spikes reach 400 pixels from the core above the sky. Where the scene mask does not cover them, a linear estimator reads them as slow 1/f: azimuthally averaged over the unmasked pixels, the halo excess is 20 percent of the sky within 100 pixels of the star, 8 percent at 100 to 200 and 3 percent at 200 to 400, and the stream clean removes three quarters of the first and all of the rest, remstriping a quarter, 70 and 90 percent, the pipeline median little. The scene mask's tiled sky, with its 32-pixel boxes, absorbs the halo and masks only a quarter of the pixels within 100 pixels of the star. A second, coarse tier of the mask (a 12-pixel kernel on a 256-pixel tiled sky, with a higher detection quantile; a plain large-kernel tier at the fine tier's quantile masked 35 to 40 percent of the sky and was discarded) now masks the inner 100 pixels entirely and halves the removal there, at a cost of 3 percent of the pixels and no change in any sky statistic. Beyond 100 pixels it changes nothing: an excess of a few percent of the sky, smooth over hundreds of pixels, is what slow 1/f looks like to any method that has one exposure to work with, and remstriping removes it too (70 percent of the halo at 100 to 200 pixels, against 85 for the stream clean). Keeping the extended halo of a bright star needs the box background model, which protects structure above its box at the cost of the slow 1/f, or a scene model from more than one exposure. The vertical diffraction spike is a separate matter: it runs on past the mask at a few columns wide and a tenth of the sky for hundreds of rows, and a per-column offset estimated over the whole column then darkens the columns through the star, by 2 percent of the sky relative to their neighbours here. The scene mask now follows the column-aligned spike of each saturated star (a star being a large masked region around saturated pixels whose mask has spike arms, so it fills a small fraction of its bounding box, unlike a galaxy's) outward in blocks of rows against the local background on either side, until it fades; that adds 0.3 percent of the pixels and leaves the spike columns 1.6 percent of the sky above their neighbours, as remstriping does (1.1 percent), with no change in any sky statistic.

The short ramps also settled the reset offset. On SHALLOW4 the reset
fit raised the sky by 1.9 percent with the calibration's offsets, and
the offset measured on the exposure itself was 11 DN against the
calibration's -3, with the same superbias: the reset level drifts by
10 DN and more over months, so the offset cannot be a calibration
constant. The step now anchors it to each exposure's own sky pixels,
per channel, and with that the first-difference residual is centred
and the reset fit agrees with the likelihood fit without the reset
point to 0.1 percent at every brightness. What remains between the
reset fit and OLS_C on the sky of that exposure, half a percent, is
the likelihood fit's own difference from OLS_C on a faint 6-group ramp,
present without the reset point as well (see the observations below).

#### A NIRISS field

Six NIRISS F200W direct images from WFSS programs (NIS and NISRAPID, 7
to 20 groups) are the first NIRISS fields. NIRISS carries much stronger
column structure than NIRCam: on an 8-group NISRAPID exposure the
column-mean scatter of the sky pixels is 0.038 DN/s before cleaning,
visible as vertical bands, against 0.017 along rows.

Figure `docs/figures/field_rate_images_niriss_f200w_full.jpg`: NIRISS F200W, NISRAPID, 8 groups, flat-fielded, full frame binned 2 by 2. Row-mean scatter 0.0165 DN/s (SIRS), 0.0147 (jwst median), 0.0022 (stream clean); column-mean scatter 0.0384, 0.0117, 0.0039. On a 7-group NIS exposure with a 1.4 DN/s sky the stream clean reaches 0.0006 and 0.0009 against 0.0045 and 0.0066 for SIRS. Aperture photometry of 75 sources with complete apertures agrees between every chain and SIRS to 0.5 percent in every flux quartile.

The NIRISS frame also has four occulting spots, dark patches at a
tenth of the sky that no DQ flag marks except the flat reference's,
where they are DO_NOT_USE with a flat of zero. The jwst flat reader
smooths over such pixels so that dividing by the flat is harmless, and
the first run let them into the noise estimate, which read the deficit
as slow noise and raised the cleaned sky by 8 percent around the
largest spot. The step now reads the flat's own DQ and keeps those
pixels out of the background, and the excess is gone. The same holds
for any pixel a flat marks unusable on NIRCam.

The NISRAPID exposure also gave the NIRISS first-read timing: k =
0.950 at a first-group time of 10.7 s, a 0.5 s delay like NIRCam's,
now in the packaged calibration; the reset offset of every NIRISS field
exposure sits 5 to 7 DN above the darks', which the per-exposure sky
anchoring absorbs.

The 7-group NIS exposure showed where likelihood fitting and OLS_C part
company. Fitting the same ramp six ways (raw and cleaned, with OLS_C,
stcal's LIKELY and the reset fit) and measuring 115 sources with
complete apertures in all six, the likelihood fit is 0.8 percent above
OLS_C and the reset fit 1.3 percent, on the raw ramp and the cleaned
ramp alike. Per pixel, between 5 and 50 DN/s, both likelihood fits
carry an upper tail of 5 to 7 percent at the 84th percentile against
OLS_C, and in the cores of bright stars, where the jump step and the
charge migration step flag three to five of the seven groups, the two
fits differ by up to 17 percent either way: with most groups flagged
they weigh the few remaining differences differently, and neither has
a claim to the truth. The reset point itself adds half a percent. This
is the LIKELY-against-OLS_C observation of the CEERS section, larger
on frame-averaged 7-group data with charge-migration flags, and one for
the stcal developers rather than for this step. The same comparison on
the NISRAPID exposure is within 0.2 percent. (A first look at this
exposure had the reset fit's aperture photometry off by tens of
percent; that was the measurement summing apertures with missing
pixels as zero, where OLS_C leaves the saturating cores without a rate
and the reset fit fills them, 2197 pixels here with values consistent
with each pixel's first difference. Apertures with a missing pixel in
any chain are now left out.)

#### Dispersed exposures

A NIRISS WFSS exposure carries the dispersed zodiacal and stray light as
a smooth background that varies along the dispersion by a factor of
three across the frame. A scalar median background hands that gradient
to the filter, which removes it (a change of 25 to 40 percent of the sky
at the 95th percentile of pixels, the sky lowered by 2 percent), and the
box-spline background keeps the gradient but gives back the banding. The
pipeline's own WFSS background step scales a CRDS reference image of the
dispersed background to the data; the stream clean does the same with
`background_method='crds_background'`, its default on WFSS exposures:
the reference is scaled once to the
exposure, by the median ratio to the ramp's own slope image over the
masked background pixels, and the scaled reference is subtracted from
every group difference, scaled only by its time, through the same fixed
background path as the box-spline model, so nothing is fitted per
difference. The scene mask is then built on the residual, where the
traces stand against a flat sky.

On a dispersed exposure the row- and column-mean scatter of the rate is
the dispersed background itself, so the banding has to be measured on
the background-subtracted rate (rate minus the scaled reference,
flat-fielded, over the mask). GR150C (NIS, 8 groups) and GR150R (NIS, 19
groups), rows and columns in DN/s:

| grism | chain | rows | cols |
|---|---|---|---|
| GR150C | SIRS | 0.0059 | 0.0063 |
| GR150C | jwst median | 0.0065 | 0.0172 |
| GR150C | stream clean, median | 0.0071 | 0.0239 |
| GR150C | stream clean, box 256 | 0.0050 | 0.0060 |
| GR150C | stream clean, crds_background | 0.0040 | 0.0010 |
| GR150R | SIRS | 0.0021 | 0.0067 |
| GR150R | jwst median | 0.0021 | 0.0041 |
| GR150R | stream clean, median | 0.0250 | 0.0077 |
| GR150R | stream clean, box 256 | 0.0053 | 0.0078 |
| GR150R | stream clean, crds_background | 0.0004 | 0.0003 |

With the CRDS background the column banding (the readout stream) drops
by a factor 6 on GR150C, the residual median of the background-subtracted
sky stays within 0.05 percent of the sky and the 5th and 95th percentiles
of the per-pixel change are SIRS's own, so nothing at large scales is
touched. The row banding on GR150C drops only from 0.0059 to 0.0040: the
per-column offsets are off on dispersed exposures (see below), and with
them on it went to 0.0010. The GR150R rows were measured before the skip
rule and with those offsets still on; they are kept as the record. The
jwst median imprints column structure on GR150C, where the dispersion
runs along rows.

Figure `docs/figures/field_rate_images_wfss_gr150c_full.jpg`: NIRISS GR150C, NIS, 8 groups, the scaled background reference subtracted from every panel for display, flat-fielded, full frame binned 2 by 2. The stream clean with the scaled CRDS background takes the column-mean scatter of the background-subtracted sky from 0.006 to 0.001 DN/s and the row-mean scatter to 0.004; the jwst median leaves the banding and adds column structure at the right edge, where the dispersed background falls off.

The sky statistics do not tell the two grisms apart, but the geometry
does. NIRISS reads along science columns, and GR150R disperses along
that axis, so a trace is a feature of a few hundred clocks in the
readout stream (over a thousand for a NIRCam GRISMR trace) with its
continuum shape and emission lines at the 30 to 300
pixel scales the Wiener gain passes. Masked traces are bridged from the
samples either side, but faint traces and trace wings under the mask's
threshold would have their spectral shape read as slow noise and
removed, a spectrophotometric bias no sky statistic can see. GR150C
traces cross the stream in a few clocks, above the cutoff, and are safe.
NIRCam has the same split the other way round: it reads along rows, so
GRISMR is the stream-parallel case and GRISMC the safe one. The step
therefore skips a stream-parallel grism with a warning, decided from the
filter and pupil keywords before any WCS exists, and
`allow_dispersion_along_stream` overrides it at that risk. The skip
stands; no injection test was run, since the bias follows from the
geometry and would only be larger for the longer NIRCam traces.

The safe grisms are the ones whose traces run along the axis orthogonal
to the stream, and two NIRCam F444W GRISMC exposures, a deep field
(program 5398, NRCALONG, SHALLOW2, 5 groups) and the standard star
P330-E (program 1076, BRIGHT1, 5 groups), showed four ways in which a
source along that axis gets into the noise estimate. The measure used
throughout is the correction (cleaned minus SIRS) in trace pixels minus
the correction in the sky pixels of the same row, per row, in DN/s: zero
when the traces are left alone, since the local background of an
extraction, taken from the columns beside a trace, sees the same change
as the trace.

First, that axis has its own correction: `column_offsets` removes each
column's masked mean from every difference (a per-column offset that
drifts during the integration; on NIRISS the same correction runs along
science rows). On an imaging field an undetected galaxy covers 10 or 20
pixels of a 2048-pixel column and biases that mean by nothing; a
dispersed trace covers hundreds to over a thousand pixels of it. With
the per-column offsets on, the trace excess on the deep field was -0.019
DN/s, a third of one pixel's noise in every pixel of every trace, and
the summed flux of the eight longest traces fell by 3 to 29 percent. The
outskirts of the detected traces and the whole of the undetected ones
raise the masked mean of their columns, and the offset removed takes
that flux out of every trace in the column. No estimator repairs it: a
median instead of a mean, the mask widened by 6 to 48 pixels across the
dispersion, or a high-pass of the offsets across columns each leave at
least a third of the bias, because the column structure the correction
is after is itself smooth over tens of columns (a hold-out on even and
odd rows shows the removable part is largely at scales above 17 columns)
and so indistinguishable from the outskirts of a dispersed source. The
step therefore turns `column_offsets` off on WFSS exposures and says so
in the log; the column banding it would have removed stays.

Second, the CRDS background references are not all in the same units.
The NIRISS ones are stacks of rate images and carry the detector's
pixel response (the correlation of their pixel-scale structure with the
flat's is 0.96); the NIRCam ones are flat-fielded and smooth
(correlation 0.00). Taken as raw units, the NIRCam reference left the
sky times (flat minus 1) in the residual, and the flat's large-scale
structure (5 percent at the left edge of channel 0, for instance) was
read as slow noise: dark bands 300 columns long beside the edge and
beside bright traces, and cross-hatching in the flat-fielded figures.
Neither the files (no BUNIT, a history that does not mention the flat)
nor the pipeline documentation say which convention a file follows, and
the pipeline itself subtracts both before flat fielding, so the step
tells the two apart from the pixel pattern: the correlation of the
reference's pixel-scale structure with the flat's is 0.82 to 0.97 for
the NIRISS files against the flats CRDS selects for them and within
0.03 of zero for the six NIRCam files in hand, from 2018 and 2025. The
value is logged, a warning marks a correlation between 0.3 and 0.7, and
`reference_units` settles it by hand; a flat-fielded reference is
multiplied by the flat before scaling.
The pipeline's own background step shows the same thing. Run as
delivered on the deep field (Detector1 defaults, the step's WFSS
subtraction with the stream clean's background mask, then divided by
the flat as `flat_field` does next), it leaves the flat's large-scale
structure in the sky: the residual's structure on 33-pixel scales is
0.020 DN/s on a sky of 0.54, its column-mean scatter 0.016 and its
pixel scatter 0.055, against 0.013, 0.011 and 0.048 with the reference
multiplied by the flat first; the scaling factors are 0.576 and 0.579.
The same run shows a second, larger mismatch that has nothing to do
with the flat: along the dispersion the reference falls to 0.2 of its
plateau at the bottom edge of the frame while the sky falls only to
0.48, so the ratio of sky to scaled reference is 2.4 at row 16, 1.3 at
row 256 and within 6 percent of one above row 500. The P330-E exposure,
three years earlier and on another field, has the same shape (2.0 at
row 16, 1.25 at row 256), an F444W direct image has no edge excess at
all (flat-fielded sky 0.68 DN/s in every row), and 93 percent of the
variance of the residual is a function of row alone. The pipeline
product therefore keeps a gradient of a quarter of the sky over the
bottom quarter of the frame, the same for either reference convention;
the figure removes that one-dimensional profile from both panels to
show the two-dimensional difference, which is the flat's. The NIRCam
WFSS known issues page notes that the single master background can
over-subtract and recommends a local background subtraction on the
extracted spectra, which handles both.

Figure `docs/figures/wfss_bkg_units.jpg`: The jwst background step on the program 5398 GRISMC exposure, the result divided by the flat, the row profile along the dispersion removed from both panels, full frame binned 2 by 2. Left, the NIRCam reference as delivered: the flat's structure times the sky stays in the frame, large-scale and pixel-scale. Right, the same reference multiplied by the flat before scaling. Made by `scripts/make_wfss_bkg_units_figure.py`.

Third, the scene mask destripes the draft rate image with row and column
medians before detecting, and a trace along 1190 of the 2048 rows of a
column set that column's median to 6.5 DN/s: the brightest trace of the
deep field was masked over 55 percent of its length and only on one
side, and its unmasked flux, at the same clock in every row, became a
sinusoid at the row period in the Wiener estimate, a bowl of 0.07 DN/s
across the whole channel in those rows. The medians now leave out
pixels more than 3 sigma from zero, and on a dispersed exposure only the
stream's axis is destriped; the mask of the deep field grows from 10 to
17 percent of pixels as the fainter traces are found, and imaging masks
change by half a percent of pixels.

Fourth, what the mask still misses along the columns, the wings and the
faintest traces, is a pattern that repeats every readout row, a line at
the row frequency (524 clocks on a full frame) that the Wiener gain
passed as noise: the mean correction against clock phase within a row
had a fundamental of 0.004 DN/s on the deep field, against 0.0004 on an
imaging field. On a WFSS exposure the gain is now notched there, a
Gaussian 2 percent wide, which takes the fundamental to 0.0002 for a
tenth more column banding left in.

With all four in place the trace excess is -0.0008 DN/s on the deep
field and -0.004 on P330-E (0.05 and 0.08 of a pixel's noise, the
measurement's own scatter), and -0.001 on NIRISS GR150C, where it had
been -0.005 with the per-column offsets on and -0.004 with them off
before the mask and the notch. Background-subtracted rate, rows and
columns in DN/s:

| exposure | chain | rows | cols |
|---|---|---|---|
| 5398 GRISMC | SIRS | 0.0146 | 0.0145 |
| 5398 GRISMC | jwst median | 0.0096 | 0.0212 |
| 5398 GRISMC | stream clean, crds_background | 0.0011 | 0.0104 |
| 1076 GRISMC P330-E | SIRS | 0.0275 | 0.0153 |
| 1076 GRISMC P330-E | jwst median | 0.0183 | 0.0237 |
| 1076 GRISMC P330-E | stream clean, crds_background | 0.0022 | 0.0067 |

The row banding (the NIRCam stream) drops by a factor 12 on both; the
column banding drops by a quarter and a half, which is what the even-odd
correction and the stream filter leave, and the rest stays by choice.
The residual median of the background-subtracted sky moves by +0.2 and
-0.2 percent of the sky; the jwst median lowers it by 8 and 4 percent,
its clipped median taken over the traces. One large-scale change does
happen on the deep field. The NIRCam reference spans a factor of four
along the dispersion, and one scale does not make it the sky: the row
medians of the residual run from -0.03 to +0.13 DN/s, a quarter of the
sky, from the top of the frame to the bottom. That gradient runs across
the stream, so the filter reads it as slow noise and removes it (the 5th
percentile of the per-pixel change is -0.14 DN/s), whereas on NIRISS
GR150C the reference spans only 0.7 to 1.06 and the residual is 0.01
DN/s. A fitted additive offset would halve the NIRCam residual and make
no difference to the cleaned rate, since the filter takes the rest; the
trace excess shows that an extraction is unaffected. The P330-E
exposure also shows the limit of the background reference: a square of
0.2 DN/s over the top 400 rows of the frame, scattered light the
reference does not hold, of which the chisel mask catches a tenth; the
stream clean removes 0.06 DN/s of it as slow noise where the mask does
not reach, the jwst median adds 0.16. A `scene_model` is the remedy
where such a feature matters.

Figure `docs/figures/field_rate_images_wfss_grismc_full.jpg`: NIRCam GRISMC, NRCALONG, 5 groups, program 5398, the scaled CRDS background subtracted from every panel for display, flat-fielded, full frame binned 2 by 2. The traces run along columns, across the readout stream; the stream clean takes the row-mean scatter of the sky from 0.015 to 0.001 DN/s and leaves the traces as SIRS has them.

#### Crowded fields and extended emission

The 1/f workshop test set adds two crowded LMC fields, a sparse MEDIUM8
field and an exposure of NGC 1566 that fills half the frame. On the
crowded fields a quarter of the frame is masked and the unmasked pixels
still carry star wings, so the tiled sky on 32-pixel tiles ranges from
0.6 to 1.4 of its median; the stream clean removes the banding (NIRCam
0.020 to 0.002 DN/s on rows, NIRISS 0.097 to 0.010 on columns), lowers
the sky by 1 and 3 percent, and leaves photometry within 0.2 percent of
SIRS in every quartile. The jwst median lowers the NIRISS LMC sky by 14
percent, its clipped-median background on a crowded field.

Figure `docs/figures/field_rate_images_ngc1566_nrcb2_full.jpg`: NGC 1566 on NRCB2, BRIGHT1, 3 groups, flat-fielded, full frame binned 2 by 2, the stream clean with the 32-pixel box background. The galaxy fills the left half of the frame and its disk is not source to any mask; under the scalar median the stream clean removes the disk's gradient (30 percent of the sky either side) and the jwst row medians do the same in stripes. In the galaxy-free right third the box model leaves the sky unchanged to 0.1 percent and removes 28 percent of the row structure; a 128-pixel box removes 57 percent for 0.8 percent of bias, and 256 pixels 60 percent for 4. In one exposure, extended emission and slow 1/f are the same thing to any estimator: the box model is the safe choice, a scene model from the other dithers the way to do better, and the step now warns when the unmasked background under the median carries that much structure.

#### What this builds on

Neither step starts from scratch, and the comparisons in this write-up
are against methods that do most of the work.

- The reference-pixel correction is SIRS, Simple Improved Reference
  Subtraction (Rauscher, Fixsen and Mosby 2022, JATIS 8, 028002,
  doi:10.1117/1.JATIS.8.2.028002). Every chain here, including the jwst
  one, runs on SIRS-corrected ramps, and SIRS is now the Detector1
  default for NIRCam. `stream_clean` removes what SIRS leaves: the
  1/f within and between the output channels that the reference
  pixels cannot see, and the per-read even-odd and column offsets.
- The idea of cleaning rows and columns after masking the sources is
  the CEERS team's (Bagley et al. 2023, ApJL 946, L12,
  doi:10.3847/2041-8213/acbb08), who subtract masked row medians per
  amplifier and masked column medians from the rate images. The
  per-column offsets of section 3 are that method's column pass moved
  into the ramp as a mean, so that it stays linear.
- The jwst `clean_flicker_noise` step, the comparison throughout,
  descends from Chris Willott's image1overf for its median method and
  from NSClean for its Fourier method, which is also the closest
  relative of the Wiener clean. image1overf is software without a
  paper of its own: Willott, C., `image1overf.py` in
  https://github.com/chriswillott/jwst (GPL-3.0), described and
  recommended on the STScI page on NIRCam 1/f noise removal methods
  (https://jwst-docs.stsci.edu/known-issues/nircam-known-issues/nircam-1-f-noise-removal-methods).
  NSClean is Rauscher 2024, PASP 136, 015001,
  doi:10.1088/1538-3873/ad1b36.
- The ramp fit is Timothy Brandt's likelihood method (Brandt 2024,
  PASP, doi:10.1088/1538-3873/ad38d9, and the companion jump-detection
  paper doi:10.1088/1538-3873/ad38da), as implemented in stcal as the
  LIKELY algorithm. `reset_rampfit` only adds one resultant, the
  reset level with its own variance and timing, to that machinery.
- The scene mask follows NoiseChisel (Akhlaghi and Ichikawa 2015,
  ApJS 220, 1).

#### Observations on stcal and jwst

- `ramp_fit_data` scales the read-noise array it is given in place
  (`readnoise_2d *= gain_2d / sqrt(2 nframes)` for OLS). This is a side
  effect rather than a bug in the fit: the pipeline step discards the
  array afterwards, so pipeline products are unaffected, but any caller
  that reuses the array it passed in gets electrons per single read
  instead of the reference file's DN. It cost this work one wrong
  calibration file (sigma_b 15.9 instead of 16.8 DN) before it was found;
  a copy inside `ramp_fit_data` would remove the trap.
- The OLS_C pedestal is `group0 - slope * ((nframes + 1) / 2 + dropframes)
  / (nframes + groupgap)` with the slope in DN/s and no group time; on a
  ramp with reset 20 DN, 1 DN/s and 10 s groups it is 29, not 20. This is
  a bug in the optional `pedestal` output only: the slope, variance and
  DQ outputs do not use it. The pedestal is wrong by the slope times the
  group time less about one second, so it is right for darks, where the
  slope is near zero, and increasingly wrong for pixels with signal. No
  downstream pipeline step reads it; the calibration here uses it on
  darks, where the error is below 0.01 DN.
- `average_dark_current` reaching ramp fitting is zero for NIRCam, and
  the plan's reading of the two fitters is that each adds it to a DN/s
  rate before dividing by gain once. That reading was not re-checked in
  this work. If it is right it is a units bug that has no effect on
  NIRCam, whose dark reference files carry no dark-current array, and
  that would mis-scale the dark contribution to the Poisson variance by
  the gain for an instrument whose files do carry one.
- On a faint sky (0.045 DN/s, 6 groups of 54 s) the LIKELY algorithm
  gives rates 0.5 percent below OLS_C on the sky pixels and 0.2 percent
  below on faint sources, with no reset point involved (0.1 percent on
  the 0.68 DN/s CEERS sky). Which of the two is right at low signal
  needs a known-rate test; the Poisson weighting of the likelihood fit
  depends on the count-rate estimate it starts from.
- The jwst `clean_flicker_noise` step inherits the random walk of
  section 3: on this dark it raises the rate noise from 0.0205 to
  0.0213 DN/s and the row-mean scatter from 0.0053 to 0.0079. This is not
  a coding error but a property of the design, a per-difference nonlinear
  estimate (the median, the per-difference background model, the
  clipping) followed by a group rebuild; it means that on a dark or a
  sparse field a user who enables the step gets a rate image that is 4
  percent noisier per pixel and 50 percent noisier in row banding than
  without it, while the group differences themselves look cleaner. A
  linear estimator applied identically to every difference avoids it, and
  the step's existing median could not be made linear.

#### Open items

- Above the background box scale the cleaned rate reproduces the model,
  so sky structure at periods of a few hundred rows is protected only to
  the extent the model follows it; a model informed by more than one
  exposure, or by the mosaic, would let the box grow without losing sky.
- The scene mask stops where a source drops below the clipping level and
  its wings extend further; the step has a `mask_dilation` option to
  grow the masked regions, not yet measured. The 4-pixel reference border
  is always excluded from the fit.
- The pixel-to-pixel spread of the first-read response, which the field
  chi-square of 2 points at, needs a brighter or repeated field to measure.
- The dark and injection experiments on NRCALONG, and a NIRISS field.
- A scene model from the dithered mosaic, blotted back into each
  exposure, in place of the per-exposure background model. Tried on
  Omega Cen NRCA1 (four dithers, Image3 with tweakreg and skymatch
  skipped, the outlier-detection blots times the flat as the model, the
  step's `scene_model` option): it does not help there. The blot sits
  5 percent above the exposure on the faintest pixels and its error
  field on the sky is 0.014 DN/s rms in 16-pixel blocks, larger than
  the 1/f being removed; the filter's unit gain at zero frequency turns
  the mean of that error into a sky shift of 2 to 5 percent per channel
  and its structure into row and column scatter four times that of the
  scalar background, with star photometry unchanged to 0.2 percent.
  Re-anchoring the model's level per channel repairs the sky but not the
  structure; smoothing the blot before use goes the wrong way (a
  4-pixel Gaussian lifts the sky by 15 percent, since it spreads star
  light into the sky pixels), and removing the model's row-aligned part,
  which is the other dithers' residual banding carried through the
  mosaic, recovers 40 percent of the structure but not the rest. A
  four-dither drizzled median of an undersampled crowded field is not
  accurate enough at the slow scales; the route stays open
  for sparse fields and smooth extended emission, where the blot error
  is small and the box-size question is the one that matters, and a
  crowded field would need a PSF-based star model instead.
- Extended-source photometry has not been tested; the injection used
  compact Gaussians only.

### 2026-11-01: the NRCB1 pedestal map on a field

The 2026-10-22 calibration left NRCB1 as the outlier (sigma_b 20.8 DN
against 15 to 18 on the other detectors, pedestal row scatter 6 DN
against about 3) and its map prior unchecked on a field. Checked on two
program 6154 exposures not used for the timing calibration
(jw06154001001_02101_00001 and jw06154001002_02101_00001, BRIGHT1, 5
groups, March 2025, a year before the 9022 darks), with NRCA1 and NRCB2
on the same exposures for comparison. Script `scripts/exp_map_prior.py`,
results in `results/2026-11-01_map_prior/` (run by `run.sh` there).
Detector1 to jump with SIRS, no stream clean; the data and the
calibration had the same superbias on every detector.

Two measurements:

- Transfer. On sky pixels (no flag in any group, first difference -20
  to 50 DN) the field's own reset level is group 1 minus the rate of
  groups 2 to 5 times the calibrated first-read time (k per channel
  times t1), with each channel's median taken out. "Field scatter" is
  its robust scatter, "after the map" the robust scatter of it minus
  the high-passed pedestal map the step uses, "map scatter" the map's
  own robust scatter, "slope" the least-squares slope of the field level
  on the map, and "variance removed" the field scatter squared minus the
  after-the-map scatter squared.
- Effect on the fit. ResetRampFitStep with the scalar prior and with
  `use_pedestal_mean`, both with the file's per-channel timing and the
  sky-anchored offsets. Chi-square of the reset difference (the mean
  squared normalized residual, over the pixels where it was used) over
  all clean pixels and over the bright ones (first difference 50 to 2000
  DN); and the rate difference map minus scalar on sky pixels, its
  robust per-pixel scatter and the robust scatter of its row means,
  against the per-pixel scatter of the OLS_C sky rate for scale.

| detector | observation | field scatter (DN) | after the map (DN) | map scatter (DN) | slope | variance removed (DN^2) | chi-square all, scalar / map | chi-square bright, scalar / map | rate difference per pixel (DN/s) | rate difference row means (DN/s) | OLS_C sky per pixel (DN/s) | field row-mean scatter (DN) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NRCA1 | 001 | 22.7 | 21.4 | 13.2 | 0.63 | 58 | 1.63 / 1.45 | 1.72 / 1.60 | 0.0143 | 0.0003 | 0.145 | 5.9 |
| NRCA1 | 002 | 22.6 | 21.3 | 13.2 | 0.63 | 56 | 1.62 / 1.45 | 1.72 / 1.58 | 0.0141 | 0.0003 | 0.126 | 5.2 |
| NRCB1 | 001 | 24.3 | 23.7 | 12.9 | 0.56 | 27 | 1.58 / 1.53 | 1.66 / 1.62 | 0.0106 | 0.0004 | 0.114 | 10.1 |
| NRCB1 | 002 | 24.5 | 23.9 | 12.9 | 0.57 | 29 | 1.60 / 1.55 | 1.65 / 1.63 | 0.0106 | 0.0004 | 0.110 | 10.2 |
| NRCB2 | 001 | 20.9 | 19.7 | 10.9 | 0.67 | 49 | 1.48 / 1.32 | 1.66 / 1.54 | 0.0114 | 0.0002 | 0.107 | 5.8 |
| NRCB2 | 002 | 20.8 | 19.6 | 10.9 | 0.67 | 49 | 1.47 / 1.31 | 1.66 / 1.54 | 0.0113 | 0.0002 | 0.097 | 5.5 |

Reading. The NRCB1 map transfers to the field, but it explains about
half as much of the field's per-pixel reset level as the NRCA1 and NRCB2
maps do (27 to 29 DN^2 of variance against 49 to 58), and it lowers the
chi-square of the reset difference by 0.05 against 0.16 to 0.17 on the
other two. A map that held only noise would give a slope near 0; a
perfect one, a slope of about 0.7 to 0.8 here, since part of each map's
scatter is the noise of its ten-integration mean; NRCB1's 0.56 sits
between. The map does no harm on NRCB1: the chi-square still goes down
on both exposures, the rate changes by a tenth of the per-pixel sky
noise, its row means by 0.0004 DN/s (0.3 percent of that noise, as on
the other two), and the sky level by 0.1 percent with either prior. The
two exposures agree to the last digit, so this is a property of the
detector and not of one exposure. Separately, NRCB1's first read has
twice the row structure of the others on this chain (10 DN of row-mean
scatter against 5 to 6); the high-passed map leaves rows alone by design,
and the stream clean removes them when it cleans the first read against
the prior. Conclusion: `use_pedestal_mean` is safe on NRCB1 but buys
about a third of what it buys on the other detectors; the scalar prior
loses little there.

### 2026-11-01: persistence after saturation on the program 6154 field

Goal: a `time` for PersistenceFlagStep per detector, packaged like the
reset calibration. First pass on the data on disk: the program 6154
`_rate` files from MAST (jwst 1.17.1), eight F115W BRIGHT1 exposures of
107 s per detector over 3000 s, so start-to-start delays of 193 to 2916
s; the pointing moves 800 to 6000 pixels between any two, so the stars of
one exposure fall on unrelated pixels of the next. Script
`scripts/measure_persistence.py` (the 2026-10-31 script, generalised),
results in `results/2026-11-01_persistence_6154/` with `pool.py` and
`fit_decay.py`, which made the tables below.

Method. Each exposure is a source: its pixels go in classes by signal
(rate times integration time); a pixel is saturated when it has
SATURATED or more than 60000 DN, since before jwst 2.0 ramp fitting set
SATURATED only where every group saturated. Each later exposure is a
target: its background (64-pixel boxes, the bright pixels of it and of
the recent exposures masked) is subtracted, and for each class of the
source the excess is the median, over the class's connected groups of
pixels (one group is about one star), of the group's mean residual,
minus the median residual of the target's other pixels. Neighbouring
pixels of one star land on one patch of the target, so the error comes
from the scatter of the group means, not of the pixels. The same
statistic with the groups moved to unrelated places in the target (two
fixed shifts) is the null, and persistence is excess minus null. Below,
the saturated classes of each detector are pooled per bin of
start-to-start delay with weights of one over the error squared (DN/s):

| detector | 0-300 s | 300-700 s | 700-1100 s | 1100-2100 s | 2100-2500 s | 2500-3100 s |
|---|---|---|---|---|---|---|
| NRCA1 | +0.066 +- 0.009 | +0.026 +- 0.005 | -0.005 +- 0.004 | +0.004 +- 0.003 | -0.008 +- 0.004 | -0.006 +- 0.004 |
| NRCA2 | +0.083 +- 0.009 | +0.011 +- 0.005 | -0.007 +- 0.005 | -0.008 +- 0.003 | -0.002 +- 0.004 | -0.013 +- 0.004 |
| NRCA3 | +0.260 +- 0.015 | +0.079 +- 0.005 | +0.032 +- 0.005 | +0.016 +- 0.003 | +0.016 +- 0.004 | +0.002 +- 0.004 |
| NRCA4 | +0.094 +- 0.008 | +0.023 +- 0.004 | +0.007 +- 0.004 | -0.002 +- 0.003 | -0.002 +- 0.004 | +0.003 +- 0.004 |
| NRCALONG | +0.126 +- 0.015 | +0.045 +- 0.009 | +0.001 +- 0.010 | +0.013 +- 0.006 | +0.011 +- 0.007 | -0.014 +- 0.007 |
| NRCB1 | +0.076 +- 0.009 | +0.015 +- 0.005 | +0.008 +- 0.005 | +0.008 +- 0.004 | -0.006 +- 0.004 | -0.000 +- 0.005 |
| NRCB2 | +0.142 +- 0.007 | +0.042 +- 0.005 | +0.029 +- 0.005 | -0.008 +- 0.003 | +0.003 +- 0.004 | +0.000 +- 0.004 |
| NRCB3 | +0.163 +- 0.014 | +0.022 +- 0.005 | +0.015 +- 0.004 | +0.002 +- 0.003 | +0.009 +- 0.004 | +0.010 +- 0.004 |
| NRCB4 | +0.221 +- 0.016 | +0.070 +- 0.006 | +0.032 +- 0.005 | +0.014 +- 0.003 | +0.017 +- 0.004 | +0.013 +- 0.004 |
| NRCBLONG | +0.038 +- 0.012 | +0.019 +- 0.008 | -0.002 +- 0.008 | -0.012 +- 0.005 | -0.012 +- 0.006 | -0.022 +- 0.006 |

The per-pixel noise of these short exposures is 0.10 to 0.15 DN/s (SW)
and 0.2 to 0.4 DN/s (LW). Persistence is clear on every detector
within 700 s, and NRCA3 and NRCB4 are three to four times stronger than
the others, still at 0.013 to 0.017 DN/s after 2000 s. But values down
to -0.022 at long delays, which persistence cannot give, and nulls that
scatter by about 0.01 beyond their errors show what this crowded field
allows: a floor of 0.01 to 0.015 DN/s from the scene itself.

A power law, persistence = A (delay / 1000 s)^-alpha, fitted to the
pairs of each detector (reduced chi-square 1.7 to 2.8, so the errors
below are about 1.5 times too small), and the start-to-start delay at
which it reaches three levels:

| detector | A (DN/s) | alpha | 0.02 DN/s | 0.01 DN/s | 0.005 DN/s |
|---|---|---|---|---|---|
| NRCA1 | 0.0035 +- 0.0018 | 1.81 +- 0.33 | 383 s | 562 s | 824 s |
| NRCA2 | 0.0006 +- 0.0008 | 2.96 +- 0.74 | 313 s | 396 s | 500 s |
| NRCA3 | 0.0274 +- 0.0023 | 1.35 +- 0.06 | 1262 s | 2107 s | 3519 s |
| NRCA4 | 0.0038 +- 0.0015 | 1.95 +- 0.26 | 425 s | 606 s | 864 s |
| NRCALONG | 0.0110 +- 0.0042 | 1.46 +- 0.26 | 666 s | 1069 s | 1717 s |
| NRCB1 | 0.0021 +- 0.0014 | 2.18 +- 0.43 | 355 s | 488 s | 671 s |
| NRCB2 | 0.0103 +- 0.0020 | 1.60 +- 0.13 | 661 s | 1021 s | 1576 s |
| NRCB3 | 0.0058 +- 0.0018 | 2.03 +- 0.21 | 541 s | 761 s | 1071 s |
| NRCB4 | 0.0268 +- 0.0022 | 1.18 +- 0.07 | 1283 s | 2313 s | 4169 s |
| NRCBLONG | no fit | | | | |

Reading. Only NRCA3 and NRCB4 are measured well enough for a time:
their decay is slow (alpha 1.2 to 1.35) and reaches 0.01 DN/s after
2100 to 2300 s and 0.005 DN/s after 3500 to 4200 s, longer than the
current 2500 s default. On the other eight the fit is set by the first
two bins, since the later ones sit at the floor, so their steep alphas
(1.5 to 3) and short times are not trustworthy below 0.015 DN/s; a
deep exposure, with about 0.01 DN/s of per-pixel noise, needs the
times at 0.005 to 0.01 DN/s. So these data rank the detectors but do
not set the times. That needs a sparser field, longer sequences and
deeper targets: the NEXUS rate files (700 exposures per short-wave
detector, delays of hours) run through the same script on an astronode.

### 2026-11-02: persistence times from the NEXUS field

Data: the NEXUS `_rate` files (programs 5105 and 9263) from the
2026-11-01 run of snowblind_detector1, 700 exposures per short-wave
detector and 412 per long-wave detector (the long-wave grism exposures
are not among them), SHALLOW4 (311 s) and MEDIUM8 (623 s). Delays from
the end of one exposure to the middle of a later one run from 150 to
20500 s. `scripts/measure_persistence.py` ran on astronode-13 in 35
minutes (about 2 s per exposure); the tables are in
`results/2026-11-02_persistence_nexus/`. The per-pixel noise of a target
is 0.007 to 0.019 DN/s (short wave) and 0.015 to 0.055 DN/s (long wave),
ten times below the 6154 exposures. A test on one visit of NRCA3 first:
the 2k-5k DN class, which should barely persist, gave 0.001 DN/s at
every delay, so this field allows a floor well below 0.005 DN/s.

Two changes from the 6154 entry, both now in the scripts:

1. Pooling by median. With weights of one over the error squared, a few
   pairs took over their bins. In NRCB3's 60k-100k DN class at
   9600-19200 s, one pair whose null was 3.55 DN/s (one of the two
   shifted copies of a three-star source landed on a star of the
   target, and the row's error does not include the null's scatter)
   carried the bin to -0.032 +- 0.0001 DN/s; the median of its 403 pairs
   is +0.0012. A bin is now the median of its pairs, and its error comes
   from resampling the source exposures with replacement (300 times),
   since the pairs of one source share its stars.
2. Delay from the end of the source to the middle of the target. Against
   the start-to-start delay, the 425-600 s bin sat below the 600-850 s
   bin on several detectors (NRCA1 0.023 then 0.033 DN/s, NRCB4 0.036
   then 0.060): a short source ends long before the next exposure
   starts, a long one just before. Persistence decays from the end of
   the saturated exposure, and the rate of the target averages it over
   the target, which its value at the middle approximates (within 2% at
   the delays where the times fall, for alpha 1.3 and a 600 s target).
   Against this delay every curve falls steadily. PersistenceFlagStep
   now uses it, so an exposure longer than `time` still flags the next.

Saturated stars (every saturated class but 0k-60k), median per bin, in
mDN/s, against the delay from the end of the source to the middle of the
target:

| detector | 300-425 s | 600-850 s | 1200-1700 s | 2400-3400 s | 4800-6800 s | 9600-13600 s |
|---|---|---|---|---|---|---|
| NRCA1 | 40.9 +- 1.0 | 9.7 +- 1.1 | 4.6 +- 0.6 | 1.6 +- 0.3 | 1.3 +- 0.2 | 0.5 +- 0.2 |
| NRCA2 | 40.8 +- 2.2 | 12.2 +- 1.0 | 4.2 +- 1.0 | 1.6 +- 0.4 | 1.2 +- 0.2 | 0.2 +- 0.2 |
| NRCA3 | 84.8 +- 3.4 | 42.8 +- 3.3 | 19.0 +- 1.1 | 7.1 +- 0.8 | 4.3 +- 0.4 | 2.4 +- 0.3 |
| NRCA4 | 34.2 +- 2.3 | 9.3 +- 0.7 | 5.1 +- 0.5 | 2.4 +- 0.4 | 1.2 +- 0.2 | 0.6 +- 0.2 |
| NRCALONG | 31.5 +- 2.3 | 20.0 +- 3.6 | 10.2 +- 2.6 | 4.6 +- 0.7 | 3.4 +- 0.7 | 2.3 +- 0.5 |
| NRCB1 | 38.8 +- 1.4 | 9.7 +- 1.7 | 4.7 +- 0.2 | 1.1 +- 0.6 | 0.8 +- 0.2 | 0.4 +- 0.2 |
| NRCB2 | 52.6 +- 4.5 | 18.3 +- 4.1 | 7.6 +- 1.1 | 3.9 +- 0.3 | 2.1 +- 0.2 | 0.7 +- 0.4 |
| NRCB3 | 68.3 +- 3.5 | 27.2 +- 3.4 | 11.3 +- 1.1 | 3.9 +- 0.5 | 2.8 +- 0.5 | 1.5 +- 0.4 |
| NRCB4 | 73.2 +- 6.0 | 27.6 +- 8.4 | 15.9 +- 2.4 | 6.5 +- 0.8 | 4.4 +- 0.4 | 2.6 +- 0.3 |
| NRCBLONG | 18.6 +- 1.6 | 9.8 +- 3.4 | 3.0 +- 0.8 | 0.8 +- 0.3 | 0.7 +- 0.5 | 0.1 +- 0.3 |

The power law persistence = A (delay / 1000 s)^-alpha fitted to the 14
bins from 212 to 38400 s (each with at least 20 pairs). "Typical" is
the delay at which the median star falls to 0.005 DN/s, by the fit and
by the bins themselves (where they first fall below it, log-log
interpolation between neighbours). "Packaged" is the time now in the
package: the delay at which twice the median star's persistence falls
to 0.005 DN/s (the median star at 0.0025 DN/s), the longer of the
fit's and the bins'. "Start to start" is the typical time by the fit
from the same median pooling against the old delay, for comparison:

| detector | A (mDN/s) | alpha | reduced chi-square | typical, fit | typical, bins | packaged | start to start |
|---|---|---|---|---|---|---|---|
| NRCA1 | 7.9 +- 1.1 | 1.50 +- 0.17 | 19.0 | 1350 s | 1364 s | 2141 s | 1570 s |
| NRCA2 | 9.1 +- 0.6 | 1.45 +- 0.05 | 2.4 | 1506 s | 1347 s | 2428 s | 1810 s |
| NRCA3 | 29.8 +- 0.7 | 1.14 +- 0.02 | 1.4 | 4770 s | 4720 s | 10502 s | 5100 s |
| NRCA4 | 8.1 +- 0.5 | 1.23 +- 0.07 | 4.0 | 1484 s | 1467 s | 2931 s | 1790 s |
| NRCALONG | 16.9 +- 1.0 | 0.95 +- 0.05 | 2.6 | 3575 s | 2754 s | 9563 s | 3310 s |
| NRCB1 | 7.7 +- 0.6 | 1.50 +- 0.09 | 6.8 | 1340 s | 1019 s | 2129 s | 1530 s |
| NRCB2 | 14.5 +- 0.3 | 1.21 +- 0.02 | 1.2 | 2422 s | 2271 s | 4571 s | 2420 s |
| NRCB3 | 20.3 +- 0.9 | 1.27 +- 0.04 | 2.7 | 3006 s | 2663 s | 6074 s | 3350 s |
| NRCB4 | 25.7 +- 1.1 | 1.03 +- 0.03 | 1.5 | 4939 s | 4859 s | 12006 s | 5140 s |
| NRCBLONG | 6.1 +- 0.5 | 1.46 +- 0.09 | 1.7 | 1145 s | 1144 s | 2250 s | 1480 s |

Why twice (James, 2026-11-02): persistence varies across a detector,
likely with its quantum efficiency (as seen on the similar Euclid NISP
detectors), and with each pixel's history of illumination, how long and
how often it saturated; the median star shows neither. So the packaged
times flag more than the typical star needs, and the files keep
`typical_time` (the longer of the two typical columns) for those who
would rather flag less. At the median star's 0.0025 DN/s the decay has
flattened, so the bins give the longer delay on seven of the ten
detectors (by up to 24%, NRCB4 12006 against 9710 s from the fit).

Reading:

- Every time lies inside the delays measured (up to 20500 s), so none
  is extrapolated. The typical times against the old 2500 s default:
  shorter on NRCA1, NRCA2, NRCA4, NRCB1 and NRCBLONG (1150 to 1500 s),
  about the same on NRCB2, longer on NRCB3 and NRCALONG (3000 to 3600
  s) and twice as long on NRCA3 and NRCB4 (4800 to 4900 s). The ranking
  matches 6154, whose extrapolated 3519 and 4169 s for NRCA3 and NRCB4
  were short. The packaged times are 2100 to 2900 s on NRCA1, NRCA2,
  NRCA4, NRCB1 and NRCBLONG, 4600 to 6100 s on NRCB2 and NRCB3, and
  9600 to 12000 s on NRCALONG, NRCA3 and NRCB4. The flags cover only
  the pixels that saturated, a median of 240 per short-wave exposure
  and 700 per long-wave exposure (0.006% and 0.017% of the detector), so
  flagging longer costs little.
- Where the typical fit and bins differ (NRCALONG, NRCB1, NRCA2) the
  bins flatten near 0.005 DN/s (NRCB1: 0.0049 at 1200-1700 s, 0.0047
  at 1700-2400 s), so the crossing is poorly defined.
- NRCA1's chi-square of 19 comes from two bins at almost the same delay
  (medians 408 and 435 s) at 0.041 and 0.020 DN/s, so the delay is not
  all that matters. By the readout of the source, at 300-600 s stars
  from MEDIUM8 (623 s) exposures persist 1.1 to 1.9 times more than
  from SHALLOW4 (311 s) exposures (NRCA1 0.042 against 0.023, NRCA3
  0.087 against 0.079, NRCB4 0.077 against 0.049 DN/s); near the times,
  the two differ by up to a factor of 2 either way. How long a pixel
  stayed saturated matters, so exposures much longer than these may
  persist beyond the packaged times.
- SATURATED pixels with less than 60000 DN of signal (cosmic rays and
  snowballs that saturated partway through the ramp, stars just at the
  saturation level) persist clearly but less than the stars: 0.005 DN/s
  at 400 to 550 s on NRCA1, NRCA2, NRCA4 and NRCB1, 640 to 770 s on the
  long-wave detectors, 1100 to 1270 s on NRCB2 and NRCB3, and 1800 s on
  NRCA3 and NRCB4. That is well inside each detector's star time, which
  covers them; they stay out of the fit.
- Unsaturated pixels persist too (for issue #13), reaching 0.005 DN/s
  at: 2k-5k DN, under 400 s; 5k-10k DN, under 650 s (NRCA3 900 to 1000
  s); 10k-20k DN, 280 to 850 s (NRCA3 1100 to 1260 s); 20k-30k DN, 460
  to 1570 s (NRCA3 2000 to 2100 s). The 30k-40k and 40k-60k DN classes
  have 1 to 60 pairs per bin, too few for a time: each pair needs 20
  pixels of a class, and the ring of such pixels around a star is
  small. Pooling the stars of all pairs of a bin would measure them.
- NRCALONG's faintest class (2k-5k DN) sits at 0.004 to 0.006 DN/s from
  200 to 2400 s and 0.002 to 0.003 DN/s out to 20000 s, which is not
  how persistence decays (NRCBLONG: 0.0005 to 0.001 DN/s after 3000 s).
  The same offset under its star curve would lengthen its time, and it
  has the largest gap between fit and bins. The cause is unknown; a
  check: measure the bright pixels of later exposures in earlier ones,
  where persistence cannot be.
- Flagging on the ramp (a future step in Detector1 on `_uncal` or
  `_jump` data) can use the same times: the fitted curve is the
  persistence rate against the time since the saturated exposure ended,
  so such a step would flag each group whose time since that end is
  below the detector's time. It would also know how long each pixel
  stayed saturated, which the readout dependence above says matters.

Packaged with `python scripts/package_persistence_defaults.py
results/2026-11-02_persistence_nexus --dataset "..."` into
`src/snowblind/data/persistence_<detector>.asdf`.

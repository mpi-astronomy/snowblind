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
- Test data: see section C.4. Ask James for the file list before running on
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

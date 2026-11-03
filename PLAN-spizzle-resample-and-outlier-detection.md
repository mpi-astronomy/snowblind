# Plan: spizzle in place of drizzle in ResampleStep and OutlierDetectionStep

Status: handed off 2026-11-02 for implementation in `snowblind`. Parts A and
B were implemented on 2026-11-03 on the branch `feature-resample-spizzle`
(Part A committed as e72a8fa; sections 3 and 4, "Done"); Part C is not
started. The plan is written for an implementing agent
working in `~/dev/snowblind`, with `~/dev/spizzle`, `~/dev/jwst` and
`~/dev/stcal` checked out beside it.

spizzle (`~/dev/spizzle`, mpi-astronomy/spizzle, 0.1.0 on PyPI) is drizzle's
square kernel written out as an explicit sparse operator `W`, built once per
exposure and applied to the science data, the weights and every variance
plane, in float64 and in parallel. Its design history, measurements and the
decisions behind this plan are in `~/dev/spizzle/drizzle-sparse-plan.md`,
section 11, steps 5 and 6. Problems found in drizzle are in
`~/dev/spizzle/drizzle-issues.md`.

Deliverables, in order:

- Part A: `SpizzleResampleImage`, a subclass of jwst's
  `jwst.resample.resample.ResampleImage` that drizzles with spizzle. It has a
  sanity mode that reproduces the stock outputs, and a default mode with the
  exact variance and a noise factor.
- Part B: `ResampleStep` and `OutlierDetectionStep` running on it, plus a
  replacement for stcal's `gwcs_blot`, validated against the stock steps on
  real data.
- Part C, later: a better outlier detection step. It needs spizzle components
  that do not exist yet (section 6). Do not start it without the user.

spizzle is a library of numerical parts; the pipeline logic lives in snowblind.
If snowblind needs a change in spizzle, write it down and ask the user; do
not edit `~/dev/spizzle` from here.


## 0. Decisions already made (do not re-open)

All by the user, 2026-11-02, unless noted.

- The caller lives in snowblind: a subclass of `ResampleImage`, used by
  subclassed or patched steps. jwst 2.0.0 is the minimum (snowblind already
  requires it).
- Weights stay as in stcal: inverse variance of the background (`ivm`, from
  `VAR_RNOISE`), zero where DQ has DO_NOT_USE (and NON_SCIENCE, via jwst's
  `GOOD_BITS`). This gives the science mosaic the best noise.
- stcal's VAR and ERR in the mosaic are a kludge (section 5.1) and are not
  reproduced by default. The default products are:
  - VAR_RNOISE, VAR_POISSON, VAR_FLAT: the exact pixel variances,
    `Coadd.out_var`;
  - the noise factor as a separate plane;
  - ERR = `sqrt(sum of the VAR planes * noise factor)`.

  photutils can then use the variance and the noise factor together for
  aperture photometry.
- The correlation images (`Covariance.write_correlation`) stay out of the
  mosaic; at most an optional separate file.
- A mode that computes stcal's VAR and ERR (`spizzle.StcalVariance`) is a
  sanity check against the stock pipeline. It is not the default.
- No DQ propagation.
- No "INDEF" anywhere: fill values are numbers, NaN by default.
- The noise factor is computed in a second pass over the exposures
  (`spizzle.NoiseFactor`); reading the inputs twice is acceptable.
- Outlier detection will eventually be a snowblind step that calls spizzle for
  the drizzle parts (section 6).

Answers of 2026-11-03 (user), to section 7 and to the first read of this plan:

- Weights: ivm by default. The steps are not steered by what jwst's CRDS
  step parameters say; the aim is image fidelity. (CRDS's NIRCam
  `pars-resamplestep` sets `weight_type="exptime"` for every date; with
  likelihood ramp fits, ivm gives a mosaic whose noise makes sense, and is
  optimal.)
- Names: not jwst's class names, because stpipe fetches CRDS step
  parameters by class name (`pars-<classname>`). The resampling class is
  `SparseResampleImage` and its step `SparseResampleStep`. The outlier step
  of the new method (section 6) is to be named for it, e.g.
  `PairwiseOutlierStep` (user's suggestion). Part B's outlier step, jwst's
  method on spizzle, is `SparseOutlierDetectionStep` (agreed). The `strun`
  name of the resampling step is `sparse_resample` (user, replacing the
  earlier `spizzle_resample`).
- Noise factor: written as a separate file next to the mosaic with its own
  suffix, as source_catalog writes `_segm.fits`. A FITS extension through a
  schema snippet (as `META_SCHEMA` maps metadata to keywords) is possible
  later; deferred. Suffix not chosen.
- No DQ propagation: `propagate_dq` is not supported.
- The second pass rebuilds the operators.
- The switch to stcal's variance: `variance = option("exact", "stcal",
  default="exact")`.
- No context planes unless asked for: the user agrees to a spizzle change
  that makes `Coadd` skip them by default. Nothing in jwst reads CON; it is
  only copied into the product when `enable_ctx` is set.
- spizzle is an optional extra of snowblind for now; a hard dependency once
  the user is happy with it.
- MIRI is out of scope for now.
- Pixel scale ratio other than 1, or pixfrac below 1: later.


## 1. What spizzle provides

Install it in the `snowblind` env with `pip install spizzle`; it brings
numba. Use `pip install -e ~/dev/spizzle` only if the user is changing
spizzle at the same time. Public API, all in `spizzle`:

| name | what it does |
| --- | --- |
| `pixel_map(wcs_from, wcs_to, shape=None, stepsize=16, order=3, disable_bbox="to", threads=None)` | Pixel map like stcal's `calc_pixmap`. By default the WCS is evaluated every 16 pixels and interpolated with bicubic splines: 38x faster, at most 2.2e-8 pixel different on NIRCam. `stepsize=1` evaluates every pixel, bit for bit equal to `calc_pixmap`, on several threads. |
| `drizzle_operator(pixmap, out_shape, pixfrac=1.0, weights=None, weight_scale=1.0, region=None)` | The `Operator` of one exposure: `matrix` (scipy CSR, rows = output pixels of `box`), `box` (pair of slices into the output), `out_shape`, `reach`, `full()`. `region=(slice(y0, y1), slice(x0, x1))` drizzles only those input pixels, as drizzle's `xmin..xmax`, `ymin..ymax`, which are inclusive. Zero-weight entries are left out, so NaN data under zero weight never reach the output. |
| `Coadd(out_shape, n_var=0, fillval=nan)` | `.add(op, data, variances=(...), iscale=1.0)`; `.out_img`, `.out_wht`, `.out_var` (list, exact `(W*W) @ var / wsum**2`), `.out_ctx` (int32 planes, the same bit layout as drizzle's context). Sums in float64, divided once at the end. |
| `NoiseFactor(wsum, fillval=nan)` | Second pass: `NoiseFactor(coadd.out_wht)`, then `.add(op, var, iscale)` for each exposure, then `.noise_factor`. Every offset is included, in 16 bytes per output pixel. |
| `Covariance(out_shape, radius=None, dtype=float32)` | The covariance per pixel offset up to the reach, `noise_factor`, `correlation(k)`, and `write_correlation(path)` (needs astropy). It takes `(2r(r+1)+1)*4+8` bytes per output pixel: 60 at radius 2, 108 at radius 3. |
| `StcalVariance(out_shape, n_var=3, rnoise=0, weight_type="ivm", pixel_scale_ratio=1.0)` | stcal's VAR planes and ERR from the same operators: `.add(op, [rnoise, flat, poisson], iscale, exposure_time=None)`, `.out_var`, `.out_err`. The same arithmetic from stcal 1.18 to 1.20. |
| `blot(data, pixmap, fillval=nan, iscale=1.0, out=None)` | Bilinear blot, as drizzle's `blot_image(interp="linear")`, the method stcal uses. Bit for bit equal to drizzle's float64 code except within half a pixel of the edges (section 5.4). About 7x faster than drizzle on 11 threads. |

Each exposure's results do not depend on the number of threads. Everything
comes out in float64; jwst's datamodels want float32.


## 2. How jwst and stcal call drizzle today

Facts are for jwst 3.0.0 / main `e15d3e779` (the commit snowblind's
`_jwst_imports.py` is checked against) and stcal 1.20 (resample code identical
to stcal main of 2026-10-07).

- `stcal.resample.resample.Resample`, the base of `ResampleImage`:
  - `reset_arrays` creates a `drizzle.resample.Drizzle` for the science
    image, and a second one for `compute_err="driz_err"`.
  - `add_model(model_dict)`:
    - validates the input and computes the pixel scale ratio and `iscale`;
    - builds the pixel map with `calc_pixmap(wcs, output_wcs, shape,
      stepsize=pixmap_stepsize, order=pixmap_order)`. jwst's defaults are
      1.0 and 1, i.e. every pixel;
    - builds the weights with `build_driz_weight(model, weight_type,
      good_bits, flag_name_map)`, a float32 array;
    - subtracts `level` when not `subtracted`;
    - gets `xmin, xmax, ymin, ymax = resample_range(data.shape,
      wcs.bounding_box)`;
    - calls `Drizzle.add_image`, then `add_model_hook` and `update_time`;
    - calls `resample_variance_arrays`;
    - copies `out_img`, `out_wht` and `out_ctx` into `_output_model` after
      every exposure.
  - `resample_variance_arrays` makes a new full-frame `Drizzle` for each
    variance plane of each exposure, drizzling `sqrt(var)`.
  - `finalize` assigns the outputs, calls `finalize_resample_variance`
    (divides by `pixel_scale_ratio**2` for imaging), computes ERR as
    `sqrt(nansum(var planes))` (NaN where all are NaN), and calls
    `finalize_time_info`.
  - Private attributes in use: `_driz`, `_driz_error`, `_variance_info`,
    `_output_model`, `_finalized`, `_n_res_models`, `_group_ids`. A
    subclass that overrides these methods depends on them, so pin stcal and
    write down the version, as `_jwst_imports.py` does for jwst.
- `jwst/resample/resample.py`, `ResampleImage(Resample)`:
  - `add_model` (`:470`) converts the model with
    `input_jwst_model_to_dict` (`:801`);
  - `finalize` (`:492`), `reset_arrays` (`:512`);
  - `resample_group` (`:541`) is used by `resample_many_to_many` (`:605`),
    which outlier detection calls;
  - `resample_many_to_one` (`:656`) is used by stage 3;
  - `dq_flag_name_map` is set on the class.
- `ResampleStep.process` (`jwst/resample/resample_step.py:126` and `:142`)
  builds `resample.ResampleImage(...)` through the module attribute.
  `GOOD_BITS = "~DO_NOT_USE+NON_SCIENCE"`. With `enable_err` (the
  default), it passes `compute_err="from_var"` and `enable_var=True`.
- Outlier detection (imaging):
  - `jwst/outlier_detection/imaging.py:105` also builds
    `resample.ResampleImage`, with `enable_ctx=False, enable_var=False,
    compute_err=None`;
  - it then calls `median_with_resampling` and, for each exposure,
    `flag_resampled_model_crs`, which calls `gwcs_blot` twice
    (`jwst/outlier_detection/utils.py:393` and `:403`), for the median and
    for its error. Each call computes the full pixel map again, which is
    the same pixel map that resampled the exposure.
  - `gwcs_blot` is imported by name into `jwst.outlier_detection.utils`,
    so a patch must replace `jwst.outlier_detection.utils.gwcs_blot`.
  - The thresholds are `snr="5.0 4.0"`, `scale="1.2 0.7"`, `maskpt=0.7`.
    `flag_resampled_crs` flags `|data - blot| > snr * err + scale * D`,
    with `D` the largest difference between a blot pixel and its four
    neighbors, in two passes.
- Spectral resampling (`ResampleSpec`) uses the same stcal class with a 3D
  WCS. It stays on drizzle.


## 3. Part A: `SpizzleResampleImage`

The recipe that reproduces the stock outputs exists already:
`~/dev/spizzle/benchmarks/compare_stcal.py`, function `run_spizzle`. Per
exposure:

```python
m = input_jwst_model_to_dict(model, weight_type, True, compute_err)
weights = build_driz_weight(m, weight_type=weight_type, good_bits=good_bits,
                            flag_name_map=self.dq_flag_name_map)
data = m["data"] if m["subtracted"] or m["level"] is None else m["data"] - m["level"]
pixmap = pixel_map(m["wcs"], self.output_wcs, shape=data.shape, stepsize=..., order=...)
xmin, xmax, ymin, ymax = resample_range(data.shape, m["wcs"].bounding_box)
op = drizzle_operator(pixmap, out_shape, pixfrac=self.pixfrac, weights=weights,
                      region=(slice(ymin, ymax + 1), slice(xmin, xmax + 1)))
iscale = self._get_intensity_scale(m)       # 1 for imaging
variances = [m["var_rnoise"], m["var_flat"], m["var_poisson"]]
coadd.add(op, data, variances, iscale)       # science, weights, context, exact variances
stcal_var.add(op, variances, iscale)         # sanity mode only
```

Notes for the subclass:

- Keep stcal's flow for metadata: `validate_input_model`, the pixel scale
  ratio, group ids and pointings, `update_time`, `finalize_time_info`, and
  `add_model_hook` if anything uses it. Replace only the drizzle calls and
  the variance arrays. Do not normalize the output after every exposure, as
  stcal does: `Coadd` divides once at the end, and doing it per exposure
  costs a full pass over the mosaic each time.
- Order the variance planes as stcal does (`variance_array_names =
  ["var_rnoise", "var_flat", "var_poisson"]`). In `StcalVariance`, `rnoise=0`
  is the read-noise plane for the ivm weights.
- Sanity mode: `StcalVariance(out_shape, weight_type=self.weight_type,
  pixel_scale_ratio=self.pixel_scale_ratio)`. For "exptime" weights it needs
  `exposure_time=get_tmeasure(m)[0]`. ERR is `out_err`. jwst also accepts
  `weight_type=None`; check how stcal treats it in
  `resample_variance_arrays` before supporting it.
- Default mode: the VAR planes are `coadd.out_var` without stcal's
  `1/pixel_scale_ratio**2`. That factor is part of stcal's approximation;
  the exact variance needs no scale factor. The noise factor comes from a
  second loop over the model library in `resample_many_to_one`:
  `NoiseFactor(coadd.out_wht)`, then `nf.add(op, sum of the variance
  planes, iscale)` per exposure. Rebuild each operator (pixel map 0.02 s
  plus operator 0.2-0.3 s per NIRCam exposure), or keep them if memory
  allows. ERR = `sqrt(sum(VAR) * noise_factor)`.
- `compute_err="driz_err"` (`ResampleStep` with `single=True`) drizzles the
  ERR array as data, divided by the pixel scale ratio. With spizzle that is
  one more `Coadd` (or one more data plane) on the same operator.
- Square kernel only. For any other kernel, fall back to the stock class.
  `propagate_dq=True` is not supported (decision 0); see section 7.
- Cast the outputs to float32 for the datamodel.
- Memory per output pixel:
  - `Coadd` with 3 variance planes: 44 bytes;
  - `StcalVariance`: 48 bytes;
  - `NoiseFactor`: 16 bytes.

  At 2x oversampling and 10 times an exposure's area, the mosaic has about
  1.7e8 pixels, so `Coadd` alone is 6.9 GiB.

Done (2026-11-03, uncommitted on `feature-resample-spizzle`):

- `src/snowblind/resample.py`: `SparseResampleImage(ResampleImage,
  _SpizzleResample)`. `_SpizzleResample(stcal Resample)` overrides only
  `reset_arrays`, `add_model` and `finalize`, and sits between jwst's class
  and stcal's in the method resolution order, so jwst's methods (which call
  `super()`) reach it and none of jwst's code is copied.
  `resample_many_to_one` is overridden for the second pass (`add_noise_factor`
  per exposure); `finalize` raises if the exact variance has no noise
  factor, e.g. after `add_model` alone or `resample_group` with
  `enable_var`.
- Defaults that differ from jwst's class: `variance="exact"`,
  `pixmap_stepsize=16`, `pixmap_order=3`, `enable_ctx=False`. Other kernels,
  `propagate_dq=True`, a fill value that is not a number ("INDEF") and a
  non-imaging output WCS raise ValueError, rather than falling back to the
  stock class: the steps will not offer them.
- The noise factor is `SparseResampleImage.noise_factor` (float32), not
  attached to the output model; the step is to write it as its own file.
- spizzle change (user's go-ahead, uncommitted on spizzle `main`): `Coadd`
  keeps the context only with `context=True`; `out_ctx` is None otherwise.
  snowblind's optional extra is `snowblind[spizzle]`, `spizzle>0.1.0`, so it
  needs that change released.
- `tests/test_resample.py` (29 tests, synthetic TAN exposures, no CRDS):
  `variance="stcal"` against jwst's class plane by plane, with ivm and
  exptime weights and `driz_err`; the exact variance, noise factor (1 on
  the exposure grid, 2 half a pixel off it) and pixel scale ratio 2; fill
  values; rejected options.
- `scripts/compare_sparse_resample.py`: the comparison on real cal files
  (results in section 4).


## 4. Part B: the steps and their validation

- `ResampleStep`: a snowblind step class (e.g. `SpizzleResampleStep`, a
  subclass of jwst's `ResampleStep`) whose `process` builds
  `SpizzleResampleImage`. Alternatively, patch
  `jwst.resample.resample.ResampleImage` for the duration of a run. Both
  jwst steps look the class up through that module attribute, so one patch
  covers both. Follow snowblind's conventions: `snowblind.step.Step` for
  logging and `set_cal_step`, and the `stpipe.steps` entry point.
- `OutlierDetectionStep`: the resampling goes through the same class. Blot
  needs its own replacement:
  - `gwcs_blot(median_data, median_wcs, blot_shape, blot_wcs, pix_ratio,
    fillval, pixmap_stepsize, pixmap_order)` becomes `blot(median_data,
    pixel_map(blot_wcs, median_wcs, shape=blot_shape, ...),
    fillval=fillval)`, with float32 output;
  - to compute the pixel map once for the median and its error, replace
    `flag_resampled_model_crs`, or cache the last pixel map by
    `(id(blot_wcs), id(median_wcs), blot_shape)`;
  - stcal sets NaN pixel map entries to -1 because older drizzle needed it;
    spizzle's `blot` takes NaN.
- Validation against the stock steps:
  - Use the `snowblind` env, which has stock jwst 3.0.0, stcal 1.20 and
    drizzle 2.2.0.
  - The `mizzle` env (spizzle's) has jwst 3.0.0, but with the user's
    float64 drizzle branch installed over stcal's `drizzle<2.3` pin, so it
    is not stock.
  - Data: `/Users/jdavies/data/jwst_6154/reduction_mast/*_nrcalong_cal.fits`,
    8 NIRCam long-wavelength exposures. Short-wavelength NIRCam and MIRI
    are needed too, especially for outlier detection: ask the user for
    them.
- Targets for sanity mode, with `pixmap_stepsize=1` as the stock step uses.
  Measured by `compare_stcal.py` in the `mizzle` env, relative difference
  over the 13.8 M pixels with weight:

  | plane | median | max |
  | --- | ---: | ---: |
  | SCI | 3.1e-8 | 9.1e-6 |
  | WHT | 3.2e-8 | 3.2e-7 |
  | VAR_RNOISE | 6.4e-8 | 5.1e-7 |
  | VAR_FLAT | 6.8e-8 | 1.0e-6 |
  | VAR_POISSON | 6.7e-8 | 7.6e-7 |
  | ERR | 3.5e-8 | 3.1e-7 |

  CON is identical, and no pixel is NaN in only one of the two. Against
  drizzle 2.2.0 expect the same order: float32 round-off. drizzle 2.2.0's
  blot interpolates in single precision, so blot differs from spizzle by up
  to about 1e-5. Outlier flags may then differ at a few pixels near the
  thresholds. Count them, but do not treat them as errors.
- Timing targets, M3 Pro with 11 threads (`benchmarks/speed_table.py`, under
  some background load):

  | step | stock | spizzle |
  | --- | ---: | ---: |
  | pixel map | 0.77 s | 0.020 s |
  | science + 3 variance planes | 0.51 s | 0.24 s |
  | blot of median and error | 0.039 s | 0.0058 s |
  | `ResampleImage` of 8 exposures | 20 s | 4.8 s (2.7 s quiet) |

  The two `gwcs_blot` calls of one exposure take 1.6 s; spizzle's pixel map
  plus two blots take 0.03 s.
- The noise factor plane in the output product: an array that is not in
  the `ImageModel` schema is saved only in the ASDF extension, not as a
  FITS image (checked 2026-11-03: the file has PRIMARY, SCI and ASDF;
  `datamodels.open` gets it back). The user chose a separate file
  (section 0).
- Measured 2026-11-03 with `scripts/compare_sparse_resample.py` in the
  `snowblind` env (jwst 3.0.0, stcal 1.20.0, drizzle 2.2.0, spizzle main
  with the context option), on the 8 NRCALONG exposures of 6154 above
  (4897x2858 frame, 13.8 M pixels with weight), M3 Pro:
  - time: jwst 18.4 s; `variance="stcal"` with every pixel's WCS 6.7 s;
    the defaults (spline pixel maps, exact variance, second pass for the
    noise factor) 5.4 s;
  - `variance="stcal"` against jwst, relative difference median / 99.9% /
    max: SCI 7.0e-8 / 3.4e-7 / 2.3e-5, WHT 0 / 2.0e-7 / 3.4e-7,
    VAR_RNOISE 8.6e-8 / 4.3e-7 / 6.9e-7, VAR_FLAT 9.1e-8 / 4.4e-7 /
    1.1e-6, VAR_POISSON 9.0e-8 / 4.4e-7 / 8.3e-7, ERR 0 / 2.2e-7 /
    3.8e-7; no pixel NaN in only one; CON identical. SCI's largest
    differences are at values near zero (float32 drizzle);
  - the defaults against jwst: SCI as above, WHT max 2.3e-4 (spline pixel
    maps);
  - median (5-95%): noise factor 2.35 (1.58-3.44); jwst's summed VAR over
    the exact one 2.25 (1.49-3.29); over the exact one times the noise
    factor 0.967 (0.838-1.040); jwst's ERR over the new ERR 0.984
    (0.915-1.020). As section 5.1 expects.

Done (2026-11-03, Part B, uncommitted on `feature-resample-spizzle`):

- `SparseResampleStep` (`sparse_resample`, in `resample.py`): its own spec
  (jwst's without `kernel` and `propagate_dq`; `weight_type` ivm or exptime;
  `enable_ctx` False; `variance`; `pixmap_stepsize` integer 16 and
  `pixmap_order` 3) and its own `process`, jwst's input handling with
  `SparseResampleImage`. The noise factor is saved with
  `self.save_model(..., suffix="nfactor")`, as source_catalog saves
  `_segm.fits`, when results are saved; `Step.record` writes `variance`
  (`VARTYPE`) and `noise_factor_file` (`NFACFILE`) under
  `meta.snowblind.sparse_resample`, and `set_cal_step` `S_SPARSE`.
- `SparseOutlierDetectionStep` (`sparse_outlier_detection`, in
  `outlier_detection.py`): jwst's imaging `detect_outliers` copied with
  `SparseResampleImage` and a `flag_resampled_model_crs` that blots with
  spizzle; raises for modes other than imaging; `S_SPOUTL`.
- Neither step subclasses jwst's: stpipe merges the specs of all classes
  in a step's hierarchy (`config_parser.get_merged_spec_file`), so a
  subclass would accept jwst's `kernel` and the IFU and TSO parameters
  again. The outlier step calls jwst's `_guess_mode` and `_get_asn_id`
  unbound; `_flag_resampled_model_crs` and `_fileio.save_blot` come
  through `_jwst_imports.py` (both exist from jwst 2.0.0).
- `Image3Pipeline` (`snowblind_image3`, in `image3.py`): jwst's with the two
  slots replaced. On synthetic exposures it writes `<product>_i2d.fits`,
  `<product>_nfactor.fits` and the `_crf` files.
- In imaging mode jwst blots once per exposure, not twice: it passes no
  median error, so the flagging uses the exposure's own ERR (section 2's
  two `gwcs_blot` calls are the spectral path).
- Measured 2026-11-03 with `scripts/compare_sparse_outlier_detection.py`
  (the `snowblind` env, no CRDS step parameters), 8 exposures of 6154, M3
  Pro. Time / pixels newly flagged OUTLIER / flagged by only one of it and
  jwst:
  - NRCALONG: jwst 21.2 s / 48647; sparse, every pixel's WCS 13.2 s /
    48649 / 16; sparse defaults 9.2 s / 48649 / 16;
  - NRCA1: jwst 28.2 s / 13259; sparse, every pixel's WCS 19.8 s / 13263 /
    10; sparse defaults 15.9 s / 13263 / 10.

  About one pixel in 2-3 million differs (drizzle 2.2.0's single-precision
  blot, as expected); the median and the flagging are jwst's code, which
  limits the speed-up.
- Tests: `tests/test_resample.py` (step), `tests/test_outlier_detection.py`
  (flags the injected cosmic ray and nothing else; same flags as jwst with
  every pixel's WCS), `tests/test_image3.py` (an end-to-end run). Docs:
  `docs/resampling.rst`, README, CHANGES.


## 5. Things learned that matter here

1. stcal's variance does not propagate through the drizzle weights. It
   drizzles `sqrt(var)` as an image, squares it, and combines the exposures
   with `1 / resampled VAR_RNOISE` weights. On the 8 exposures (native
   scale, pixfrac 1):
   - stcal's total VAR is 2.23x the exact pixel variance (5-95%:
     1.55-3.22);
   - it is 0.97x the exact variance times the noise factor (0.84-1.04).

   So today's ERR already roughly equals `sqrt(var * noise factor)`, and the
   new default ERR will be close to it on average, and locally right. The
   exact VAR planes will be about 2.3 times smaller than today's. Say so in
   any comparison, and expect it in downstream steps that read VAR.
2. The noise factor on that mosaic has a median of 2.33 (5-95%: 1.65-3.38).
   Casertano et al. (2000) give 2.25 everywhere for pixfrac/scale = 1.
3. Aperture photometry: the variance alone underestimates aperture errors
   by 25-33%. Variance times the noise factor overestimates them by 13% at
   1.5-2 pixel radius, 4.5% at 5, 1% at 20. stcal's VAR is close to the
   latter, with more scatter, and 5% low at 20 pixels. The exact aperture
   variance needs the correlation images; it is planned in spizzle for
   later (plan step 4).
4. drizzle differences that show up in comparisons
   (`~/dev/spizzle/drizzle-issues.md`):
   - Issue 1: drizzle drops input pixels centered outside the output frame,
     so edge pixels get less weight. jwst's default frame encloses every
     input on the NIRCam test, so stock and spizzle agree there. It does not
     always: on synthetic exposures rotated by 0.4 degrees (2026-11-03,
     `tests/test_resample.py`), a few input pixel centers landed 0.03 pixel
     outside jwst's frame, and two edge pixels got less weight from drizzle.
     The tests compare inside the outermost ring of the frame.
   - Issue 7: drizzle's blot accepts `[0, n)` instead of the image's
     `[-0.5, n - 0.5)`. spizzle uses the image's extent, so blot differs
     within half a pixel of the median's border. On the NIRCam test no
     pixel changed.
   - Issue 12: drizzle takes fill values as strings. jwst's default
     `fillval="NAN"` is its string; do not pass strings to spizzle.
5. Memory and speed figures for planning are in sections 3 and 4.


## 6. Part C preview: a better outlier detection (not yet)

The stock scheme was chosen for its simplicity, not its accuracy (user):
- each exposure drizzled alone (square kernel, pixfrac 1, input pixel
  scale);
- the median of those images;
- the median blotted back bilinearly;
- jwst's two-pass flagging rule.

JWST PSFs are undersampled, as is usual for space telescopes, and that
makes the comparison hard. The outlier step compares each exposure with a
prediction of it, the reference image blotted back. Where the prediction
misses real structure, such as a star's peak, good pixels look like
outliers, unless the threshold is loosened, and a looser threshold also
hides real outliers. The design goals (user):

1. flag real outliers, including those on or next to stars;
2. do not flag good pixels that only look like outliers: Poisson noise in
   bright sources makes spikes;
3. do not flag the cores of undersampled stars.

The proposal: build the reference from all exposures at once, on a grid
finer than the input pixels. For each output pixel, take the median of the
input pixels of every exposure that land on it, weighted by their overlap,
then a clipped mean of them, and optionally one correction step (the
exposures' residuals combined the same way and added).

`scripts/exp_outlier_reference.py` simulates this with stpsf stars. The
setup:
- NIRCam F070W, the most undersampled NIRCam filter, on NRCA1: stpsf
  2.3.0's `OVERDIST` PSF, summed over each pixel. Its FWHM is 1.2 pixels,
  and a centered star puts 33% of its light in one pixel.
- 160 stars on 192x192 pixels, in 4 exposures offset by random sub-pixel
  shifts: no rotation or distortion between them, no saturation, no IPC.
- The script's docstring says how to get stpsf and its 184 MB of data. The
  run below was made in a scratch environment with Python 3.14, stpsf
  2.3.0 and spizzle 0.1.

Without noise, the error of the predicted brightest pixel of each star, as
a fraction of its true value (typical, and 5-95% of stars):

| reference image | typical | 5-95% |
| --- | ---: | ---: |
| today | 47% | 22% to 57% |
| median of input pixels, grid 2/3 pixel, pixfrac 1 | 24% | 1% to 54% |
| median of input pixels, grid 1/2 pixel, pixfrac 1 | 8% | -14% to 42% |
| clipped mean, grid 1/2 pixel, pixfrac 0.8, plus correction | 3% | -14% to 26% |
| median, grid 1/2 pixel, pixfrac 0.6, plus correction | 0% | -8% to 11% |

With noise and cosmic rays the setup is:
- noise is 5 e- plus Poisson; the stars have 1e3 to 1e6 e-;
- cosmic rays hit 0.5% of the pixels;
- three random realizations;
- pixels are flagged with jwst's rule (snr 5 and 4), whose tolerance
  `scale * D` loosens the threshold where the reference changes steeply;
- a cosmic ray counts only if it is at least 8 times the noise at its
  pixel, so in principle detectable.

| reference image | tolerance | cosmic rays found | within 3 px of a star | star-core pixels flagged | good wing pixels flagged per 10^4 | blank-sky pixels flagged per 10^4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| today | 1.2 / 0.7 (jwst's) | 88% | 48% | 16% | 1 | 0.0 |
| today | 0.5 / 0.3 | 93% | 61% | 41% | 152 | 2.0 |
| clipped mean, grid 1/2, pixfrac 1 | 1.2 / 0.7 | 86% | 47% | 0.3% | 0 | 0.0 |
| clipped mean, grid 1/2, pixfrac 1 | 0.5 / 0.3 | 93% | 61% | 6.5% | 8 | 1.7 |
| clipped mean, grid 1/2, pixfrac 0.8, plus correction | 0.5 / 0.3 | 92% | 60% | 2.8% | 1 | 1.3 |
| clipped mean, grid 1/2, pixfrac 1 | 0.3 / 0.2 | 95% | 68% | 13% | 48 | 3.8 |
| clipped mean, grid 1/2, pixfrac 0.6, plus correction | 0.5 / 0.3 | 70% | 45% | 0.0% | 0 | 0.4 |

What it shows:
- Today's rule flags 16% of the pixels within a pixel of a star's center
  on F070W, consistent with what the user sees on real data. Lowering its
  tolerance to find more cosmic rays raises that to 41%, and floods the
  star wings with flags.
- With a sharper reference the tolerance can drop. At 0.5 / 0.3 about 5
  points more cosmic rays are found than with today's default (12 points
  more near stars), and 3-7% of star-core pixels are flagged instead of
  16%.
- The sharpest reference backfires. With pixfrac 0.6 and the correction
  step, so few input pixels land on each output pixel that an exposure's
  own cosmic ray partly gets into its own prediction, and 30% of the cosmic
  rays are missed. A half-pixel grid with pixfrac 0.8-1.0 is the balance in
  this test.
- Noise spikes on blank sky are flagged at 0-4 per 10^4 pixels in every
  row: plain noise is not the problem.
- With 173 cosmic rays near stars, the near-star percentages are uncertain
  by about +-4 points.
- The cosmic-ray rate is probably higher than what the jump step leaves.

An earlier test with Gaussian stars on blank sky (in spizzle's plan, step
6) also showed the limits:
- on smooth sky today's blurred reference does as well or better;
- the finer grid lets a cosmic ray of one exposure leak into the
  reference, and so into the other exposures' predictions;
- rebuilding the reference without the flagged pixels removes most of
  those leaks.

With only two exposures no combination rejects outliers reliably; minmed is
possible but its noise is doubtful (user). Real data (NIRCam short and long
wavelength, MIRI, with injected cosmic rays) will decide.

The spizzle components for this are planned as spizzle plan step 6:
- the input pixels of several exposures per output pixel;
- weighted median and clipped mean;
- the scatter of those input pixels, as a measured tolerance to replace
  `scale * D`;
- recomputing the reference without flagged pixels.

The simulation's own versions of the first two, in numba, are in the
script. Methods without drizzle are listed in spizzle's plan too. Leave
all of this until Parts A and B are done and the user starts it.

Part C, 2026-11-03: the step is to be named `PairwiseOutlierStep` (user).
Only the name was settled; the method is the one above and in spizzle's
plan step 6: a reference from all exposures' input pixels on a finer
grid, predicted onto each exposure with spizzle's `blot`.

Per-exposure predictions, discussed 2026-11-03. The user's idea: to flag
outliers in one exposure, blot every exposure that overlaps it onto its
frame, combine them optimally, compare, and repeat for each exposure: low
peak memory and parallel over exposures. Simulated with
`scripts/exp_outlier_reference.py --per-exposure` (spizzle's blot only;
same field as above), against today's scheme and the global fine-grid
clipped mean. The per-exposure schemes:
- pairwise: the other exposures blotted onto the exposure, combined by
  their median, a clipped mean weighted by inverse variance, or a clipped
  mean weighted toward the nearly aligned exposures (blurred least);
- leave-one-out: the fine-grid clipped mean of the other exposures' input
  pixels only, blotted onto the exposure;
- pairwise on a 2x frame (the user's second idea): the other exposures
  blotted onto the exposure's own frame sampled twice as finely, their
  median, averaged 2x2 back onto the exposure's pixels.

Error of the predicted brightest pixel of each star, without noise
(median over stars):

| prediction | 4 exposures | 8 exposures |
| --- | ---: | ---: |
| today | 47% | 47% |
| global clipped mean, scale 0.5, pixfrac 1 | 17% | 25% |
| same, plus correction | 5% | 12% |
| pairwise, median | 22% | 32% |
| pairwise, inverse variance | 28% | 33% |
| pairwise, toward aligned exposures | 21% | 27% |
| pairwise on a 2x frame, averaged back | 36% | 39% |
| leave-one-out, pixfrac 1 / 0.8 / 0.6 | 21 / 15 / 11% | 28 / 23 / 20% |

With noise and cosmic rays (as above). At jwst's tolerance 1.2 / 0.7 every
scheme finds the same cosmic rays as today (86-88%, 46-53% near stars);
star-core pixels flagged: today 15.8% (4 exposures) and 13.9% (8);
pairwise 1.2% and 0.6%; pairwise on a 2x frame 1.1% and 2.3%;
leave-one-out 1.0% and 0.1%; global 0.3% and 0.0%. At 0.5 / 0.3 (found / near stars / sky, next to another exposure's
cosmic ray and wings per 10^4 / cores):

| prediction | 4 exposures | 8 exposures |
| --- | --- | --- |
| today | 92.8% / 61% / 2.0, 11, 152 / 41% | 93.3% / 66% / 4.2, 2, 142 / 39% |
| global clipped mean | 92.9% / 61% / 1.7, 2, 8 / 6.5% | 93.7% / 69% / 4.8, 3, 3 / 5.0% |
| global, plus correction | 92.7% / 60% / 1.4, 1, 3 / 3.3% | 93.5% / 67% / 3.9, 3, 1 / 0.5% |
| pairwise (all three combinations) | 93.5% / 61% / 2.5, 18, 14 / 7.6% | 93.6% / 64% / 2.5, 1, 15 / 16% |
| pairwise on a 2x frame | 93.7% / 62% / 2.3, 21, 16 / 23% | 93.5% / 64% / 2.6, 1, 29 / 27% |
| leave-one-out, pixfrac 1 | 93.2% / 61% / 3.6, 7, 15 / 9.4% | 94.1% / 69% / 5.4, 3, 5 / 12% |
| leave-one-out, pixfrac 0.8 | 93.1% / 60% / 2.9, 94, 10 / 7.4% | 93.9% / 68% / 5.2, 3, 4 / 5.9% |
| leave-one-out, pixfrac 0.6 | 92.7% / 60% / 2.4, 307, 7 / 5.9% | 93.7% / 67% / 5.2, 3, 4 / 4.3% |

What it shows:
- Pairwise fixes the star cores at jwst's tolerance (1% or less instead of
  14-16%) at the same cosmic-ray rate, with no new spizzle parts. It does
  not allow a lower tolerance: cores then go to 8% (4 exposures) and 16%
  (8). More exposures make it worse, not better: the median of more
  blurred copies tends to the typical blur. The weighting makes no
  difference.
- Finding more cosmic rays, near stars especially, needs the lower
  tolerance, and there a reference built from the input pixels is better:
  the global one with the correction step flags 0.5% of cores with 8
  exposures. Part of its advantage is that it includes the exposure's own
  pixels.
- Leave-one-out keeps the per-exposure structure. With 8 exposures it is
  close to the global reference without the correction (4-6% of cores at
  pixfrac 0.6-0.8, no leaks); with 4, too few input pixels remain, and
  below pixfrac 1 other exposures' cosmic rays leak into it (94 and 307
  per 10^4). Its correction step is not tried.
- Blotting onto a finer frame adds no information: each blotted copy is
  the bilinear surface through one exposure's own samples, and averaging
  2x2 back blurs once more. What locates the flux on a finer grid is the
  input pixels with their footprints (the drizzle operator), combined per
  fine pixel: the leave-one-out reference, or with the exposure's own
  pixels included, the global reference computed over each exposure's
  footprint.
- A cubic spline in place of the blot did better for pairwise (1.6% of
  cores at 0.5 / 0.3 with 4 exposures) but is 52 times slower than
  spizzle's blot (232 ms against 4.5 ms per NIRCam frame); not in the
  script.
- The global reference can also be built tile by tile over the mosaic,
  which bounds its memory and runs in parallel.


Prototype on real data, 2026-11-03 (user: prototype in snowblind, test
with injected cosmic rays). `src/snowblind/outlier_reference.py`: for each
exposure, the input pixels of the exposures that overlap it (found from
their footprints) are drizzled onto its own frame sampled twice as finely
(pixfrac 0.8), after each one's sky level is subtracted (skymatch's level,
or the sigma-clipped median); per fine pixel their weighted median, then
the 3-sigma clipped mean; the result blotted onto the exposure (on that
frame, the 2x2 mean) and flagged by jwst's rule (stcal's
`flag_resampled_crs`). Where fewer than three exposures cover a pixel, only
pixels above the prediction are flagged (user). The fine frame is done in
strips of 512 rows: 6-7 s and 2.9 GB per NIRCam exposure on the laptop.

The exposure's own pixels have to be left out (leave-one-out). Real
coverage is low: on 6154 the eight dithers of one detector cover most of
an exposure only once or twice more, and with all 64 short-wavelength
exposures 35% of an exposure's pixels are covered by one other exposure
only. With its own pixels in, the weighted median of two exposures often
takes the exposure's own value, cosmic ray included: 14 pixels flagged on
an NRCA1 exposure where jwst flags 1625.

Test: `scripts/exp_injected_outliers.py` on the 16 long-wavelength cal
files of 6154 (F277W, NRCALONG and NRCBLONG, MAST products), 2000 single-
pixel cosmic rays injected per exposure, 10-1000 times the local noise,
half 1-3 px from one of the 300 brightest sources. Coverage of the pixels
(exposures including the exposure itself): 1: 8.8%, 2: 41.4%, 3: 3.9%, 4:
45.9%. Flagged "elsewhere" counts the pixels flagged that were not injected
(more than a pixel from any injected one): real outliers as well as false
flags, the same for every method.

| method | time | found | near sources | found where 1 / 2 / 3 / 4 cover | not injected, flagged: source cores | wings per 10^4 | elsewhere per 10^4 |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| jwst's step | 67 s | 63.0% | 39.9% | 31 / 60 / 67 / 72% | 2.23% | 26 | 13.8 |
| prototype, 1.2 / 0.7 | 96 s | 66.8% | 46.5% | 0 / 73 / 72 / 75% | 0.29% | 25 | 16.1 |
| prototype, 1.0 / 0.6 | 96 s | 68.7% | 49.7% | 0 / 75 / 74 / 77% | 0.40% | 32 | 18.7 |
| prototype, 0.8 / 0.5 | 96 s | 71.2% | 54.0% | 0 / 77 / 79 / 80% | 1.19% | 45 | 23.7 |
| prototype, 0.5 / 0.3 | 96 s | 75.8% | 62.4% | 0 / 83 / 83 / 85% | 6.19% | 88 | 41.9 |
| prototype with correction, 1.2 / 0.7 | 140 s | 66.7% | 46.4% | 0 / 73 / 72 / 75% | 0.26% | 24 | 15.5 |
| prototype with correction, 0.5 / 0.3 | 140 s | 75.7% | 62.2% | 0 / 83 / 83 / 85% | 5.12% | 84 | 40.1 |

- Where another exposure covers the pixel, the prototype at jwst's
  tolerance finds 73-75% of the injected cosmic rays against jwst's
  60-72%, more near sources, and flags 0.29% of bright-source cores
  against 2.23%; wings and elsewhere about the same.
- Where only the exposure itself covers (8.8%), it finds none. jwst finds
  31% there: its median is then the exposure itself drizzled and blotted,
  a blurred copy, so it flags pixels that stand out from their neighbors,
  star cores among them.
- Lowering the tolerance finds more (85% where four cover, 62% near
  sources at 0.5 / 0.3) at the cost of more flags at cores, in wings and
  elsewhere; 1.0 / 0.6 or 0.8 / 0.5 look like the balance. Whether the
  extra flags elsewhere are false or real outliers is not known.
- The correction step changes little here, for 45% more time.
- To do: short-wavelength (64 exposures), the single-coverage pixels, and a
  look at the extra flags.


Course correction, 2026-11-03 (from the spizzle session, user): the
per-exposure variants are dropped. The design: one global reference from
all exposures' input pixels on the outlier mosaic's grid sampled twice as
finely, built in tiles (1024 fine pixels plus a halo of 4) from geometry
computed once per exposure, never a mosaic-sized array; weighted by overlap
times ivm; buffered weighted median, clipped mean, correction step; fixed
defaults (2x, pixfrac 0.8, correction on, snr 5/4, below 3 exposures only
pixels above the prediction); and a tolerance measured from the data in
place of `scale * D` if it works for every filter.

Tolerance, simulated with `scripts/exp_outlier_reference.py --part 2
--defaults` (4 exposures, 160 stpsf stars; NRCA1 for F070W and F150W,
NRCA5 for F277W and F444W). Found / near stars / sky, wings per 10^4 /
star cores flagged:

| filter (FWHM) | jwst's step, 1.2 / 0.7 | defaults, `scale * D` 0.5 / 0.3 | defaults, measured | defaults, robust |
| --- | --- | --- | --- | --- |
| F070W (1.21 px) | 87.6% / 48% / 0.0, 1 / 15.8% | 92.0% / 60% / 1.3, 1 / 2.8% | 98.8% / 95% / 18.5, 489 / 52% | 88.4% / 57% / 6.6, 34 / 0.8% |
| F150W (1.76 px) | 86.3% / 49% / 0.0, 0 / 3.1% | 91.8% / 63% / 2.1, 1 / 0.2% | 98.8% / 95% / 14.5, 578 / 39% | 88.4% / 60% / 6.5, 38 / 0.7% |
| F277W (1.69 px) | 87.2% / 49% / 0.0, 0 / 3.8% | 92.2% / 65% / 1.1, 1 / 0.2% | 98.7% / 96% / 6.8, 549 / 41% | 89.4% / 67% / 2.5, 32 / 0.7% |
| F444W (2.45 px) | 85.3% / 49% / 0.0, 0 / 0.0% | 90.6% / 60% / 1.8, 0 / 0.0% | 98.6% / 92% / 10.6, 636 / 24% | 88.6% / 61% / 5.5, 35 / 0.6% |

- "measured": the weighted variance of the input pixels kept by the
  clipping minus their mean err**2. The clipping removes the scatter that
  structure makes, so it is near zero at star cores: fails everywhere.
  Taken over all input pixels instead it would include a cosmic ray's own
  deviation (about 0.43 of its amplitude with 4 exposures) and hide it.
- "robust": (1.4826 times the weighted MAD of all input pixels about the
  clipped mean)**2 minus their weighted median err**2. Cores are fine, but
  it finds fewer cosmic rays than `scale * D` at 0.5 / 0.3 and flags 30
  times more wing pixels and 2.5-6.6 sky pixels per 10^4 (target 5): on
  sky the median spread of a few input pixels is often zero, so the
  prediction's own noise is not counted, which `scale * D` partly is.
- `scale * D` at 0.5 / 0.3 with the defaults meets the targets on all four
  filters with the same values: cores 0-2.8%, more cosmic rays found than
  jwst's step (and 11-16 points more near stars), sky 1.1-2.1 per 10^4.
  That is the default, then; the robust spread is kept as an option.

Tiled implementation (`src/snowblind/outlier_reference.py`, uncommitted):
per exposure once, the pixel map onto the 2x grid, its corners (kept as
float32 offsets from the float64 centers, exact to about 1e-7 fine pixel,
half the memory), Jacobian, ivm weights, sky level (sigma-clipped median of
200 000 good pixels when skymatch did not run), and the regions of input
pixels reaching each tile (from the centers every 8 pixels, widened by 8
plus a margin of 3). Per tile, the operators of those regions only
(corners shifted by the tile's origin, `overlap_triplets` and
`build_operator`), one numba kernel looping over the parts (buffered
weighted median, clipped mean, robust spread, count of exposures), the
correction from the blotted residuals, and the predictions of the pixels
whose centers lie in the tile. Tests: prediction independent of the tile
size (16 against 1024 fine pixels), cosmic ray flagged, nothing flagged
without one. On the 8 nrcalong files of 6154: 13.8 s (geometry 2.9 s, 60
tiles 10.1 s, flags 0.9 s) and 3.2 GB peak, of which about 310 MB per
exposure; per tile, spizzle's operators take 4.6 s of the 10.1, the
combination 1.8 s and the correction 1.8 s. Target was under 10 s.

Injected cosmic rays on the 16 long-wavelength files of 6154 (as above;
coverage 1: 8.8%, 2: 40.2%, 3: 4.9%, 4+: 45.8%):

| method | time | found | near sources | found where 1 / 2 / 3 / 4+ cover | source cores | wings per 10^4 | elsewhere per 10^4 |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| jwst's step | 67 s | 63.0% | 39.9% | 29 / 60 / 67 / 72% | 2.23% | 26 | 13.8 |
| tiled, 1.2 / 0.7 | 30 s | 38.6% | 26.0% | 0 / 8 / 60 / 73% | 0.05% | 6 | 5.0 |
| tiled, 0.8 / 0.5 | 30 s | 41.2% | 29.9% | 0 / 10 / 64 / 77% | 0.17% | 11 | 6.6 |
| tiled, 0.5 / 0.3 | 30 s | 44.5% | 34.6% | 0 / 13 / 68 / 81% | 0.74% | 19 | 9.7 |
| tiled, robust | 30 s | 52.6% | 36.6% | 3 / 45 / 61 / 69% | 4.51% | 34 | 13.6 |

- Where 4 or more exposures cover, the tiled reference at 0.5 / 0.3 finds
  81% against jwst's 72%, with a third of jwst's flags at source cores and
  fewer in wings and elsewhere.
- Where 2 cover, it finds 8-13%: the reference includes the exposure's
  own pixels, and the weighted median of two exposures takes whichever has
  more weight at that fine pixel, often the one with the cosmic ray. The
  per-exposure test found 73-83% there by leaving the exposure out.
- Where 1 covers, nothing can be found; jwst finds 29% by comparing the
  exposure with a blurred copy of itself.


Decisions, 2026-11-03 (user):
- `scale * D` at 0.5 / 0.3 is the default tolerance; check that it holds
  for the other NIRCam filters up to F444W, NIRISS and NIRSpec (MIRI
  eventually).
- Two exposures: no special handling beyond flagging only pixels above
  the prediction. Most outliers should be gone after snowblind_detector1;
  this step is cleanup.
- Real data: MAST program 2234 (NIRCam F090W short wavelength, F444W with
  F470N long wavelength, 10 fields, four small-offset dithers each);
  observation 1's 40 cal files (4.7 GB) in `~/data/jwst_2234/o001`.


The default tolerance on more filters, simulated the same way (`--part 2
--defaults --instrument ... --filter ... --detector ...`; NRCA1 or NRCA5,
NIS, NRS1). Found / near stars / sky per 10^4 / star cores flagged, jwst's
step at its tolerance against the defaults at `scale * D` 0.5 / 0.3:

| instrument, filter (FWHM) | jwst's step, 1.2 / 0.7 | defaults, 0.5 / 0.3 |
| --- | --- | --- |
| NIRCam F090W (1.30 px) | 88.0% / 47% / 0.0 / 13.4% | 92.0% / 59% / 1.0 / 2.2% |
| NIRCam F115W (1.46 px) | 87.2% / 44% / 0.0 / 9.2% | 91.1% / 60% / 0.7 / 0.8% |
| NIRCam F200W (2.23 px) | 86.3% / 52% / 0.0 / 0.3% | 91.1% / 62% / 1.7 / 0.0% |
| NIRCam F356W (2.05 px) | 86.2% / 51% / 0.2 / 0.8% | 92.6% / 65% / 0.8 / 0.0% |
| NIRISS F090W (1.24 px) | 87.7% / 44% / 0.0 / 14.5% | 93.5% / 64% / 0.1 / 2.9% |
| NIRISS F115W (1.26 px) | 88.5% / 45% / 0.0 / 13.6% | 93.4% / 64% / 0.6 / 2.6% |
| NIRISS F150W (1.30 px) | 87.8% / 45% / 0.0 / 11.9% | 92.5% / 60% / 0.6 / 2.2% |
| NIRISS F200W (1.40 px) | 87.4% / 46% / 0.0 / 9.5% | 92.1% / 59% / 0.6 / 1.4% |
| NIRISS F277W (1.65 px) | 87.3% / 54% / 0.0 / 4.9% | 92.0% / 64% / 0.7 / 0.2% |
| NIRISS F356W (1.97 px) | 86.5% / 53% / 0.5 / 1.2% | 92.4% / 68% / 2.3 / 0.0% |
| NIRISS F444W (2.33 px) | 86.2% / 52% / 0.6 / 0.0% | 91.5% / 63% / 2.5 / 0.0% |
| NIRSpec F110W (1.25 px) | 90.1% / 44% / 0.0 / 15.6% | 94.2% / 64% / 0.0 / 2.9% |
| NIRSpec F140X (1.25 px) | 90.0% / 44% / 0.0 / 15.1% | 94.2% / 64% / 0.0 / 2.8% |

With F070W, F150W, F277W and F444W above, the same defaults meet the
targets on all 17: star cores 0-2.9% (the most undersampled, NIRISS F090W
and NIRSpec F110W, are closest to 3%), more cosmic rays than jwst's step
(4-6 points; 10-20 points near stars), sky 0-2.5 per 10^4, star wings 0-2.
The robust spread keeps cores under 1% everywhere but flags 16-37 wing
pixels per 10^4 and 3.6-7.8 sky pixels for the long-wavelength filters.
MIRI not yet simulated.


Program 2234, 2026-11-03: `exp_injected_outliers.py` on observation 1
(INTRAMODULEBOX, four dithers of about 3 arcsec, MEDIUM8 10 groups, 1052
s), 2000 cosmic rays per exposure (10-1000 sigma), up to half 1-3 px from
sources above 20 sigma (`--source-snr 20`: the fields are sparse, with 4
to 26 sources above 50 sigma per detector; the first injection loop could
not place 1000 near-source cosmic rays there and never ended; it now
places as many as fit). Found / near sources / found where 1, 2, 3, 4+
exposures cover / wings and elsewhere per 10^4:

| set | method | time | found | near sources | by coverage | wings, elsewhere |
| --- | --- | ---: | ---: | ---: | --- | --- |
| F444W+F470N, 8 (coverage 4+: 83%, 2: 9%) | jwst's step | 25 s | 95.7% | 84.9% | 44 / 83 / 96 / 97.4% | 72, 55.6 |
| | tiled, 0.5 / 0.3 | 14 s | 88.5% | 83.3% | 2 / 27 / 84 / 95.8% | 80, 55.3 |
| | tiled, 1.2 / 0.7 | 14 s | 86.2% | 77.7% | 2 / 21 / 80 / 94.1% | 53, 21.3 |
| F090W, NRCA1-4, 16 (4+: 81%, 2: 15%) | jwst's step | 47 s | 93.2% | 79.5% | 41 / 85 / 91 / 95.4% | 28, 2.7 |
| | tiled, 0.5 / 0.3 | 28 s | 84.8% | 77.3% | 0 / 20 / 79 / 97.3% | 33, 2.4 |
| | tiled, 1.2 / 0.7 | 29 s | 83.3% | 70.7% | 0 / 16 / 77 / 96.4% | 22, 1.4 |

- Where four or more exposures cover, the tiled reference finds about as
  many as jwst's step (97.3% against 95.4% in F090W, 95.8% against 97.4%
  in the noisy narrowband F470N).
- Where two cover, jwst finds 83-85%: its median of two images is their
  mean, so a cosmic ray shows at half its amplitude. The tiled weighted
  median takes one exposure's value, often the one with the cosmic ray:
  16-27%. That is most of the difference overall (93-96% against 84-88%).
  Using the weighted mean of the two exposures where only two reach a fine
  pixel would match jwst there (asked the user).
- The "source cores" measure is not usable on these fields: with sources
  down to 20 sigma many are hot pixels or cosmic-ray remnants, flagged by
  every method alike (28% in F470N, 11% in F090W). The real-data
  comparison of star cores is 6154's (0.74% against jwst's 2.23% where four
  or more cover).
- 1.7-1.8 times faster than jwst's step; 4.1 GB peak.


Two exposures, 2026-11-03 (user: weighted mean): where fewer than three
exposures land on a fine pixel, the reference is their weighted mean, as
jwst's median of two images is, which halves a cosmic ray; and the
correction step is left out there, because with two exposures it pulls
the reference toward both, the cosmic ray included (a test with two
exposures failed until then). Rerun at `scale * D` 0.5 / 0.3 (found / near
sources / found where 2 cover / source cores):

| set | jwst's step | tiled, before | tiled, now |
| --- | --- | --- | --- |
| 2234 F444W+F470N, 8 | 95.7% / 84.9% / 82.7% | 88.5% / 83.3% / 27.0% | 92.2% / 86.6% / 68.9% |
| 2234 F090W NRCA1-4, 16 | 93.2% / 79.5% / 84.6% | 84.8% / 77.3% / 20.4% | 92.1% / 82.1% / 70.8% |
| 6154 long wavelength, 16 | 63.0% / 39.9% / 60.1% / 2.23% | 44.5% / 34.6% / 12.7% / 0.74% | 62.0% / 47.8% / 54.4% / 0.56% |

Overall now within 1-4 points of jwst's step, 2-8 points more near
sources, a quarter of its star-core flags in 6154, fewer wing and other
flags there, and 1.6-2.4 times faster. Where two cover it finds 54-71%
against jwst's 60-85%: the sharper reference shows the halved cosmic ray
more steeply, which raises `D`; at 1.2 / 0.7 only 5-16% are found there,
so 0.5 / 0.3 is the default for this reason too. Where one covers,
nothing changes (0-5% against 29-44%; 1% of the pixels in 2234, 9% in
6154).


## 7. Open questions for the user (ask, do not decide)

Most are answered in section 0 (2026-11-03); the pixel maps use spizzle's
defaults (`pixmap_stepsize=16, pixmap_order=3`), and the sanity checks 1
and 1. Still open:
- the noise factor file's suffix: proposed `nfactor` (or `nfac`); not in
  jwst's known suffixes (`jwst.lib.suffix.KNOW_SUFFIXES`), so no clash.

The original list:

- How should the noise factor plane be stored and named in the product, and
  should the optional correlation file be written by the step, under what
  name?
- Pixel maps: use spizzle's splines (`stepsize=16, order=3`) by default, or
  follow the step's `pixmap_stepsize`/`pixmap_order` (stock defaults 1.0 and
  1)?
- `propagate_dq=True`: fall back to the stock class, or raise?
- Second pass: keep the operators in memory, or rebuild them?
- Step names, class aliases, and how the steps hook into pipelines.
- At pixel scale ratios other than 1, or pixfrac < 1, stcal's VAR has not
  been compared with the exact variance; measure it before claiming
  anything there.


## 8. References

- `~/dev/spizzle/drizzle-sparse-plan.md`, section 11, steps 3 (covariance
  and the aperture tables), 5 (this integration) and 6 (outlier detection).
- `~/dev/spizzle/benchmarks/compare_stcal.py`: the stock `ResampleImage`
  against spizzle; the recipe in section 3.
- `~/dev/spizzle/benchmarks/speed_table.py`, `check_covariance.py`,
  `compare_jwst.py`.
- `~/dev/spizzle/README.md` and `docs/index.rst`: usage of every piece.
- `scripts/exp_outlier_reference.py` (this repository): the stpsf
  simulation of section 6.
- `~/dev/jwst/jwst/resample/resample.py`, `resample_step.py`,
  `jwst/outlier_detection/imaging.py`, `utils.py`.
- `~/dev/stcal/src/stcal/resample/resample.py` and `utils.py`,
  `src/stcal/outlier_detection/utils.py` (read upstream main with `git show
  upstream/main:<path>`; the local main is older).

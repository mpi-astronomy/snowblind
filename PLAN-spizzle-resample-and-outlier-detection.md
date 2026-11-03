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

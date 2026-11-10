Resampling and outlier detection
================================

``calwebb_image3`` drizzles twice: once to find outliers (cosmic rays and
other bad pixels the earlier steps missed), and once to make the mosaic.
``SparseResampleStep`` (``sparse_resample``) and
``SparseOutlierDetectionStep`` (``sparse_outlier_detection``) are the jwst
resample and outlier detection steps for images with drizzle replaced by
`spizzle <https://github.com/mpi-astronomy/spizzle>`_.
``TiledOutlierStep`` (``tiled_outlier``) finds outliers against a sharper
reference, built from the input pixels of all the exposures at once.  The
``snowblind_image3`` pipeline is ``calwebb_image3`` with
``TiledOutlierStep`` and ``SparseResampleStep`` in place
(``--outlier_step=sparse_outlier`` runs ``SparseOutlierDetectionStep``
for the outliers instead). spizzle builds drizzle's square-kernel weights
once per exposure as a sparse matrix and applies it in double precision,
which gives the exact variance of each output pixel and the noise factor of
the mosaic.

Summary
-------

* The science image and weights are the jwst step's, to single-precision
  round-off.
* The variance planes are the exact variance of each output pixel, and a
  separate noise factor image (``<product>_nfactor.fits``) gives what the
  correlation between neighboring output pixels adds to the noise of a sum
  over many of them. ERR is the root of the summed variance planes times
  the noise factor.
* The jwst step's variance planes and ERR are still available, with
  ``variance="stcal"``, to compare with.
* ``SparseOutlierDetectionStep`` is jwst's outlier detection unchanged, with
  spizzle's drizzling and blot.
* ``TiledOutlierStep`` compares each exposure with a reference that resolves
  undersampled star cores, so that its threshold can be tighter: it finds
  more cosmic rays near stars and flags far fewer star cores than jwst's
  method, with the same parameters for every filter.
* All three are faster than the jwst steps, and take their parameters from
  the caller only, not from CRDS.

Installation and running
------------------------

spizzle is an optional dependency::

    pip install 'snowblind[spizzle]'

Run Image3 with the snowblind steps in place:

.. code-block:: bash

    strun snowblind_image3 jw..._image3_asn.json

The pipeline replaces jwst's ``outlier_detection`` slot by two,
``tiled_outlier`` and ``sparse_outlier``, and ``outlier_step`` names the one
that runs. Each step takes its parameters in its own slot, e.g.
``--steps.tiled_outlier.scale="0.5 0.3"``,
``--steps.sparse_outlier.pixfrac=0.8`` or
``--steps.resample.pixel_scale=0.03``. Each step also runs on its own:

.. code-block:: python

    from snowblind import SparseResampleStep

    SparseResampleStep.call("jw..._image3_asn.json", save_results=True)

The variance planes and the noise factor
----------------------------------------

Drizzling makes each output pixel a weighted mean of the input pixels that
overlap it, and with square-kernel weights ``W`` of each exposure its
variance is ``(W * W) @ var / wsum**2`` for each variance plane ``var``,
summed over the exposures: what ``VAR_RNOISE``, ``VAR_FLAT`` and
``VAR_POISSON`` hold by default. Neighboring output pixels share input
pixels, so their noise is correlated, and the variance of a sum over an
aperture is more than the sum of the pixel variances in it. The noise
factor of an output pixel is the sum of its covariance with every output
pixel over its own variance: about the ratio of the two for a large
aperture. The step computes it in a second pass over the exposures and
saves it next to the mosaic, with the file name in the mosaic's
``NFACFILE`` header card
(``meta.snowblind.sparse_resample.noise_factor_file``). ERR is
``sqrt((VAR_RNOISE + VAR_FLAT + VAR_POISSON) * noise factor)``.

For aperture photometry, the variance planes alone underestimate the error
of a sum, and the variance times the noise factor overestimates it for
small apertures, where only part of the correlated neighbors are inside:
on NIRCam exposures at their own pixel scale with ``pixfrac`` 1, the
variance alone is 25-33% low, and the variance times the noise factor about
13% high at a radius of 2 pixels, 4-5% at 5 and 1% at 20.

The jwst step drizzles the square root of each variance plane as an image,
squares it, and combines the exposures with weights from the resampled
read-noise variance, then divides by the pixel scale ratio squared. That is
not the variance of the output pixels: on NIRCam exposures at their own
pixel scale, its summed variance planes are about 2.2 times the exact ones,
and close to the exact variance times the noise factor (16% lower to 4%
higher for 90% of the pixels). Its ERR is therefore close to the new ERR
on average, while its variance planes are more than twice the new ones;
steps or scripts that read the variance planes will see that.
``variance="stcal"`` computes the jwst step's variance planes and ERR
instead, from the same matrices, to compare with the jwst step; the
``VARTYPE`` header card says which the mosaic holds.

How the parameters differ from the jwst steps
---------------------------------------------

* Only the square kernel: there is no ``kernel`` parameter.
* No ``propagate_dq``: the DQ array is not resampled.
* No ``good_bits``: each step names the DQ flags of the pixels it leaves
  out. Resampling leaves out the pixels with DO_NOT_USE or NON_SCIENCE, and
  both outlier steps those with DO_NOT_USE, as the jwst steps do by default.
* No ``resample_data`` in ``SparseOutlierDetectionStep``: it always
  drizzles, which is what it replaces.
* ``weight_type`` is ``ivm`` (inverse variance) unless given. The steps do
  not read the jwst steps' parameters from CRDS, which ask for exposure-time
  weights for some instruments.
* ``enable_ctx`` is off by default: the context array, one 32-bit plane of
  the mosaic's size per 32 exposures, is made only when asked for.
* ``pixmap_stepsize`` 16 and ``pixmap_order`` 3: the WCS is evaluated every
  16 pixels and the pixel map interpolated with bicubic splines between
  them, which for smooth distortions differs from evaluating every pixel by
  about 1e-8 pixel on NIRCam. ``pixmap_stepsize=1`` evaluates
  every pixel, as the jwst steps do by default.
* ``fillval`` must be a number (``NAN`` by default); ``INDEF`` is not
  supported. The variance planes are NaN where there is no weight, whatever
  the fill value, as in the jwst step.
* ``variance``: ``exact`` (the default) or ``stcal``, as above.
* ``SparseOutlierDetectionStep`` and ``TiledOutlierStep`` run on images
  only; spectra, coronagraphy and time series stay with the jwst step.

A parameter the step does not have, such as ``kernel``, is rejected rather
than ignored.

Outlier detection against all exposures' input pixels
-----------------------------------------------------

jwst's outlier detection drizzles each exposure alone onto the mosaic grid,
takes the median of those images and blots it back onto each exposure. JWST
point spreads are undersampled, so each of those images blurs a star's
core, the median cannot use the dithers to sharpen it, and the blot blurs it
again. A star's peak pixel is then predicted too low and looks like an
outlier, which jwst's rule allows for by loosening its threshold where the
reference changes steeply (``scale * D``, ``D`` the largest difference
between a pixel of the reference and its neighbors), and that also hides cosmic
rays on and next to stars.

``TiledOutlierStep`` builds one reference from the input pixels of all the
exposures at their true positions, on the outlier mosaic's grid sampled
twice as finely:

1. each exposure's input pixels are drizzled onto that grid (pixfrac 0.8)
   after its sky level is subtracted (skymatch's level, or the sigma-clipped
   median of its good pixels), weighted by their overlap and their inverse
   read-noise variance (1 without VAR_RNOISE), the pixels with DO_NOT_USE
   left out;
2. for each fine pixel, the input pixels that land on it are combined by
   their weighted median, then their weighted mean within 3 sigma of it;
3. the exposures' residuals from the reference are combined the same way
   and added once;
4. the reference is blotted onto each exposure and compared with it by
   jwst's two-pass rule, with ``scale`` 0.5 / 0.3 instead of jwst's 1.2 /
   0.7.

Where fewer than three exposures land on a fine pixel, the reference there
is their weighted mean, as jwst's median of two images is, the residuals are
not added, and only pixels above the prediction are flagged: with two
exposures no combination can tell which one holds an outlier, and cosmic
rays are positive. Where only one exposure covers a pixel, nothing can be
flagged. jwst's step flags some pixels there by comparing the exposure with
a blurred copy of itself; ``TiledOutlierStep`` does not.

The grid is never allocated whole: it is processed in tiles, each with a
small margin, and for each tile only the input pixels landing on it are
drizzled. Memory therefore depends on the number and size of the exposures,
not on the size of the mosaic. The outliers get DO_NOT_USE and OUTLIER in ``dq`` and NaN in the
data, error and variance arrays, as with jwst's step.

In simulations of dithered star fields with ``stpsf`` point spreads, for the
NIRCam, NIRISS and NIRSpec imaging filters, the same parameters find more
cosmic rays than jwst's step, many more within a few pixels of stars, and
flag a few percent of star-core pixels or fewer where jwst's step flags up
to about 15%. ``scripts/exp_injected_outliers.py`` repeats the comparison
on real data: it injects cosmic rays into a set of ``_cal`` files and runs
both methods on them.

Agreement with the jwst steps
-----------------------------

With every pixel's WCS evaluated and ``variance="stcal"``, the outputs agree
with the jwst step's to single-precision round-off: a relative difference
of about 1e-7 typically and 2e-5 at most in the science image (at values
near zero), about 1e-6 at most in the weights and variance planes, and the
same context. Two differences come from drizzle:

* drizzle leaves out the input pixels whose centers fall outside the output
  frame, so the outermost output pixels can get less weight than spizzle
  gives them, where a rotated exposure reaches just past the jwst frame;
* drizzle's blot interpolates in single precision, so outlier detection
  flags a few pixels near its thresholds differently: on NIRCam exposures,
  about one pixel in a few million.

``scripts/compare_sparse_resample.py`` and
``scripts/compare_sparse_outlier_detection.py`` run a step and its jwst
counterpart on the same ``_cal`` files and report these differences.

Speed
-----

On eight NIRCam exposures on a laptop (11 threads), the mosaic with the
variance planes takes a third of the jwst step's time with every pixel's
WCS evaluated, and less with the defaults, including the second pass for
the noise factor. ``SparseOutlierDetectionStep`` takes about half the jwst
step's time with the defaults (the median of the drizzled exposures and the
flagging are jwst's code and are not faster), and ``TiledOutlierStep`` half
to two thirds of it.

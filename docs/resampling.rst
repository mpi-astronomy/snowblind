Resampling and outlier detection
================================

``calwebb_image3`` drizzles twice: once to find outliers (cosmic rays and
other bad pixels the earlier steps missed), and once to make the mosaic.
``SparseResampleStep`` (``sparse_resample``) and
``SparseOutlierDetectionStep`` (``sparse_outlier_detection``) are the jwst
resample and outlier detection steps for images with drizzle replaced by
`spizzle <https://github.com/mpi-astronomy/spizzle>`_, and the
``snowblind_image3`` pipeline is ``calwebb_image3`` with both in place.
spizzle computes drizzle's square kernel as a sparse matrix for each
exposure, applies it to the science data, the weights and the variance
planes in double precision, and so gives the exact variance of each output
pixel and the noise factor of the mosaic.

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
* Outlier detection is jwst's method unchanged, with spizzle's drizzling
  and blot.
* Both steps are faster than the jwst steps, and take their parameters from
  the caller only, not from CRDS.

Installation and running
------------------------

spizzle is an optional dependency::

    pip install 'snowblind[spizzle]'

Run Image3 with both steps in place:

.. code-block:: bash

    strun snowblind_image3 jw..._image3_asn.json

The two steps take their parameters in the jwst pipeline's slots, e.g.
``--steps.resample.pixel_scale=0.03`` or
``--steps.outlier_detection.snr="5.0 4.0"``. Each step also runs on its
own:

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
aperture. The step computes it in a second pass over the exposures,
rebuilding each exposure's matrix, and saves it next to the mosaic, with
the file name in the mosaic's ``NFACFILE`` header card
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
* ``SparseOutlierDetectionStep`` runs on images only; spectra, coronagraphy
  and time series stay with the jwst step.

A parameter the step does not have, such as ``kernel``, is rejected rather
than ignored.

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

Speed
-----

On eight NIRCam exposures on a laptop (11 threads), the mosaic with the
variance planes takes a third of the jwst step's time with every pixel's
WCS evaluated, and less with the defaults, including the second pass for
the noise factor. Outlier detection takes about half the jwst step's time
with the defaults; the median of the drizzled exposures and the flagging
are jwst's code and are not faster.

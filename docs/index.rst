snowblind
=========

Algorithms for cleaning JWST near-infrared data, as steps that run like the
jwst pipeline's own: on their own, or as hooks inside the jwst pipelines.
The pages below describe what each step does, how it differs from the jwst
step it replaces or adds to, how to run it, and how to choose its settings.

* ``SnowblindStep`` masks the halos of cosmic-ray showers and snowballs, and
  ``JumpPlusStep`` flags the groups next to jumps and saturation that
  frame-averaged readouts leave unflagged.
* ``OpenPixelStep`` finds open and hot pixels missing from the reference
  bad-pixel mask.
* ``PersistenceFlagStep`` flags the pixels that saturated in an earlier
  exposure, for a time measured for each detector.
* ``StreamCleanStep`` removes 1/f noise from the ramp along the readout
  stream, and ``ResetRampFitStep`` fits the ramp with the reset level as an
  extra point; the ``snowblind_detector1`` pipeline runs both.
* ``SparseResampleStep`` and ``SparseOutlierDetectionStep`` drizzle with
  spizzle, which gives the exact variance of the mosaic and its noise
  factor; the ``snowblind_image3`` pipeline runs both.

.. toctree::
   :maxdepth: 2

   snowballs_and_flags
   persistence
   flicker_and_reset
   resampling

snowblind
=========

Algorithms for cleaning JWST near-infrared data, as steps that run like the
jwst pipeline's own: on their own, or as hooks inside the jwst pipelines.
The guides describe what each step does, how it differs from the jwst step
it replaces or adds to, how to run it and how to choose its settings. The
reference page of each step lists its parameters.

Ramps (Detector1)
-----------------

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - Step or pipeline
     - ``strun`` alias
     - What it does
   * - :doc:`steps/stream_clean`
     - ``stream_clean``
     - Removes 1/f noise from the ramp along the readout stream.
       Guide: :doc:`flicker_and_reset`.
   * - :doc:`steps/reset_rampfit`
     - ``reset_rampfit``
     - Fits the ramp with the reset level as an extra point.
       Guide: :doc:`flicker_and_reset`.
   * - :doc:`steps/snowblind`
     - ``snowblind``
     - Flags snowballs: the halo, the ring around the core, and the core's
       persistence in later integrations. Guide: :doc:`snowballs_and_flags`.
   * - :doc:`steps/jump_plus`
     - ``jump_plus``
     - Deprecated. Flags the groups next to jumps and saturation in
       frame-averaged readouts. Guide: :doc:`snowballs_and_flags`.
   * - :doc:`steps/snowblind_detector1`
     - ``snowblind_detector1``
     - ``calwebb_detector1`` with the stream clean, reset-anchored fit and snowball steps above in place.
       Guide: :doc:`flicker_and_reset`.

Calibrated images (Image2 and Image3)
-------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - Step or pipeline
     - ``strun`` alias
     - What it does
   * - :doc:`steps/persist`
     - ``persist``
     - Flags the pixels that saturated in an earlier exposure, for a time
       measured for each detector. Guide: :doc:`persistence`.
   * - :doc:`steps/open_pixel`
     - ``open_pixel``
     - Finds open and hot pixels missing from the reference bad-pixel mask.
       Guide: :doc:`snowballs_and_flags`.
   * - :doc:`steps/tiled_outlier`
     - ``tiled_outlier``
     - Outlier detection against a reference built from all the exposures'
       input pixels. Guide: :doc:`resampling`.
   * - :doc:`steps/sparse_outlier_detection`
     - ``sparse_outlier_detection``
     - The jwst outlier detection, drizzling with spizzle.
       Guide: :doc:`resampling`.
   * - :doc:`steps/sparse_resample`
     - ``sparse_resample``
     - The jwst resample step with spizzle: exact variances and the noise
       factor. Guide: :doc:`resampling`.
   * - :doc:`steps/snowblind_image3`
     - ``snowblind_image3``
     - ``calwebb_image3`` with the tiled outlier detection and sparse resampling above.
       Guide: :doc:`resampling`.

.. toctree::
   :caption: Guides
   :hidden:

   flicker_and_reset
   snowballs_and_flags
   persistence
   resampling

.. toctree::
   :caption: Steps
   :hidden:

   steps/stream_clean
   steps/reset_rampfit
   steps/snowblind
   steps/jump_plus
   steps/open_pixel
   steps/persist
   steps/tiled_outlier
   steps/sparse_outlier_detection
   steps/sparse_resample

.. toctree::
   :caption: Pipelines
   :hidden:

   steps/snowblind_detector1
   steps/snowblind_image3

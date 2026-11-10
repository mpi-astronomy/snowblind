snowblind_image3
================

``snowblind.Image3Pipeline``, run with ``strun snowblind_image3``. ``calwebb_image3`` with ``TiledOutlierStep`` as its outlier detection and
``SparseResampleStep`` as its resample step;
``outlier_step='sparse_outlier'`` runs ``SparseOutlierDetectionStep``
instead. A drop-in replacement.
Guide: :doc:`../resampling`.

.. step-parameters:: snowblind.Image3Pipeline

.. autoclass:: snowblind.Image3Pipeline
   :members: process

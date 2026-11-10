TiledOutlierStep
================

``snowblind.TiledOutlierStep``, run with ``strun tiled_outlier``. Outlier detection for images against one reference built from the input
pixels of all the exposures on a finer grid, processed in tiles. Finds more
cosmic rays near stars and flags fewer star cores than the jwst step.
Guide: :doc:`../resampling`.

.. step-parameters:: snowblind.TiledOutlierStep

.. autoclass:: snowblind.TiledOutlierStep
   :members: process

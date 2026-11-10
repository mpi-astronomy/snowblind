StreamCleanStep
===============

``snowblind.StreamCleanStep``, run with ``strun stream_clean``. Removes 1/f noise (striping) from the ramp along each output channel's
readout stream, with the same linear filter on every group difference.
Replaces the jwst ``clean_flicker_noise`` step.
Guide: :doc:`../flicker_and_reset`.

.. step-parameters:: snowblind.StreamCleanStep

.. autoclass:: snowblind.StreamCleanStep
   :members: process

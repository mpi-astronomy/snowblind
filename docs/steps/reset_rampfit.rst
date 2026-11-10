ResetRampFitStep
================

``snowblind.ResetRampFitStep``, run with ``strun reset_rampfit``. The reset-anchored fit: the jwst likelihood ramp fit with the reset level as
an extra point at the start of the ramp, so pixels with a single unsaturated
group still get a rate. Replaces the jwst ``ramp_fit`` step.
Guide: :doc:`../flicker_and_reset`.

.. step-parameters:: snowblind.ResetRampFitStep

.. autoclass:: snowblind.ResetRampFitStep
   :members: process

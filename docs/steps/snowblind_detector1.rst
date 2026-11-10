snowblind_detector1
===================

``snowblind.Detector1Pipeline``, run with ``strun snowblind_detector1``. ``calwebb_detector1`` with ``StreamCleanStep``, ``SnowblindStep`` and
``ResetRampFitStep`` in place, the SIRS reference-pixel correction turned on
and the pixels next to saturated ones flagged CHARGELOSS instead of
SATURATED. A drop-in replacement.
Guide: :doc:`../flicker_and_reset`.

.. step-parameters:: snowblind.Detector1Pipeline

.. autoclass:: snowblind.Detector1Pipeline
   :members: process

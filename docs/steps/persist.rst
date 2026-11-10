PersistenceFlagStep
===================

``snowblind.PersistenceFlagStep``, run with ``strun persist``. Flags the pixels that saturated in an earlier exposure of the same detector,
and the pixels next to them, for a time measured for each detector. Takes a
whole set of ``_rate`` or ``_cal`` files at once.
Guide: :doc:`../persistence`.

.. step-parameters:: snowblind.PersistenceFlagStep

.. autoclass:: snowblind.PersistenceFlagStep
   :members: process

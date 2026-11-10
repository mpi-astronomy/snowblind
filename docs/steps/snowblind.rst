SnowblindStep
=============

``snowblind.SnowblindStep``, run with ``strun snowblind``. Flags snowballs: grows each halo to follow the shape of the event, flags the
ring around the core to the end of the integration, and flags the core's
snowball persistence in the later integrations of the ramp. Replaces the
jump step's ``expand_large_events``; also runs on rate products.
Guide: :doc:`../snowballs_and_flags`.

.. step-parameters:: snowblind.SnowblindStep

.. autoclass:: snowblind.SnowblindStep
   :members: process

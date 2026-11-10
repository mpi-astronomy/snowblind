Snowball, jump and open-pixel flagging
======================================

Two steps add flags to the data quality arrays and leave the science data
alone: ``SnowblindStep`` (``snowblind``) flags snowballs, and
``OpenPixelStep`` (``open_pixel``) finds open and hot pixels that are not
in the reference bad-pixel mask. A third, ``JumpPlusStep`` (``jump_plus``),
is deprecated. Persistence between exposures has a page of its own,
:doc:`persistence`.

SnowblindStep
-------------

A snowball is a large cosmic-ray event: a core of saturated pixels inside a
halo of pixels that the jump step flags. The halo reaches beyond the flagged
pixels, and the core keeps affecting its neighbors for the rest of the
integration and into the integrations that follow. ``SnowblindStep`` does
three things with each snowball:

- it grows the halo by ``growth_factor`` times its equal-area radius, so the
  flagged area follows the shape of the event;
- it flags the ring, the core grown by ``ring_width`` pixels, in every group
  from the jump to the end of the integration;
- in a ramp with several integrations, it flags the core's snowball
  persistence in the later integrations (see `The ring and snowball
  persistence`_).

An event is a snowball when it holds a core and a disk of ``min_radius``
pixels. In a ramp the core is the pixels that saturate in the jump group or
the one after it. A rate image or rateints cube does not record when a pixel
saturated, so there the core is the pixels flagged SATURATED. Events without
a core, such as cosmic-ray tracks, are left as the jump step flagged them.
The jwst jump step's own snowball flagging (``expand_large_events``) also
requires a saturated core, and draws enclosing circles or ellipses instead
of following the event. The default ``min_radius`` of 2 is close to its
``min_jump_area`` of 15 pixels, the area of a disk of radius 2.2; the core
keeps small cosmic rays out at that size. The step works on the ramp after
the jump step (in ``groupdq``) and on rate images and rateints cubes (in
``dq``), so it can also be applied after the fact. In a ramp it writes the
rate of snowballs to EXTNCRS, where the jump step records the rate of large
events, and logs their total.

Holes in a halo are filled before it is grown. A hole of more than 200
pixels is filled only when it is a large core, made of saturated pixels and
the pixels next to them; a large hole that also holds sky, such as the area
around a saturated star in a rate image, is not.

Near the edges of the detector part of a snowball can lie on the reference
pixels (REFERENCE_PIXEL) or past the border of a subarray, where nothing is
flagged. The step completes such an event past the edge before measuring
it, so a snowball whose center is on the detector is grown by its whole
size, next to one edge or in a corner. An event centered on the outermost
light-sensitive row or column may have its core past the edge, and counts
as a snowball without one if it holds a disk of ``min_radius`` that does not
extend past the edge. An event whose center lies past the edge is measured
from the part on the detector, so it is grown less than its full size.

A snowball that strikes before group 0 is read, or during its frames, has
its core SATURATED already in group 0, so in a ramp the core never newly
saturates and no jump marks it. The step takes as such a core an area of at
least 5 pixels SATURATED in group 0, outside the bad-pixel mask, whose
saturated area has grown by no more than 10% by the last group and whose
value has fallen by more than 0.5%; a saturated star keeps collecting
light, so its area grows and its core does not empty. Ramps whose last
group is read less than 3 times later than group 0 are too short for the
test and are not searched. Such a core gets its ring from group 0 to the
end of the integration and its snowball persistence in the later
integrations. It also counts as a core in group 1, where the jump of an
event during group 0's frames shows, so a halo the jump step flags there is
grown; it is given no halo of its own.

The ring and snowball persistence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Charge leaks from the core into the pixels around it, as from any saturated
pixel (brighter-fatter at saturation). The ring is flagged in every group
from the jump to the end of the integration. When a group averages several
frames, a snowball during those frames puts part of its charge in that
group and the jump shows only in the next; the saturation step then marks
the core DO_NOT_USE in the earlier group, and the ring is flagged from that
group too.

The core and the pixels next to it also hold charge after the reset, which
they release over the next hours. In each later integration of the exposure
a group is flagged on the core while it starts less than
``core_persistence_time`` seconds after the end of the integration of the
snowball, and on the core grown by one pixel while it starts less than
``neighbor_persistence_time`` seconds after it. Pixels two pixels from the
core never persist above the threshold, so the one-pixel growth is enough.
The group times come from the ``int_times`` table of the exposure, or from
its frame counts when the table is missing. The step sees one exposure, so
nothing carries into the next one; ``PersistenceFlagStep`` does that.

Both times default to those measured for the detector and packaged with
snowblind (``snowblind/data/snowball_persistence_<detector>.asdf``), for
the ten NIRCam detectors, NIRISS and the two NIRSpec detectors. A detector
without a file gets the longest packaged times (about 21000 s for the core
and 3800 s for the pixels next to it) and a warning. Each time is the 99th
percentile over the detector's snowballs of the time at which a pixel falls
to 0.005 DN/s, the threshold the ``PersistenceFlagStep`` times also use:
from about 1400 s (NRCA4) to 21000 s (NRCB4) for the core and from about
550 s to 3800 s for the pixels next to it.

The times were measured in dark exposures taken back to back, where
nothing but cosmic rays saturates. Each integration is run through
Detector1 to the jump step, the cores are found with the step's own rule,
and each group difference, with the 1/f noise removed, is averaged per
pixel in blocks of about 100 s. A pixel's excess is its rate in a block
less its median rate in the integrations before the snowball and those
more than 100 ks after it. The mean excess over cores falls as a power law
in the time since the end of the snowball's integration; its exponent is
fitted to that mean, and the amplitude to each core, which gives the
core's decay time. Where the mean over cores crosses the threshold later
than its power law, the percentile is scaled up by the ratio, and where
darks of the detector taken later show more excess, both times are
lengthened as if every core were brighter by that ratio. Control pixels
far from every core, treated the same way, show how far noise alone
pushes the percentile: 0.7 ks to 3.2 ks for the core and 0.4 ks to 2.3 ks
for its neighbors, well below the packaged times. Each file holds the
fits, the mean excess per time bin, every core's decay time and the
exposures used. The scripts ``measure_snowball_persistence.py``,
``fit_snowball_persistence.py`` and ``package_snowball_persistence.py``
under ``scripts/`` make the files for a new detector or new darks.

.. figure:: figures/snowball_persistence.png
   :width: 100%

   The excess rate of snowball cores in dark exposures against the time
   since the end of the snowball's integration, one panel per detector:
   the mean over cores of the saturated pixels (blue), of the pixels one
   pixel from them (orange) and two pixels from them (green), with the
   power law fitted to the saturated pixels, the 5 mDN/s threshold
   (dashed), and the times the step uses (dotted), the 99th percentile of
   the per-core decay times. Error bars are the standard error over cores;
   bins at or below zero are not drawn.

``ring_width`` and the one-pixel growth are counted from the SATURATED
flags. With the jwst saturation step's default, ``n_pix_grow_sat=1``,
SATURATED already includes a one-pixel ring around the pixels that
reached the saturation threshold, so the ring reaches ``ring_width`` + 1
pixels from them and the core's time covers its neighbors too.
``snowblind_detector1`` instead runs the saturation step with
``n_pix_grow_sat=0``, so SATURATED marks only the pixels that reached the
threshold, and flags their neighbors CHARGELOSS and DO_NOT_USE
(``snowblind.saturation.flag_saturation_neighbors`` as a post-hook of the
saturation step), which keeps them out of the fits and the jump search as
before. CHARGELOSS is the flag the jwst pipeline uses for charge that
spreads from a very bright pixel into its neighbors (its
``charge_migration`` step). The reset-anchored fit keeps it in the rate and
rateints products, so ``PersistenceFlagStep`` flags these pixels in later
exposures, as it does the saturated ones. The jwst ramp fit
(``--reset_fit=False``) drops it from the rate products and, as for charge
migration, gives these pixels the read-noise variance of a whole ramp, so
that resampling does not weight them down. The pipeline runs
``SnowblindStep`` with ``ring_width=3``, which reaches as far from the
saturated pixels as the default does with the jwst saturation default.
Given another ``n_pix_grow_sat``, the pipeline narrows ``ring_width`` by as
much, so the ring always reaches 3 pixels from the pixels that saturated.

``after_jumps``, which flagged the ring in that many groups after the jump
only, is deprecated; set, it still does.

Running it
~~~~~~~~~~

It runs like any pipeline step. On the result of the jump step::

    strun snowblind jw001234_010203_00001_nrcalong_jump.fits --suffix=snowblind

Or as a post-hook of the jump step in a full Detector1 run, with the jump
step's own snowball flagging turned off::

    strun calwebb_detector1 jw001234_010203_00001_nrcalong_uncal.fits --steps.jump.post_hooks="snowblind.SnowblindStep", --steps.jump.expand_large_events=False

The same in Python:

.. code-block:: python

    from jwst.pipeline import Detector1Pipeline
    from snowblind import SnowblindStep

    steps = {
        "jump": {
            "save_results": True,
            "expand_large_events": False,
            "post_hooks": [SnowblindStep],
        },
    }
    Detector1Pipeline.call("jw001234_010203_00001_nrcalong_uncal.fits", steps=steps, save_results=True)

A hook given as a class runs with the step's default parameters, which suit
the jwst saturation step's default (``n_pix_grow_sat=1``). If the
saturation step ran with ``n_pix_grow_sat=0``, give a step instance with a
ring one pixel wider, ``SnowblindStep(ring_width=3)``, so that it reaches
as far from the pixels that saturated. The ``snowblind_detector1`` pipeline
(see :doc:`flicker_and_reset`) runs it as a post-hook of its jump step and
sets ``ring_width`` to match its saturation step, as described above.

JumpPlusStep (deprecated)
-------------------------

In readouts that average several frames into each group (NFRAMES > 1),
``JumpPlusStep`` flags the group after each detected jump, since a cosmic
ray that arrives partway through a group also changes the next group
difference, and flags the group before a saturated group as saturated,
since the saturation may have happened in a frame of that group. With
NFRAMES = 1 the step is skipped.

It is deprecated and will be removed, because the pipeline now takes care
of both cases. Since jwst 1.16 the saturation step flags the group before
saturation when its last frames may have saturated, and the jump step flags
a group that a cosmic ray struck partway through. The likelihood ramp fits
(``ResetRampFitStep``, the jwst ``LIKELY`` algorithm) leave out both
differences next to a flagged group, and ``OLS_C`` leaves out the one-group
segment, so what ``JumpPlusStep`` flags on top is mostly good data. Run
after ``SnowblindStep``, it also copies the grown halos into the next group,
and every halo pixel loses one more difference of its ramp, which made the
halos noisier in the rate. ``snowblind_detector1`` no longer runs it.

OpenPixelStep
-------------

The counts of an open pixel move to its neighbors, so it shows as a faint
pixel inside a cross of bright ones, and new ones appear over time.
``OpenPixelStep`` finds those that have appeared since the reference
bad-pixel mask was made by self-calibration: it takes all the images of a
visit or program, groups them by detector, and flags in every image of a
detector the pixels that stand out above the local background in the median
of that detector's images (with ``flag_low_signal_pix``, also those below
it), setting ``DO_NOT_USE`` and ``ADJ_OPEN``. Run it on flat-fielded images,
after ``calwebb_image2``, or as a pre-hook of ``calwebb_image3`` before
resampling. It is experimental: use it with caution, and expect to tune
``threshold``.

Snowball, jump and open-pixel flagging
======================================

Two steps add flags to the data quality arrays and leave the science data
alone: ``SnowblindStep`` (``snowblind``) masks the halos of snowballs, and
``OpenPixelStep`` (``open_pixel``) finds open and
hot pixels that are not in the reference bad-pixel mask. A third,
``JumpPlusStep`` (``jump_plus``), is deprecated. Persistence has a page of
its own, :doc:`persistence`.

SnowblindStep
-------------

A snowball is a saturated core inside a halo of pixels that the jump step
flags. The halo reaches beyond the flagged pixels, and the saturated core
keeps affecting the following groups. ``SnowblindStep`` takes each
contiguous region of jump-flagged pixels, with its holes filled, that holds
a core and contains a disk of ``min_radius`` pixels, and grows it by
``growth_factor`` times its equal-area radius, so the mask follows the
actual shape of the event; the core is grown by ``ring_width`` pixels and
flagged in the ``after_jumps`` following groups too. In a ramp the core is
the pixels that saturate in the jump group or the next; a rate image or
rateints cube keeps no saturation per group, so there the holes in the
region take its place. The core persists after the reset: in a ramp it is
also flagged, not grown, in every group of the next integration (group 0
included, so the reset-anchored fit leaves out its reset difference too);
its neighbors show no persistence, and later integrations none either.
Large jump regions without a core, such as
cosmic-ray tracks, are left as the jump step flagged them. The jwst jump
step's own snowball flagging (``expand_large_events``) also looks for a
saturated core, and draws enclosing circles or ellipses. The default
``min_radius`` of 2 is close to its ``min_jump_area`` of 15 pixels, the
area of a disk of radius 2.2; the core keeps small cosmic-ray hits out at
that size. The step works on
the ramp after the jump step (in ``groupdq``), and on rate images and
rateints cubes (in ``dq``), so it can also be applied after the fact.

Near the edges of the detector part of an event can lie on the reference
pixels (REFERENCE_PIXEL in the data quality arrays) or past the border of a
subarray, where nothing is flagged. There the disk of ``min_radius`` may
extend past the edge as long as its center is on a light-sensitive pixel,
and before its radius is measured the event is completed past the edge
with the largest disk that fits in it around one of its pixels and comes
closest to enclosing it. A round event whose center is on the detector,
next to one edge or in a corner, is then grown by its whole size. An event
whose center is on the outermost light-sensitive row or column may have its
core past the edge, and is kept without one if it holds a disk of
``min_radius`` that does not extend past the edge. An event whose center
lies past the edge is measured from the part on the detector, so it is
grown less than its full size.

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

A hook given as a class runs with the step's default parameters; give a step
instance, ``SnowblindStep(after_jumps=5)``, for others. The
``snowblind_detector1`` pipeline (see :doc:`flicker_and_reset`) runs it as a
post-hook of its jump step, with those settings.

JumpPlusStep (deprecated)
-------------------------

In readouts that average several frames into each group (NFRAMES > 1),
``JumpPlusStep`` flags the group after each detected jump, since a cosmic
ray that arrives partway through a group also changes the next group
difference, and flags the group before a saturated group as saturated,
since the saturation may have happened in a frame of that group. With
NFRAMES = 1 the step is skipped. It is deprecated and will be removed:

- Since jwst 1.16 the saturation step flags the group before saturation
  (DO_NOT_USE) when its average shows that its last frames may have
  saturated, and the jump step flags a group that a cosmic ray hit partway
  through. The likelihood ramp fits leave out both differences next to a
  flagged group, and ``OLS_C`` leaves out the one-group segment it makes, so
  the groups ``JumpPlusStep`` adds before saturation are good data.
- The likelihood ramp fits (``ResetRampFitStep``, the jwst ``LIKELY``
  algorithm) also leave out the difference after a flagged group, so the
  copy of each jump into the next group takes out a difference with no
  jump in it. With ``OLS_C`` the jump left in that difference is below the
  jump threshold, and small on average.

Run after ``SnowblindStep``, it also copies the grown halos into the next
group, and the halo pixels lose one more difference of their ramp.

OpenPixelStep
-------------

The counts of an open pixel move to its neighbors, so it shows as a faint
pixel inside a cross of bright ones, and new ones appear over time.
``OpenPixelStep`` finds those that have appeared since the reference
bad-pixel mask was made by self-calibration: it takes all the images of a
visit or program, groups them by detector, and flags in every image of a
detector the pixels that stand out above the local background in the median
of that detector's images (with ``flag_low_signal_pix``, also those below
it), setting ``DO_NOT_USE`` and ``ADJ_OPEN``. Run it on flat-fielded images, after ``calwebb_image2``, or as
a pre-hook of ``calwebb_image3`` before resampling. It is experimental: use
it with caution, and expect to tune ``threshold``.

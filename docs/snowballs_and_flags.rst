Snowball, jump and open-pixel flagging
======================================

Two steps add flags to the data quality arrays and leave the science data
alone: ``SnowblindStep`` (``snowblind``) masks the halos of cosmic-ray
showers and snowballs, and ``OpenPixelStep`` (``open_pixel``) finds open and
hot pixels that are not in the reference bad-pixel mask. A third,
``JumpPlusStep`` (``jump_plus``), is deprecated. Persistence has a page of
its own, :doc:`persistence`.

SnowblindStep
-------------

A large cosmic-ray shower or snowball leaves a halo around the pixels that
the jump step flags, and its saturated core keeps affecting the following
groups. ``SnowblindStep`` takes each contiguous region of jump-flagged
pixels larger than a disk of ``min_radius`` pixels and grows it by
``growth_factor`` times its equal-area radius, so the mask follows the
actual shape of the event; the saturated core inside such a region is grown
by ``ring_width`` pixels and flagged in the ``after_jumps`` following groups
too. The jwst jump step's own snowball flagging (``expand_large_events``)
draws enclosing circles or ellipses instead. The step works on the ramp after
the jump step (in ``groupdq``), and on rate images and rateints cubes (in
``dq``), so it can also be applied after the fact.

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
instance, ``SnowblindStep(min_radius=3, after_jumps=5)``, for others. The
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

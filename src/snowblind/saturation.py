"""Flag the pixels next to saturated pixels for the charge that leaks into them, apart from the saturated ones."""
import logging

import numpy as np
from jwst import datamodels
from jwst.lib import pipe_utils
from jwst.saturation import x_irs2
from scipy import ndimage

__all__ = ["flag_saturation_neighbors"]

log = logging.getLogger(__name__)

SATURATED = datamodels.dqflags.group["SATURATED"]
CHARGELOSS = datamodels.dqflags.group["CHARGELOSS"]
DO_NOT_USE = datamodels.dqflags.group["DO_NOT_USE"]
SQUARE = np.ones((3, 3), dtype=bool)


def flag_saturation_neighbors(input_model):
    """Flag the 8 pixels around each saturated pixel with CHARGELOSS and DO_NOT_USE, group by group.

    A post-hook of the saturation step run with ``n_pix_grow_sat=0``, as
    ``snowblind_detector1`` runs it.  The jwst step's default,
    ``n_pix_grow_sat=1``, sets SATURATED on these pixels instead, so that a
    pixel that saturated (and persists after the reset) cannot be told from a
    neighbor flagged for the charge spilling out of it.  Here the neighbors
    leave the fits and the jump search as they do with the jwst default, while
    SATURATED keeps marking only the pixels that reached the saturation
    threshold.  NIRSpec IRS2 groups are grown in their normal layout, without the
    interleaved reference pixels, as the saturation step grows them.

    Run as a hook, stpipe wraps it in a step when given its import path,
    ``"snowblind.saturation.flag_saturation_neighbors"``.

    Parameters
    ----------
    input_model : RampModel
        Output of the saturation step; its ``groupdq`` is updated in place.

    Returns
    -------
    RampModel
        The input, with CHARGELOSS and DO_NOT_USE set in each group on the pixels
        next to that group's SATURATED pixels that are not SATURATED themselves.
    """
    irs2_mask, detector = None, None
    if pipe_utils.is_irs2(input_model):
        irs2_mask = x_irs2.make_mask(input_model)
        detector = input_model.meta.instrument.detector
    n_flagged = 0
    for integration in input_model.groupdq:
        saturated, neighbors = None, None
        for plane in integration:
            now = (plane & SATURATED) != 0
            if not now.any():
                continue
            # SATURATED carries to the later groups, so most groups repeat the one before
            if saturated is None or not np.array_equal(now, saturated):
                saturated = now
                if irs2_mask is None:
                    neighbors = _pixels_next_to_saturated(now)
                else:
                    neighbors = _pixels_next_to_saturated_irs2(now, irs2_mask, detector)
            plane[neighbors] |= CHARGELOSS | DO_NOT_USE
        if neighbors is not None:
            n_flagged += int(neighbors.sum())
    log.info(f"Flagged {n_flagged} pixels next to saturated pixels with CHARGELOSS and DO_NOT_USE, counted in the "
             "last group of each integration")
    return input_model


def _pixels_next_to_saturated(saturated):
    """The pixels within one pixel of a saturated pixel that are not saturated themselves."""
    return ndimage.binary_dilation(saturated, structure=SQUARE) & ~saturated


def _pixels_next_to_saturated_irs2(saturated, irs2_mask, detector):
    """`_pixels_next_to_saturated` for an IRS2 group: found in its normal layout, then placed back in IRS2 layout."""
    neighbors = np.zeros(saturated.shape, dtype=bool)
    normal = x_irs2.from_irs2(saturated, irs2_mask, detector)
    x_irs2.to_irs2(neighbors, _pixels_next_to_saturated(normal), irs2_mask, detector)
    return neighbors

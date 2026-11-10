"""Flag the pixels next to saturated pixels for the charge that leaks into them, apart from the saturated ones."""
import functools
import logging

import numpy as np
from jwst import datamodels
from jwst.lib import pipe_utils
from jwst.saturation import x_irs2
from scipy import ndimage

__all__ = ["flag_saturation_neighbors", "saturated_in_group"]

log = logging.getLogger(__name__)

SATURATED = datamodels.dqflags.group["SATURATED"]
CHARGELOSS = datamodels.dqflags.group["CHARGELOSS"]
DO_NOT_USE = datamodels.dqflags.group["DO_NOT_USE"]
# a pixel and its eight neighbors: the pixels this hook flags, and those SnowblindStep flags next to a core
SQUARE = np.ones((3, 3), dtype=bool)


def saturated_in_group(groupdq, group):
    """The pixels saturated by the end of a group of an integration.

    These are the pixels flagged SATURATED in the group, and those the saturation step
    flagged DO_NOT_USE in it with SATURATED in the next group: when a group averages
    several frames, the step marks so the group during whose frames a pixel saturated.

    Parameters
    ----------
    groupdq : ndarray, shape (ngroups, nrows, ncols)
        Group DQ of one integration.
    group : int
        Index of the group.

    Returns
    -------
    ndarray of bool, shape (nrows, ncols)
        True on the saturated pixels.
    """
    saturated = (groupdq[group] & SATURATED) != 0
    if group + 1 < len(groupdq):
        saturated |= ((groupdq[group] & DO_NOT_USE) != 0) & ((groupdq[group + 1] & SATURATED) != 0)
    return saturated


def flag_saturation_neighbors(input_model):
    """Flag the 8 pixels around each saturated pixel with CHARGELOSS and DO_NOT_USE, group by group.

    A post-hook of the saturation step run with ``n_pix_grow_sat=0``, as
    ``snowblind_detector1`` runs it.  The jwst step's default,
    ``n_pix_grow_sat=1``, sets SATURATED on these pixels instead, so that a
    pixel that saturated (and persists after the reset) cannot be told from a
    neighbor flagged for the charge spilling out of it.  Here the neighbors
    leave the fits and the jump search as they do with the jwst default, while
    SATURATED keeps marking only the pixels that reached the saturation
    threshold.  As with the jwst default, the neighbors are flagged from the group
    during whose frames a pixel saturated (`saturated_in_group`), and the zero frame
    of the pixels next to a pixel whose zero frame saturated is set to 0, which marks
    it unusable, as the step does with the saturated one: it holds the charge spilling
    out.  NIRSpec IRS2 groups are grown in their normal layout, without the
    interleaved reference pixels, as the saturation step grows them.

    Run as a hook, stpipe wraps it in a step when given its import path,
    ``"snowblind.saturation.flag_saturation_neighbors"``.

    Parameters
    ----------
    input_model : RampModel
        Output of the saturation step; its ``groupdq``, and its ``zeroframe`` when the
        exposure has one, are updated in place.

    Returns
    -------
    RampModel
        The input, with CHARGELOSS and DO_NOT_USE set in each group on the pixels
        next to that group's saturated pixels that are not saturated themselves, and
        the zero frame set to 0 next to the pixels whose zero frame saturated.
    """
    pixels_next_to = _neighbor_search(input_model)
    has_zero_frame = bool(input_model.meta.exposure.zero_frame)
    n_flagged, n_zeroed = 0, 0
    for i, integration in enumerate(input_model.groupdq):
        saturated, neighbors = None, None
        for group, plane in enumerate(integration):
            now = saturated_in_group(integration, group)
            if not now.any():
                continue
            # SATURATED carries to the later groups, so most groups repeat the one before
            if saturated is None or not np.array_equal(now, saturated):
                saturated = now
                neighbors = pixels_next_to(now)
            plane[neighbors] |= CHARGELOSS | DO_NOT_USE
        if neighbors is not None:
            n_flagged += int(neighbors.sum())
        if has_zero_frame:
            n_zeroed += _zero_the_zero_frame_next_to_saturated(input_model.zeroframe[i], integration[0], pixels_next_to)
    log.info(f"Flagged {n_flagged} pixels next to saturated pixels with CHARGELOSS and DO_NOT_USE, counted in the "
             f"last group of each integration, and set the zero frame to 0 on {n_zeroed}")
    return input_model


def _neighbor_search(model):
    """The function that finds the pixels next to the saturated ones in a group of ``model``: in the normal layout of a
    NIRSpec IRS2 group, as the saturation step grows them."""
    if not pipe_utils.is_irs2(model):
        return _pixels_next_to_saturated
    return functools.partial(
        _pixels_next_to_saturated_irs2, irs2_mask=x_irs2.make_mask(model), detector=model.meta.instrument.detector
    )


def _zero_the_zero_frame_next_to_saturated(zeroframe, first_group, pixels_next_to):
    """Set to 0 the zero frame of one integration on the pixels next to a pixel whose zero frame saturated.

    The saturation step sets a saturated zero frame to 0, and SATURATED in group 0; returns the pixels set.
    """
    saturated = (zeroframe == 0) & ((first_group & SATURATED) != 0)
    if not saturated.any():
        return 0
    neighbors = pixels_next_to(saturated)
    zeroframe[neighbors] = 0.0
    return int(neighbors.sum())


def _pixels_next_to_saturated(saturated):
    """The pixels within one pixel of a saturated pixel that are not saturated themselves."""
    return ndimage.binary_dilation(saturated, structure=SQUARE) & ~saturated


def _pixels_next_to_saturated_irs2(saturated, irs2_mask, detector):
    """`_pixels_next_to_saturated` for an IRS2 group: found in its normal layout, then placed back in IRS2 layout."""
    neighbors = np.zeros(saturated.shape, dtype=bool)
    normal = x_irs2.from_irs2(saturated, irs2_mask, detector)
    x_irs2.to_irs2(neighbors, _pixels_next_to_saturated(normal), irs2_mask, detector)
    return neighbors

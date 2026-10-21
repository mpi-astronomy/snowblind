"""Readout geometry of the JWST near-infrared H2RG detectors.

Everything here works in the detector frame returned by
``jwst.lib.reffile_utils.science_detector_frame_transform``: rows are the
slow axis, columns are the fast axis, and the four output channels are
equal-width column blocks.  Within a channel the pixels of a row are read
one per pixel time (10 microseconds, called a clock below), adjacent
channels read in opposite directions, and there is a fixed overhead of
``LINE_OVERHEAD`` pixel times between the end of one row and the start of
the next, so a full-frame row takes 524 pixel times or 5.24 ms.  The clock
index of a pixel is::

    t = row * (width + LINE_OVERHEAD) + k

where ``k`` is the column index counted in the direction the channel is
read.  A "stream" is the channel image laid out on that clock grid, with
the overhead clocks filled with zeros.

The line overhead and the channel layout follow NSClean
(``jwst/clean_flicker_noise/nsclean.py``, ``NSCleanSubarray.nloh``) and the
refpix step (``jwst/refpix/reference_pixels.py``).  The direction in which
the first channel is read is a convention that must be checked on real data
(see ``scripts/check_readout_order.py``).
"""
import numpy as np
from jwst.lib.reffile_utils import (
    detector_science_frame_transform,
    science_detector_frame_transform,
)

__all__ = [
    "NOUTPUTS",
    "LINE_OVERHEAD",
    "NREFPIX",
    "to_detector_frame",
    "to_science_frame",
    "channel_width",
    "channel_slices",
    "channel_direction",
    "channel_blocks",
    "clock_index",
    "stream_length",
    "to_stream",
    "from_stream",
    "stream_valid",
]

NOUTPUTS = 4
LINE_OVERHEAD = 12
NREFPIX = 4


def to_detector_frame(array, fastaxis, slowaxis):
    """Transform an array from the science (DMS) frame to the detector frame.

    Parameters
    ----------
    array : ndarray
        Array with at least two trailing image dimensions.
    fastaxis, slowaxis : int
        Values of ``meta.subarray.fastaxis`` and ``meta.subarray.slowaxis``.

    Returns
    -------
    ndarray
        A view of ``array`` with rows along the slow axis and columns along
        the fast axis.
    """
    return science_detector_frame_transform(array, fastaxis, slowaxis)


def to_science_frame(array, fastaxis, slowaxis):
    """Inverse of `to_detector_frame`."""
    return detector_science_frame_transform(array, fastaxis, slowaxis)


def channel_width(ncols):
    """Return the number of columns per output channel."""
    if ncols % NOUTPUTS != 0:
        raise ValueError(f"{ncols} columns cannot be split into {NOUTPUTS} output channels")
    return ncols // NOUTPUTS


def channel_slices(ncols):
    """Return the column slice of each output channel in the detector frame."""
    width = channel_width(ncols)
    return [slice(k * width, (k + 1) * width) for k in range(NOUTPUTS)]


def channel_direction(channel, first_direction=1):
    """Return +1 or -1 for the column direction in which a channel is read.

    Adjacent outputs are read in opposite directions. ``first_direction``
    is the direction of channel 0; +1 means increasing detector column.
    """
    return first_direction if channel % 2 == 0 else -first_direction


def clock_index(nrows, width, direction):
    """Return the clock index ``t`` of every pixel of a channel image.

    Parameters
    ----------
    nrows, width : int
        Shape of the channel image.
    direction : int
        +1 if the channel is read in increasing column order, -1 otherwise.

    Returns
    -------
    ndarray of int64, shape (nrows, width)
    """
    k = np.arange(width, dtype=np.int64)
    if direction < 0:
        k = k[::-1]
    rows = np.arange(nrows, dtype=np.int64)[:, np.newaxis]
    return rows * (width + LINE_OVERHEAD) + k[np.newaxis, :]


def stream_length(nrows, width):
    """Return the number of clocks spanned by a channel image."""
    return nrows * (width + LINE_OVERHEAD)


def to_stream(channel_image, direction, fill=0.0):
    """Lay a channel image out on its clock grid.

    Parameters
    ----------
    channel_image : ndarray, shape (nrows, width)
        Channel image in the detector frame.
    direction : int
        Readout direction of the channel, +1 or -1.
    fill : scalar
        Value placed in the line-overhead clocks.

    Returns
    -------
    ndarray, shape (nrows * (width + LINE_OVERHEAD),)
    """
    nrows, width = channel_image.shape
    padded = np.full((nrows, width + LINE_OVERHEAD), fill, dtype=channel_image.dtype)
    if direction < 0:
        padded[:, :width] = channel_image[:, ::-1]
    else:
        padded[:, :width] = channel_image
    return padded.ravel()


def from_stream(stream, nrows, width, direction):
    """Inverse of `to_stream`: drop the overhead clocks and restore column order."""
    image = stream.reshape(nrows, width + LINE_OVERHEAD)[:, :width]
    if direction < 0:
        image = image[:, ::-1]
    return np.ascontiguousarray(image)


def stream_valid(nrows, width):
    """Return a boolean stream that is True at pixel clocks and False at overhead clocks."""
    return to_stream(np.ones((nrows, width), dtype=bool), 1, fill=False)


def channel_blocks(shape, fastaxis, slowaxis):
    """Science-frame slices ``(rows, cols)`` covered by each output channel.

    Channels are column blocks in the detector frame; in the science frame
    they are column blocks for the NIRCam geometries and row blocks for
    NIRISS.  The blocks are returned in channel order.
    """
    det_shape = to_detector_frame(np.zeros(shape, dtype=np.uint8), fastaxis, slowaxis).shape
    width = channel_width(det_shape[-1])
    ids_det = np.repeat(np.arange(NOUTPUTS, dtype=np.int16), width)[np.newaxis, :] * np.ones((det_shape[0], 1), np.int16)
    ids_sci = to_science_frame(ids_det, fastaxis, slowaxis)
    blocks = []
    for k in range(NOUTPUTS):
        ys, xs = np.nonzero(ids_sci == k)
        blocks.append((slice(int(ys.min()), int(ys.max()) + 1), slice(int(xs.min()), int(xs.max()) + 1)))
    return blocks

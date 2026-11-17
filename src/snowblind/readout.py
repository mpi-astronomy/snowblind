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

NIRSpec IRS2 (``NRSIRS2`` and ``NRSIRS2RAPID``) interleaves reference reads
with the pixels of a row, so the clock grid of a row is longer (712 clocks
for a full-frame channel) and has gaps; `RowLayout` describes either grid,
and the module functions take the traditional one.
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
    "IRS2_NORMAL",
    "IRS2_REFERENCE",
    "IRS2_ROW_PAD",
    "RowLayout",
    "row_layout",
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
# IRS2 defaults, the values of the NRS_NORM and NRS_REF keywords and of ``pad`` in
# ``jwst.refpix.irs2_subtract_reference``: pixels read between two groups of reference
# reads, reference reads in a group, and the idle clocks that end a row
IRS2_NORMAL = 16
IRS2_REFERENCE = 4
IRS2_ROW_PAD = 8


class RowLayout:
    """Clock layout of one row of a channel: the clock of each pixel and the length of the row.

    ``offsets[k]`` is the clock, counted from the start of the row, at which
    the k-th pixel of the row is read (k counted in the read direction), and
    ``row_clocks`` the clocks from the start of one row to the start of the
    next.  The traditional readout reads the pixels back to back and idles
    for `LINE_OVERHEAD` clocks.  NIRSpec IRS2 reads ``reference`` reference
    pixels, with two clocks of settling, after every ``normal`` pixels,
    starting half a block in, and ends the row with ``pad`` idle clocks; the
    offsets and row length are those of the time-ordered row that the jwst
    refpix step builds (``hnorm1`` and ``row`` in
    ``jwst.refpix.irs2_subtract_reference``).  That step also strips the
    interleaved reads from the data, so a channel image is ``width`` pixels
    wide under either layout and only the clock grid differs.

    Parameters
    ----------
    offsets : array_like of int, shape (width,)
        Clock of each pixel within the row, in read order.
    row_clocks : int
        Clocks per row.
    """

    def __init__(self, offsets, row_clocks):
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.row_clocks = int(row_clocks)
        if self.offsets.size and (self.offsets.min() < 0 or self.offsets.max() >= self.row_clocks):
            raise ValueError("every pixel clock must lie within the row")

    @property
    def width(self):
        """Number of pixels in the row."""
        return self.offsets.size

    @classmethod
    def traditional(cls, width):
        """The layout of a traditional readout: ``width`` pixels back to back, then `LINE_OVERHEAD` idle clocks."""
        return cls(np.arange(width, dtype=np.int64), width + LINE_OVERHEAD)

    @classmethod
    def irs2(cls, width, normal=IRS2_NORMAL, reference=IRS2_REFERENCE, pad=IRS2_ROW_PAD):
        """The layout of a NIRSpec IRS2 readout (712 clocks for a 512-pixel row with the defaults)."""
        k = np.arange(width, dtype=np.int64)
        offsets = k + (reference + 2) * ((k + normal // 2) // normal)
        row_clocks = (normal + reference + 2) * (width // normal) + pad
        return cls(offsets, row_clocks)

    def __eq__(self, other):
        return (
            isinstance(other, RowLayout)
            and self.row_clocks == other.row_clocks
            and np.array_equal(self.offsets, other.offsets)
        )

    def __repr__(self):
        return f"RowLayout(width={self.width}, row_clocks={self.row_clocks})"

    def clock_index(self, nrows, direction):
        """Return the clock index of every pixel of a channel image of ``nrows`` rows (see `clock_index`)."""
        k = self.offsets if direction > 0 else self.offsets[::-1]
        rows = np.arange(nrows, dtype=np.int64)[:, np.newaxis]
        return rows * self.row_clocks + k[np.newaxis, :]

    def stream_length(self, nrows):
        """Return the number of clocks spanned by ``nrows`` rows."""
        return nrows * self.row_clocks

    def to_stream(self, channel_image, direction, fill=0.0):
        """Lay a channel image out on the clock grid (see `to_stream`)."""
        nrows, width = channel_image.shape
        if width != self.width:
            raise ValueError(f"channel image has {width} columns, the row layout {self.width}")
        padded = np.full((nrows, self.row_clocks), fill, dtype=channel_image.dtype)
        padded[:, self.offsets] = channel_image[:, ::-1] if direction < 0 else channel_image
        return padded.ravel()

    def from_stream(self, stream, nrows, direction):
        """Inverse of `RowLayout.to_stream` (see `from_stream`)."""
        image = stream.reshape(nrows, self.row_clocks)[:, self.offsets]
        if direction < 0:
            image = image[:, ::-1]
        return np.ascontiguousarray(image)

    def stream_valid(self, nrows):
        """Return a boolean stream that is True at pixel clocks and False elsewhere."""
        return self.to_stream(np.ones((nrows, self.width), dtype=bool), 1, fill=False)


def row_layout(model, width):
    """The `RowLayout` of a channel of ``width`` pixels for an exposure.

    Parameters
    ----------
    model : `~stdatamodels.jwst.datamodels.JwstDataModel`
        The exposure; ``meta.exposure.readpatt`` picks the layout and, for
        IRS2, ``meta.exposure.nrs_normal`` and ``nrs_reference`` (NRS_NORM
        and NRS_REF) give the interleave, with the defaults when missing.
    width : int
        Pixels per channel row.

    Returns
    -------
    `RowLayout`
        The IRS2 layout for the ``NRSIRS2`` read patterns, else the traditional one.
    """
    readpatt = str(model.meta.exposure.readpatt or "").upper()
    if readpatt.startswith("NRSIRS2"):
        normal = model.meta.exposure.nrs_normal or IRS2_NORMAL
        reference = model.meta.exposure.nrs_reference or IRS2_REFERENCE
        return RowLayout.irs2(width, int(normal), int(reference))
    return RowLayout.traditional(width)


def to_detector_frame(array, fastaxis, slowaxis):
    """Transform an array from the science (DMS) frame to the detector frame.

    Parameters
    ----------
    array : ndarray
        Array with at least two trailing image dimensions.
    fastaxis : int
        Value of ``meta.subarray.fastaxis``.
    slowaxis : int
        Value of ``meta.subarray.slowaxis``.

    Returns
    -------
    ndarray
        A view of ``array`` with rows along the slow axis and columns along
        the fast axis.
    """
    return science_detector_frame_transform(array, fastaxis, slowaxis)


def to_science_frame(array, fastaxis, slowaxis):
    """Inverse of `to_detector_frame`.

    Parameters
    ----------
    array : ndarray
        Array in the detector frame, with at least two trailing image
        dimensions.
    fastaxis : int
        Value of ``meta.subarray.fastaxis``.
    slowaxis : int
        Value of ``meta.subarray.slowaxis``.

    Returns
    -------
    ndarray
        A view of ``array`` in the science (DMS) frame.
    """
    return detector_science_frame_transform(array, fastaxis, slowaxis)


def channel_width(ncols):
    """Return the number of columns per output channel.

    Parameters
    ----------
    ncols : int
        Number of detector-frame columns; must be a multiple of `NOUTPUTS`.

    Returns
    -------
    int
        ``ncols // NOUTPUTS``.

    Raises
    ------
    ValueError
        If ``ncols`` is not a multiple of `NOUTPUTS`.
    """
    if ncols % NOUTPUTS != 0:
        raise ValueError(f"{ncols} columns cannot be split into {NOUTPUTS} output channels")
    return ncols // NOUTPUTS


def channel_slices(ncols):
    """Return the column slice of each output channel in the detector frame.

    Parameters
    ----------
    ncols : int
        Number of detector-frame columns.

    Returns
    -------
    list of slice
        One column slice per channel, in channel order.
    """
    width = channel_width(ncols)
    return [slice(k * width, (k + 1) * width) for k in range(NOUTPUTS)]


def channel_direction(channel, first_direction=1):
    """Return +1 or -1 for the column direction in which a channel is read.

    Adjacent outputs are read in opposite directions. ``first_direction``
    is the direction of channel 0; +1 means increasing detector column.

    Parameters
    ----------
    channel : int
        Channel index, 0 to ``NOUTPUTS - 1``.
    first_direction : int
        Readout direction of channel 0, +1 or -1.

    Returns
    -------
    int
        ``first_direction`` for even channels, its negative for odd ones.
    """
    return first_direction if channel % 2 == 0 else -first_direction


def clock_index(nrows, width, direction, layout=None):
    """Return the clock index ``t`` of every pixel of a channel image.

    Parameters
    ----------
    nrows : int
        Number of rows of the channel image.
    width : int
        Number of columns of the channel image.
    direction : int
        +1 if the channel is read in increasing column order, -1 otherwise.
    layout : `RowLayout` or None
        Clock layout of a row; the traditional one of ``width`` pixels when None.

    Returns
    -------
    ndarray of int64, shape (nrows, width)
    """
    return (layout or RowLayout.traditional(width)).clock_index(nrows, direction)


def stream_length(nrows, width, layout=None):
    """Return the number of clocks spanned by a channel image.

    Parameters
    ----------
    nrows : int
        Number of rows of the channel image.
    width : int
        Number of columns of the channel image.
    layout : `RowLayout` or None
        Clock layout of a row; the traditional one of ``width`` pixels when None.

    Returns
    -------
    int
        ``nrows`` times the clocks per row (``width + LINE_OVERHEAD`` for the traditional layout).
    """
    return (layout or RowLayout.traditional(width)).stream_length(nrows)


def to_stream(channel_image, direction, fill=0.0, layout=None):
    """Lay a channel image out on its clock grid.

    Parameters
    ----------
    channel_image : ndarray, shape (nrows, width)
        Channel image in the detector frame.
    direction : int
        Readout direction of the channel, +1 or -1.
    fill : scalar
        Value placed in the clocks without a pixel (line overhead, reference reads).
    layout : `RowLayout` or None
        Clock layout of a row; the traditional one of ``width`` pixels when None.

    Returns
    -------
    ndarray, shape (nrows * row_clocks,)
    """
    return (layout or RowLayout.traditional(channel_image.shape[1])).to_stream(channel_image, direction, fill)


def from_stream(stream, nrows, width, direction, layout=None):
    """Inverse of `to_stream`: drop the clocks without a pixel and restore column order.

    Parameters
    ----------
    stream : ndarray, shape (nrows * row_clocks,)
        Stream on the clock grid, as laid out by `to_stream`.
    nrows : int
        Number of rows of the channel image.
    width : int
        Number of columns of the channel image.
    direction : int
        Readout direction of the channel, +1 or -1.
    layout : `RowLayout` or None
        Clock layout of a row; the traditional one of ``width`` pixels when None.

    Returns
    -------
    ndarray, shape (nrows, width)
        Contiguous channel image in the detector frame.
    """
    return (layout or RowLayout.traditional(width)).from_stream(stream, nrows, direction)


def stream_valid(nrows, width, layout=None):
    """Return a boolean stream that is True at pixel clocks and False elsewhere.

    Parameters
    ----------
    nrows : int
        Number of rows of the channel image.
    width : int
        Number of columns of the channel image.
    layout : `RowLayout` or None
        Clock layout of a row; the traditional one of ``width`` pixels when None.

    Returns
    -------
    ndarray of bool, shape (nrows * row_clocks,)
    """
    return (layout or RowLayout.traditional(width)).stream_valid(nrows)


def channel_blocks(shape, fastaxis, slowaxis):
    """Science-frame slices ``(rows, cols)`` covered by each output channel.

    Channels are column blocks in the detector frame; in the science frame
    they are column blocks for the NIRCam geometries and row blocks for
    NIRISS.  The blocks are returned in channel order.

    Parameters
    ----------
    shape : tuple of int
        Shape ``(nrows, ncols)`` of the science-frame image.
    fastaxis : int
        Value of ``meta.subarray.fastaxis``.
    slowaxis : int
        Value of ``meta.subarray.slowaxis``.

    Returns
    -------
    list of tuple of slice
        ``(rows, cols)`` science-frame slices, one per channel.
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

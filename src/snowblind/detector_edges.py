"""Noise that the columns near the physical edges of a detector carry differently from the rest.

Two effects, seen on NIRSpec NRS2 and absent on NRS1 and the NIRCam
detectors tested: within about 200 columns of the two physical edges of the
detector (the outermost columns of channels 0 and 3 in the detector frame)
the pixels carry the 1/f noise of their channel at only part of the
interior's amplitude, about half at the edge itself; and they carry a noise
of their own, common along a readout row and shared by the two edges,
which the interior does not have.  A stream filter that treats every pixel
of a row alike over-subtracts the first and leaves the second.

Everything here works in the detector frame (rows along the slow axis,
columns along the fast axis, see `snowblind.readout`), on
background-subtracted images with a mask of usable pixels, and measures
per exposure; on a detector without the effects the measurements come
out as 1 (coupling) and nothing (edge noise) and change nothing.
"""
import logging

import numpy as np
from scipy.ndimage import uniform_filter1d

from . import readout

log = logging.getLogger(__name__)

__all__ = [
    "coupling_sums",
    "coupling_profiles",
    "edge_stream_sums",
    "edge_bin_gains",
    "edge_profiles",
    "weighted_edge_series",
    "edge_stream_gain",
    "filter_edge_series",
    "edge_columns",
    "edge_reach",
]

COUPLING_BIN = 16  # columns per bin of a coupling profile
COUPLING_SMOOTH = 16  # readout rows the row means are smoothed over before the regression
COUPLING_TOLERANCE = 0.05  # a profile within this of 1 everywhere is no profile
EDGE_WINDOW = 96  # columns from the reference border averaged into an edge series
EDGE_REACH = 256  # columns from the reference border over which the edge noise profile is measured
EDGE_SMOOTH = 32  # readout rows the row means are smoothed over for the edge noise profile
MIN_FILL = 0.25  # fraction of a bin's pixels that must be usable for a row to count


def far_half(channel, nbins):
    """The bins of a channel on the side away from the nearest physical edge of the detector.

    Channels are numbered from one edge of the detector to the other, so
    the first half of them have their nearest physical edge on the low
    column side and the second half on the high side.

    Parameters
    ----------
    channel : int
        Channel index.
    nbins : int
        Number of bins (or columns) across the channel.

    Returns
    -------
    slice
    """
    if channel < readout.NOUTPUTS // 2:
        return slice(nbins // 2, nbins)
    return slice(0, nbins // 2)


def _binned_row_sums(values, good, columns, bin_width):
    """Sum and count of the usable pixels of every row in bins of ``bin_width`` columns of ``values[:, columns]``."""
    nrows = values.shape[0]
    block = values[:, columns]
    block_good = good[:, columns]
    nbins = block.shape[1] // bin_width
    shape = (nrows, nbins, bin_width)
    sums = block[:, : nbins * bin_width].reshape(shape).sum(axis=2)
    counts = block_good[:, : nbins * bin_width].reshape(shape).sum(axis=2).astype(np.float64)
    return sums, counts


def _smoothed_means(sums, counts, smooth):
    """Row means from sums and counts smoothed over ``smooth`` rows, and whether a row has enough pixels."""
    sums = uniform_filter1d(sums, smooth, axis=0, mode="nearest")
    counts = uniform_filter1d(counts, smooth, axis=0, mode="nearest")
    return sums, counts


def coupling_sums(image_det, mask_det, bin_width=COUPLING_BIN, smooth=COUPLING_SMOOTH):
    """Regression sums of each column bin's row means on the row means of the channel's far half.

    For every channel the usable pixels of every readout row are averaged in
    bins of ``bin_width`` columns and over the half of the channel away from
    the nearest physical edge (`far_half`), both smoothed over ``smooth``
    rows; the sums of products and of squares over the rows are returned, so
    that several images (the group differences of an integration) can be
    added up before `coupling_profiles` divides them.

    Parameters
    ----------
    image_det : ndarray, shape (nrows, ncols)
        Background-subtracted image in the detector frame.
    mask_det : ndarray of bool, shape (nrows, ncols)
        True for usable pixels.

    Returns
    -------
    xy, xx : ndarray, shape (NOUTPUTS, nbins)
        Sums of bin mean times far-half mean, and of the far-half mean
        squared, over the rows where both have enough pixels.
    """
    nrows, ncols = image_det.shape
    width = readout.channel_width(ncols)
    nbins = width // bin_width
    good = mask_det & np.isfinite(image_det)
    values = np.where(good, image_det, 0.0).astype(np.float64)
    xy = np.zeros((readout.NOUTPUTS, nbins))
    xx = np.zeros((readout.NOUTPUTS, nbins))
    for k, columns in enumerate(readout.channel_slices(ncols)):
        sums, counts = _smoothed_means(*_binned_row_sums(values, good, columns, bin_width), smooth)
        far = far_half(k, nbins)
        ref_sum = sums[:, far].sum(axis=1)
        ref_count = counts[:, far].sum(axis=1)
        ref_ok = ref_count >= MIN_FILL * (far.stop - far.start) * bin_width
        ref = np.where(ref_ok, ref_sum / np.maximum(ref_count, 1e-9), 0.0)
        bin_ok = counts >= MIN_FILL * bin_width
        means = np.where(bin_ok, sums / np.maximum(counts, 1e-9), 0.0)
        ok = bin_ok & ref_ok[:, np.newaxis]
        xy[k] = np.sum(means * ref[:, np.newaxis] * ok, axis=0)
        xx[k] = np.sum(ref[:, np.newaxis] ** 2 * ok, axis=0)
    return xy, xx


def coupling_profiles(xy, xx, width, bin_width=COUPLING_BIN, tolerance=COUPLING_TOLERANCE):
    """The coupling of every column of each channel to the channel's 1/f noise, from `coupling_sums`.

    The ratio of the sums is the least-squares gain of a bin against the
    far half, 1 for a bin that carries the same noise as the far half.  The
    far half itself is set to 1 (it is the reference), the gains are
    limited to 0 to 1.5 and interpolated between the bin centers.

    Parameters
    ----------
    xy, xx : ndarray, shape (NOUTPUTS, nbins)
        Sums from `coupling_sums`, added over the images measured.
    width : int
        Columns per channel.
    bin_width : int
        Columns per bin, as in `coupling_sums`.
    tolerance : float
        A channel whose gains are all within this of 1 gets None.

    Returns
    -------
    list of ndarray of float32 or None
        One profile of ``width`` values per channel, in detector column
        order, or None for a channel that couples like its far half
        everywhere.
    """
    nbins = xy.shape[1]
    centers = (np.arange(nbins) + 0.5) * bin_width
    profiles = []
    for k in range(xy.shape[0]):
        with np.errstate(invalid="ignore", divide="ignore"):
            gains = np.where(xx[k] > 0, xy[k] / xx[k], 1.0)
        gains = np.clip(gains, 0.0, 1.5)
        gains[far_half(k, nbins)] = 1.0
        if np.all(np.abs(gains - 1.0) <= tolerance):
            profiles.append(None)
            continue
        profiles.append(np.interp(np.arange(width) + 0.5, centers, gains).astype(np.float32))
    return profiles


def edge_reach(ncols, reach=EDGE_REACH, border=readout.NREFPIX, bin_width=COUPLING_BIN):
    """The columns per edge an edge profile covers: ``reach``, or what fits in half a narrower frame, whole bins."""
    return min(reach, (ncols // 2 - border) // bin_width * bin_width)


def edge_columns(ncols, reach=EDGE_REACH, border=readout.NREFPIX):
    """The column slices of the two physical edges, ``reach`` columns in from the reference border.

    Parameters
    ----------
    ncols : int
        Detector-frame columns.
    reach : int
        Columns per edge.
    border : int
        Reference pixel border left out.

    Returns
    -------
    list of slice
        The low edge and the high edge, in increasing column order; the
        high edge's first column in read-from-the-edge order is its last.
    """
    return [slice(border, border + reach), slice(ncols - border - reach, ncols - border)]


def edge_stream_sums(
    residual_det, mask_det, window=EDGE_WINDOW, reach=EDGE_REACH, bin_width=COUPLING_BIN, smooth=EDGE_SMOOTH
):
    """The residual's mean per readout row at each physical edge, and the sums for its column profile.

    The edge series of an edge is the mean of the usable pixels of each row
    over the ``window`` columns next to the reference border, minus the
    row's mean over the columns beyond ``reach`` of both edges, so that
    what the whole row has left is not counted as edge noise.  For the
    profile, the row means of each bin of ``bin_width`` columns out to
    ``reach`` (with the same interior mean removed) are regressed on the
    series (both smoothed over ``smooth`` rows); a bin inside the window is
    regressed on the series without itself, so that its own pixel noise
    does not count as signal.  The bin means and counts are returned as
    well, for `weighted_edge_series` once the profile is known.

    Parameters
    ----------
    residual_det : ndarray, shape (nrows, ncols)
        Cleaned, background-subtracted image in the detector frame.
    mask_det : ndarray of bool, shape (nrows, ncols)
        True for usable pixels.

    Returns
    -------
    series : ndarray, shape (2, nrows)
        The edge series of the low and the high edge, NaN where a row has
        no usable pixel in the window.
    xy, xx : ndarray, shape (2, nbins)
        Regression sums of each bin, counted from the edge inward.
    means, counts : ndarray, shape (2, nbins, nrows)
        Mean of the usable pixels of each bin in each row (0 where none)
        and their number.
    """
    nrows, ncols = residual_det.shape
    good = mask_det & np.isfinite(residual_det)
    values = np.where(good, residual_det, 0.0).astype(np.float64)
    reach = edge_reach(ncols, reach, bin_width=bin_width)
    nbins = reach // bin_width
    window_bins = min(window, reach) // bin_width
    edges = edge_columns(ncols, reach)
    interior = slice(edges[0].stop, edges[1].start)
    interior_count = good[:, interior].sum(axis=1)
    interior_mean = values[:, interior].sum(axis=1) / np.maximum(interior_count, 1)
    values = np.where(good, values - interior_mean[:, np.newaxis], 0.0)
    series = np.full((2, nrows), np.nan)
    xy = np.zeros((2, nbins))
    xx = np.zeros((2, nbins))
    bin_means = np.zeros((2, nbins, nrows))
    bin_counts = np.zeros((2, nbins, nrows))
    for e, columns in enumerate(edges):
        sums, counts = _binned_row_sums(values, good, columns, bin_width)
        if e == 1:
            sums, counts = sums[:, ::-1], counts[:, ::-1]
        bin_means[e] = (sums / np.maximum(counts, 1)).T
        bin_counts[e] = counts.T
        window_sum = sums[:, :window_bins].sum(axis=1)
        window_count = counts[:, :window_bins].sum(axis=1)
        series[e] = np.where(window_count > 0, window_sum / np.maximum(window_count, 1e-9), np.nan)
        sums, counts = _smoothed_means(sums, counts, smooth)
        window_sum = sums[:, :window_bins].sum(axis=1)
        window_count = counts[:, :window_bins].sum(axis=1)
        for b in range(nbins):
            ref_sum, ref_count = window_sum, window_count
            if b < window_bins:
                ref_sum, ref_count = window_sum - sums[:, b], window_count - counts[:, b]
            ok = (counts[:, b] >= MIN_FILL * bin_width) & (ref_count >= MIN_FILL * (window - bin_width))
            ref = np.where(ok, ref_sum / np.maximum(ref_count, 1e-9), 0.0)
            means = np.where(ok, sums[:, b] / np.maximum(counts[:, b], 1e-9), 0.0)
            xy[e, b] = np.sum(means * ref)
            xx[e, b] = np.sum(ref**2)
    return series, xy, xx, bin_means, bin_counts


def edge_bin_gains(xy, xx):
    """How much of its edge series each column bin from the edge inward carries, from `edge_stream_sums`.

    Parameters
    ----------
    xy, xx : ndarray, shape (2, nbins)
        Sums from `edge_stream_sums`, added over the images measured.

    Returns
    -------
    ndarray, shape (2, nbins)
        The least-squares gain of each bin, limited to 0 to 1.5; NaN for
        a bin without data.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        gains = np.where(xx > 0, xy / np.where(xx > 0, xx, 1.0), np.nan)
    return np.clip(gains, 0.0, 1.5)


def edge_profiles(xy, xx, bin_width=COUPLING_BIN):
    """How much of its edge series each column from the edge inward carries, from `edge_stream_sums`.

    Parameters
    ----------
    xy, xx : ndarray, shape (2, nbins)
        Sums from `edge_stream_sums`, added over the images measured.
    bin_width : int
        Columns per bin.

    Returns
    -------
    ndarray of float32, shape (2, nbins * bin_width)
        The gains of `edge_bin_gains` interpolated between bin centers;
        zeros for an edge without data.
    """
    bin_gains = edge_bin_gains(xy, xx)
    nbins = xy.shape[1]
    reach = nbins * bin_width
    centers = (np.arange(nbins) + 0.5) * bin_width
    columns = np.arange(reach) + 0.5
    profiles = np.zeros((2, reach), dtype=np.float32)
    for e in range(2):
        measured = np.isfinite(bin_gains[e])
        if measured.any():
            profiles[e] = np.interp(columns, centers[measured], bin_gains[e][measured])
    return profiles


def weighted_edge_series(means, counts, bin_gains):
    """The edge noise of each row from the bin means, weighted by the bins' gains: the matched combination.

    For bin means ``m_b = p_b e + n_b`` with white noise of variance
    inversely proportional to the count, the least-squares ``e`` is
    ``sum(p_b c_b m_b) / sum(p_b^2 c_b)``.  Its scale is that of the edge
    series the gains were measured against, so the profile from the same
    sums applies to it as it is.

    Parameters
    ----------
    means, counts : ndarray, shape (2, nbins, nrows)
        Bin means and counts from `edge_stream_sums`.
    bin_gains : ndarray, shape (2, nbins)
        Gains from `edge_bin_gains`.

    Returns
    -------
    ndarray, shape (2, nrows)
        The combined series per edge, NaN where no bin with a gain has a
        pixel.
    """
    gains = np.where(np.isfinite(bin_gains), bin_gains, 0.0)[:, :, np.newaxis]
    numerator = np.sum(gains * counts * means, axis=1)
    denominator = np.sum(gains**2 * counts, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / np.where(denominator > 0, denominator, 1.0), np.nan)


def edge_stream_gain(series, nfft, smooth_bins=9, min_excess=4.0, min_band_ratio=2.0):
    """Wiener gain along the readout rows for an edge's series, from the mean periodogram over the differences.

    The white level is the mean of the periodogram at periods under 4
    rows, without bins above five times their median.  The gain is
    ``1 - white / spectrum`` where the smoothed spectrum exceeds the white
    level by more than ``min_excess`` times its expected scatter (the
    white level over the square root of the number of bins averaged), and
    zero elsewhere; and only when the lowest twentieth of the frequencies
    as a whole exceeds the white level by that much, so that a series of
    pixel noise alone gets no gain.

    Parameters
    ----------
    series : ndarray, shape (ndiff, nrows)
        The edge series of one edge for every difference measured; NaN
        where a row had no pixel.
    nfft : int
        Length of the transform the gain is returned for.
    smooth_bins : int
        Frequency bins the periodogram is smoothed over.
    min_excess : float
        Detection threshold in units of the scatter of the smoothed mean
        periodogram.
    min_band_ratio : float
        The lowest twentieth of the frequencies must also exceed the
        white level by this factor, so that edge noise too weak for the
        filter to separate from the series' own noise is left alone.  With
        the profile-weighted series the NRS2 darks show 6 to 7 and the
        NRS1 darks 1.0; an IFU exposure on NRS2 shows 1.2 to 1.8, and at a
        threshold of 1.5 the filter took 0.5 DN per difference out of its
        edge without changing the edge residual, so the threshold is 2.

    Returns
    -------
    ndarray or None
        Gain on ``np.fft.rfftfreq(nfft)``, zero at zero frequency; None
        when no frequency has power above white.
    """
    x = np.asarray(series, dtype=np.float64)
    x = np.nan_to_num(x - np.nanmean(x, axis=1, keepdims=True))
    nrows = x.shape[1]
    spectrum = np.mean(np.abs(np.fft.rfft(x, axis=1)) ** 2, axis=0)
    freq = np.fft.rfftfreq(nrows)
    high = spectrum[freq > 0.25]
    high = high[high <= 5 * np.median(high)]
    white = np.mean(high) if high.size else 0.0
    if not np.isfinite(white) or white <= 0:
        return None
    # the edge noise is red: it has to show in the lowest twentieth of the frequencies as a whole
    band = slice(1, max(2, freq.size // 20))
    band_mean = np.mean(spectrum[band])
    log.debug(f"Edge series: the lowest twentieth of the frequencies holds {band_mean / white:.2f} times the white power")
    if band_mean <= white * max(min_band_ratio, 1.0 + min_excess / np.sqrt(x.shape[0] * (band.stop - band.start))):
        return None
    smooth = uniform_filter1d(spectrum, smooth_bins, mode="nearest")
    excess = smooth > white * (1.0 + min_excess / np.sqrt(x.shape[0] * smooth_bins))
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where(excess, 1.0 - white / smooth, 0.0)
    gain[0] = 0.0
    if not np.any(gain > 0):
        return None
    return np.interp(np.fft.rfftfreq(nfft), freq, gain)


def filter_edge_series(series, gain, nfft):
    """Apply a gain along the readout rows to edge series, with zero padding to ``nfft``.

    Parameters
    ----------
    series : ndarray, shape (..., nrows)
        Edge series; NaN rows count as zero after each series' mean is
        removed.
    gain : ndarray, shape (nfft // 2 + 1,)
        Gain from `edge_stream_gain`.
    nfft : int
        Transform length, at least ``nrows``.

    Returns
    -------
    ndarray of float32, same shape as ``series``
        The filtered series: the part of the edge noise the gain passes.
    """
    x = np.asarray(series, dtype=np.float64)
    x = np.nan_to_num(x - np.nanmean(x, axis=-1, keepdims=True))
    filtered = np.fft.irfft(np.fft.rfft(x, nfft, axis=-1) * gain, nfft, axis=-1)
    return filtered[..., : x.shape[-1]].astype(np.float32)

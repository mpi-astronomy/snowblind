"""Noise estimators that act on one output channel in the detector frame.

Each function takes a channel image and a boolean background mask (True for
pixels that may be used), both of shape (nrows, width) in the detector
frame, plus the readout ``direction`` of the channel, and returns the
estimated noise for every pixel of the channel as a float32 array of the
same shape.  The caller subtracts the estimate from the image.
"""
import warnings

import numpy as np

from . import readout
from .noise_spectrum import interpolate_gain

__all__ = ["row_median_noise", "boxcar_noise", "wiener_noise", "normalized_stream_filter", "boxcar_response", "FILTERS"]


def row_median_noise(channel_image, channel_mask, direction=1):
    """Estimate the noise as the masked median of each row of the channel.

    Rows with no usable pixel get an estimate of zero.  This reproduces
    ``jwst.clean_flicker_noise.clean_flicker_noise.median_clean`` with
    ``fit_by_channel=True`` and is the baseline the other estimators are
    compared against.

    Parameters
    ----------
    channel_image : ndarray, shape (nrows, width)
        Channel image in the detector frame; the estimate is in its units.
    channel_mask : ndarray of bool, shape (nrows, width)
        Background mask, True for pixels that may be used.
    direction : int
        Readout direction of the channel; not used by this estimator.

    Returns
    -------
    ndarray of float32, shape (nrows, width)
        The masked median of each row, repeated across the row.
    """
    masked = np.where(channel_mask, channel_image, np.nan).astype(np.float32)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
        stripes = np.nanmedian(masked, axis=1)
    stripes = np.nan_to_num(stripes, nan=0.0)
    return np.repeat(stripes[:, np.newaxis], channel_image.shape[1], axis=1).astype(np.float32)


def normalized_stream_filter(channel_image, channel_mask, direction, response, min_density):
    """Filter the masked readout stream of a channel with a given frequency response.

    The estimate on the full clock grid is the normalized masked convolution
    ``irfft(G rfft(m d)) / irfft(G rfft(m))``, where ``m`` is the mask on the
    clock grid and ``G`` the response on the rfft grid of the padded stream.
    The denominator is the local density of background samples after
    filtering.  Where it falls below ``min_density`` the ratio is not trusted
    and the estimate is blended toward the per-row masked median.

    Parameters
    ----------
    channel_image : ndarray, shape (nrows, width)
        Channel image in the detector frame; the estimate is in its units.
        Non-finite pixels are treated as masked.
    channel_mask : ndarray of bool, shape (nrows, width)
        Background mask, True for pixels that may be used.
    direction : int
        Readout direction of the channel.
    response : callable
        ``response(nfft) -> ndarray`` giving the real, non-negative gain on
        ``np.fft.rfftfreq(nfft)``.
    min_density : float
        Density below which the blend toward the row median starts.  Zero
        or negative returns the ratio without any blending.

    Returns
    -------
    ndarray of float32, shape (nrows, width)
        Noise estimate for every pixel of the channel.
    """
    nrows, width = channel_image.shape
    mask = channel_mask & np.isfinite(channel_image)
    data = np.where(mask, channel_image, 0.0).astype(np.float64)
    stream = readout.to_stream(data, direction)
    weights = readout.to_stream(mask.astype(np.float64), direction)
    nfft = 1 << int(np.ceil(np.log2(2 * stream.size)))
    gain = response(nfft)
    numerator = np.fft.irfft(gain * np.fft.rfft(stream, nfft), nfft)[: stream.size]
    density = np.fft.irfft(gain * np.fft.rfft(weights, nfft), nfft)[: stream.size]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(density > 0, numerator / density, 0.0)
    estimate = readout.from_stream(ratio, nrows, width, direction)
    if min_density <= 0:
        return estimate.astype(np.float32)
    blend = readout.from_stream(np.clip(density / min_density, 0.0, 1.0), nrows, width, direction)
    fallback = row_median_noise(channel_image, channel_mask)
    return (blend * estimate + (1.0 - blend) * fallback).astype(np.float32)


def boxcar_noise(channel_image, channel_mask, direction=1, window_length=4096, min_density=0.2):
    """Estimate the noise as a normalized running mean of ``window_length`` clocks along the readout stream.

    A sanity-check estimator: it is the normalized masked convolution of
    `normalized_stream_filter` with a boxcar response.  The density is the
    fraction of the window holding background samples, so ``min_density``
    is compared against that fraction.

    Parameters
    ----------
    channel_image : ndarray, shape (nrows, width)
        Channel image in the detector frame; the estimate is in its units.
    channel_mask : ndarray of bool, shape (nrows, width)
        Background mask, True for pixels that may be used.
    direction : int
        Readout direction of the channel, +1 or -1.
    window_length : int
        Length of the running mean in clocks.  A value longer than the
        stream is reduced to the stream length, with a warning.
    min_density : float
        Fraction of the window that must hold background samples; below it
        the estimate blends toward the row median, see
        `normalized_stream_filter`.

    Returns
    -------
    ndarray of float32, shape (nrows, width)
        Noise estimate for every pixel of the channel.
    """
    window = int(min(window_length, readout.stream_length(*channel_image.shape)))
    if window != window_length:
        warnings.warn(f"boxcar window_length {window_length} exceeds the stream length; using {window}", stacklevel=2)
    return normalized_stream_filter(channel_image, channel_mask, direction, boxcar_response(window), min_density)


def boxcar_response(window_length):
    """Return a ``response(nfft)`` callable for a centered boxcar of ``window_length`` clocks.

    Parameters
    ----------
    window_length : int
        Width of the boxcar in clocks.

    Returns
    -------
    callable
        ``response(nfft)`` returning the frequency response, on
        ``np.fft.rfftfreq(nfft)``, of a boxcar of ``window_length`` clocks
        with unit sum, centered on clock zero (its first half wraps around
        the end of the ``nfft``-point kernel).
    """
    def response(nfft):
        """Frequency response of the centered boxcar on the rfft grid of an ``nfft``-point transform.

        Parameters
        ----------
        nfft : int
            Transform length; the boxcar is laid out on ``nfft`` clocks.

        Returns
        -------
        ndarray, shape (nfft // 2 + 1,)
            Real part of the rfft of the boxcar kernel.
        """
        kernel = np.zeros(nfft)
        half = window_length // 2
        kernel[: window_length - half] = 1.0 / window_length
        kernel[nfft - half:] = 1.0 / window_length
        return np.fft.rfft(kernel).real

    return response


def wiener_noise(channel_image, channel_mask, direction=1, min_density=0.2, gain=None, protect_period=None):
    """Estimate the noise with a Wiener gain derived from the channel's own spectrum.

    Parameters
    ----------
    channel_image : ndarray, shape (nrows, width)
        Channel image in the detector frame; the estimate is in its units.
    channel_mask : ndarray of bool, shape (nrows, width)
        Background mask, True for pixels that may be used.
    direction : int
        Readout direction of the channel, +1 or -1.
    min_density : float
        See `normalized_stream_filter`.
    gain : tuple of ndarray
        ``(freq, gain)`` for this channel, as produced by
        `snowblind.stream_clean.channel_gain`.  The gain is a low-pass
        response with unit value at zero frequency, so the normalized
        convolution is well defined.
    protect_period : float or None
        If given, the component of the estimate slower than this many
        clocks is removed again (as a normalized boxcar mean of that
        length), leaving anything that slow to the background model.  This
        is the well-posed form of tapering the gain to zero at the lowest
        frequencies, which cannot be done inside a normalized convolution.
        A value longer than the stream is reduced to the stream length,
        with a warning.

    Returns
    -------
    ndarray of float32, shape (nrows, width)
        Noise estimate for every pixel of the channel.

    Raises
    ------
    ValueError
        If ``gain`` is None.
    """
    if gain is None:
        raise ValueError("wiener_noise needs the (freq, gain) pair estimated for this channel")
    freq, values = gain

    def response(nfft):
        """Interpolate the channel's gain onto the rfft grid of an ``nfft``-point transform.

        Parameters
        ----------
        nfft : int
            Transform length.

        Returns
        -------
        ndarray, shape (nfft // 2 + 1,)
            Gain at ``np.fft.rfftfreq(nfft)``.
        """
        return interpolate_gain(freq, values, nfft)

    estimate = normalized_stream_filter(channel_image, channel_mask, direction, response, min_density)
    if protect_period is not None:
        period = int(min(protect_period, readout.stream_length(*channel_image.shape)))
        if period != int(protect_period):
            warnings.warn(f"protect_period {protect_period} exceeds the stream length; using {period}", stacklevel=2)
        slow = normalized_stream_filter(estimate, np.ones_like(channel_mask), direction, boxcar_response(period), 0.0)
        estimate = estimate - slow
    return estimate.astype(np.float32)


FILTERS = {
    "row_median": row_median_noise,
    "boxcar": boxcar_noise,
    "wiener": wiener_noise,
}

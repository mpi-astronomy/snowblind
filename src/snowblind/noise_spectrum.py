"""Noise spectrum estimation for gapped readout streams.

A "stream" is a channel image laid out on its clock grid (see
`snowblind.readout`), with a boolean ``valid`` array marking the clocks
that hold usable background samples.  The autocorrelation of a gapped
stream is estimated without bias as the autocorrelation of ``valid * data``
divided by that of ``valid``, both by FFT.  The spectrum is the Fourier
transform of the Hann-windowed autocorrelation (Blackman-Tukey), normalized
so that its mean over frequency equals the sample variance.  A power law
plus white floor is fitted to the log-binned spectrum, and the Wiener gain
follows from the white floor.
"""
import numpy as np
from scipy import optimize

from . import readout

__all__ = [
    "channel_spectra",
    "masked_autocorrelation",
    "spectrum_from_autocorrelation",
    "estimate_spectrum",
    "log_binned_spectrum",
    "smooth_spectrum",
    "power_law",
    "fit_power_law",
    "wiener_gain",
    "taper_low_frequencies",
    "taper_high_frequencies",
    "interpolate_gain",
    "synthetic_stream",
]


def masked_autocorrelation(stream, valid, max_lag):
    """Unbiased autocorrelation of a gapped stream for lags 0 to ``max_lag``.

    The masked mean is removed first.  Lags with no valid pair get zero.

    Parameters
    ----------
    stream : ndarray, shape (n,)
    valid : ndarray of bool, shape (n,)
    max_lag : int

    Returns
    -------
    acf : ndarray, shape (max_lag + 1,)
        ``sum(m_t m_{t+l} d_t d_{t+l}) / sum(m_t m_{t+l})``.
    pairs : ndarray, shape (max_lag + 1,)
        Number of valid pairs at each lag.
    """
    valid = np.asarray(valid, dtype=bool)
    data = np.asarray(stream, dtype=np.float64)
    if not valid.any():
        return np.zeros(max_lag + 1), np.zeros(max_lag + 1)
    centered = np.where(valid, data - data[valid].mean(), 0.0)
    weights = valid.astype(np.float64)
    nfft = 1 << int(np.ceil(np.log2(centered.size + max_lag + 1)))
    acf_data = np.fft.irfft(np.abs(np.fft.rfft(centered, nfft)) ** 2, nfft)[: max_lag + 1]
    pairs = np.rint(np.fft.irfft(np.abs(np.fft.rfft(weights, nfft)) ** 2, nfft)[: max_lag + 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        acf = np.where(pairs > 0, acf_data / pairs, 0.0)
    return acf, pairs


def spectrum_from_autocorrelation(acf):
    """Blackman-Tukey spectrum of a one-sided autocorrelation.

    Parameters
    ----------
    acf : ndarray, shape (L + 1,)
        Autocorrelation for lags 0 to L.

    Returns
    -------
    freq : ndarray, shape (L + 1,)
        Frequencies in cycles per clock, from 0 to 0.5.
    spectrum : ndarray, shape (L + 1,)
        Power per unit frequency, in the normalization where the mean over
        all ``2 L`` Fourier bins equals ``acf[0]``.
    """
    max_lag = acf.size - 1
    lags = np.arange(max_lag + 1)
    window = 0.5 * (1.0 + np.cos(np.pi * lags / max_lag))
    windowed = acf * window
    symmetric = np.concatenate([windowed, windowed[-2:0:-1]])
    spectrum = np.fft.rfft(symmetric).real
    freq = np.fft.rfftfreq(symmetric.size, d=1.0)
    return freq, spectrum


def channel_spectra(image_det, mask_det, first_direction=1, max_lag=65536):
    """Spectrum of every output channel of one detector-frame image.

    Non-finite pixels are treated as masked.  The autocorrelation length is
    capped at a quarter of the stream length.

    Parameters
    ----------
    image_det, mask_det : ndarray, shape (nrows, ncols)
        Background-subtracted image and background mask in the detector frame.
    first_direction : int
    max_lag : int

    Returns
    -------
    freq : ndarray
    spectra : list
        One spectrum per channel, or None for a channel without usable pixels.
    """
    nrows, ncols = image_det.shape
    width = readout.channel_width(ncols)
    lag = min(max_lag, readout.stream_length(nrows, width) // 4)
    freq = np.fft.rfftfreq(2 * lag, d=1.0)
    spectra = []
    for channel, columns in enumerate(readout.channel_slices(ncols)):
        direction = readout.channel_direction(channel, first_direction)
        block = np.ascontiguousarray(image_det[:, columns])
        good = np.ascontiguousarray(mask_det[:, columns]) & np.isfinite(block)
        if not good.any():
            spectra.append(None)
            continue
        stream = readout.to_stream(np.where(good, block, 0.0), direction)
        valid = readout.to_stream(good, direction, fill=False)
        acf, _pairs = masked_autocorrelation(stream, valid, lag)
        freq, spectrum = spectrum_from_autocorrelation(acf)
        spectra.append(spectrum)
    return freq, spectra


def estimate_spectrum(streams, valids, max_lag):
    """Median spectrum over several realizations of the same noise.

    Parameters
    ----------
    streams : sequence of ndarray
        One stream per group difference.
    valids : sequence of ndarray of bool
        Matching validity arrays.
    max_lag : int

    Returns
    -------
    freq : ndarray
    spectrum : ndarray
        Median over the input streams of their individual spectra.
    """
    spectra = []
    freq = None
    for stream, valid in zip(streams, valids):
        acf, _pairs = masked_autocorrelation(stream, valid, max_lag)
        freq, spectrum = spectrum_from_autocorrelation(acf)
        spectra.append(spectrum)
    return freq, np.median(spectra, axis=0)


def log_binned_spectrum(freq, spectrum, nbins=48):
    """Median of the spectrum in logarithmically spaced frequency bins.

    The zero frequency is dropped.  Empty bins are omitted.

    Returns
    -------
    bin_freq : ndarray
        Median frequency of the samples in each bin that holds data.
    bin_spectrum : ndarray
        Median power in each such bin.
    """
    positive = freq > 0
    f = freq[positive]
    s = spectrum[positive]
    edges = np.geomspace(f.min() * (1 - 1e-9), f.max() * (1 + 1e-9), nbins + 1)
    index = np.digitize(f, edges) - 1
    bin_freq, bin_spectrum = [], []
    for k in range(nbins):
        selected = index == k
        if selected.any():
            bin_freq.append(np.median(f[selected]))
            bin_spectrum.append(np.median(s[selected]))
    return np.array(bin_freq), np.array(bin_spectrum)


def smooth_spectrum(freq, spectrum, nbins=48):
    """Spectrum smoothed by log-binning and interpolated back onto ``freq``.

    Interpolation is linear in log frequency and log power; the zero
    frequency gets the lowest-bin value.
    """
    bin_freq, bin_spectrum = log_binned_spectrum(freq, spectrum, nbins)
    positive = bin_spectrum > 0
    log_s = np.interp(
        np.log(np.where(freq > 0, freq, bin_freq[0])), np.log(bin_freq[positive]), np.log(bin_spectrum[positive])
    )
    return np.exp(log_s)


def power_law(freq, amplitude, alpha, white):
    """``amplitude * freq**-alpha + white``."""
    with np.errstate(divide="ignore"):
        return amplitude * np.power(freq, -alpha) + white


LEAKAGE_BINS = 2.5


def fit_power_law(freq, spectrum, alpha_bounds=(0.5, 1.5), fixed_alpha=None, nbins=48, min_freq=None):
    """Fit ``amplitude * f**-alpha + white`` to the log-binned spectrum.

    The fit is done in log power so that each bin counts equally.  The
    lowest Fourier bins are excluded because power at periods longer than
    the autocorrelation window leaks into them through the Hann window.

    Parameters
    ----------
    freq, spectrum : ndarray
    alpha_bounds : tuple of float
        Allowed range of the slope.
    fixed_alpha : float or None
        If given, only ``amplitude`` and ``white`` are fitted.
    nbins : int
    min_freq : float or None
        Lowest frequency used.  Default is ``LEAKAGE_BINS`` times the
        frequency resolution.

    Returns
    -------
    amplitude, alpha, white : float
    """
    if min_freq is None:
        min_freq = LEAKAGE_BINS * freq[1]
    usable = freq >= min_freq
    bin_freq, bin_spectrum = log_binned_spectrum(freq[usable], spectrum[usable], nbins)
    good = bin_spectrum > 0
    if good.sum() < 3:
        raise ValueError("fit_power_law needs at least three log bins with positive power")
    f, s = bin_freq[good], bin_spectrum[good]
    white0 = np.median(s[f > f.max() / 3])
    amp0 = max(s[0] - white0, 1e-3 * white0) * f[0]
    if fixed_alpha is None:
        def residual(p):
            return np.log(power_law(f, np.exp(p[0]), p[1], np.exp(p[2]))) - np.log(s)

        alpha0 = float(np.clip(1.0, *alpha_bounds))
        result = optimize.least_squares(
            residual,
            [np.log(amp0), alpha0, np.log(white0)],
            bounds=([-np.inf, alpha_bounds[0], -np.inf], [np.inf, alpha_bounds[1], np.inf]),
        )
        return float(np.exp(result.x[0])), float(result.x[1]), float(np.exp(result.x[2]))

    def residual_fixed(p):
        return np.log(power_law(f, np.exp(p[0]), fixed_alpha, np.exp(p[1]))) - np.log(s)

    result = optimize.least_squares(residual_fixed, [np.log(amp0), np.log(white0)])
    return float(np.exp(result.x[0])), float(fixed_alpha), float(np.exp(result.x[1]))


def wiener_gain(spectrum, white):
    """``max(S - W, 0) / S`` with non-finite or non-positive S giving zero gain."""
    with np.errstate(invalid="ignore", divide="ignore"):
        gain = (spectrum - white) / spectrum
    gain = np.where(np.isfinite(gain) & (spectrum > 0), gain, 0.0)
    return np.clip(gain, 0.0, 1.0)


def taper_low_frequencies(freq, gain, cutoff_period):
    """Set the gain to zero below half the cutoff frequency and ramp it to full by the cutoff.

    Parameters
    ----------
    freq : ndarray
        Cycles per clock.
    gain : ndarray
    cutoff_period : float
        Period in clocks; anything slower is left to the background model.
    """
    cutoff = 1.0 / cutoff_period
    ramp = np.clip((freq - 0.5 * cutoff) / (0.5 * cutoff), 0.0, 1.0)
    ramp = 0.5 * (1.0 - np.cos(np.pi * ramp))
    return gain * ramp


def taper_high_frequencies(freq, gain, min_period):
    """Ramp the gain to zero between periods of 2 ``min_period`` and ``min_period`` clocks.

    Compact sources that escaped the scene mask live at high stream
    frequencies, where the 1/f power is weak; cutting the gain there makes
    the filter blind to them at a small cost in cleaning.
    """
    cutoff = 1.0 / min_period
    ramp = np.clip((cutoff - freq) / (0.5 * cutoff), 0.0, 1.0)
    ramp = 0.5 * (1.0 - np.cos(np.pi * ramp))
    return gain * ramp


def interpolate_gain(freq, gain, nfft):
    """Interpolate a gain defined on ``freq`` onto the rfft grid of an ``nfft``-point transform."""
    target = np.fft.rfftfreq(nfft, d=1.0)
    return np.interp(target, freq, gain)


def synthetic_stream(n, alpha, amplitude, white_sigma, rng):
    """White noise plus a power-law process with spectrum ``amplitude * f**-alpha``.

    The normalization matches `spectrum_from_autocorrelation`: white noise
    of standard deviation ``white_sigma`` has a flat spectrum of value
    ``white_sigma**2``.
    """
    white = rng.normal(scale=white_sigma, size=n)
    seed = rng.normal(size=n)
    freq = np.fft.rfftfreq(n, d=1.0)
    with np.errstate(divide="ignore"):
        response = np.where(freq > 0, np.sqrt(amplitude) * np.power(freq, -alpha / 2.0), 0.0)
    colored = np.fft.irfft(np.fft.rfft(seed) * response, n)
    return white + colored

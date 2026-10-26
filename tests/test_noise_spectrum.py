import numpy as np
import pytest

from snowblind.noise_spectrum import (
    estimate_spectrum,
    fit_power_law,
    interpolate_gain,
    log_binned_spectrum,
    masked_autocorrelation,
    power_law,
    smooth_spectrum,
    spectrum_from_autocorrelation,
    synthetic_stream,
    taper_high_frequencies,
    taper_low_frequencies,
    wiener_gain,
)

N = 2**18
MAX_LAG = 2048


def white_spectrum(sigma, seed, valid=None):
    """Spectrum of one white-noise stream of standard deviation sigma."""
    rng = np.random.default_rng(seed)
    stream = rng.normal(scale=sigma, size=N)
    if valid is None:
        valid = np.ones(N, dtype=bool)
    acf, _pairs = masked_autocorrelation(stream, valid, MAX_LAG)
    return spectrum_from_autocorrelation(acf)


def test_autocorrelation_lag_zero_is_variance():
    """masked_autocorrelation of white noise with sigma 3 has acf[0] near 9 and pairs[0] equal to n."""
    rng = np.random.default_rng(1)
    stream = rng.normal(scale=3.0, size=N)
    valid = np.ones(N, dtype=bool)

    acf, pairs = masked_autocorrelation(stream, valid, MAX_LAG)

    assert abs(acf[0] - 9.0) < 0.1
    assert pairs[0] == N


def test_autocorrelation_pairs_count_gaps():
    """masked_autocorrelation with the first half masked reports n/2 pairs at lag 0 and none beyond the valid span."""
    stream = np.zeros(N)
    valid = np.zeros(N, dtype=bool)
    valid[N // 2:] = True

    _acf, pairs = masked_autocorrelation(stream, valid, MAX_LAG)

    assert pairs[0] == N // 2
    assert pairs[MAX_LAG] == N // 2 - MAX_LAG


def test_autocorrelation_all_masked_is_zero():
    """masked_autocorrelation with nothing valid returns zeros without error."""
    acf, pairs = masked_autocorrelation(np.ones(100), np.zeros(100, dtype=bool), 10)

    np.testing.assert_array_equal(acf, 0.0)
    np.testing.assert_array_equal(pairs, 0.0)


def test_white_spectrum_mean_is_variance():
    """spectrum_from_autocorrelation of white noise with sigma 3 averages to 9 within 3 percent."""
    _freq, spectrum = white_spectrum(3.0, 2)

    assert abs(spectrum.mean() / 9.0 - 1.0) < 0.03


def test_white_spectrum_is_flat():
    """spectrum_from_autocorrelation of white noise has equal low- and high-frequency halves within 5 percent."""
    freq, spectrum = white_spectrum(1.0, 3)

    low = spectrum[(freq > 0) & (freq < 0.25)].mean()
    high = spectrum[freq >= 0.25].mean()
    assert abs(low / high - 1.0) < 0.05


def test_masked_white_spectrum_unbiased():
    """spectrum_from_autocorrelation with 40 percent random masking plus a 20000-clock gap still averages to the variance."""
    rng = np.random.default_rng(4)
    valid = rng.random(N) > 0.4
    valid[100000:120000] = False

    _freq, spectrum = white_spectrum(2.0, 4, valid=valid)

    assert abs(spectrum.mean() / 4.0 - 1.0) < 0.05


def test_power_law_fit_recovers_alpha():
    """fit_power_law on a synthetic alpha=1 process plus white noise returns alpha within 0.1 and white within 10 percent."""
    rng = np.random.default_rng(5)
    stream = synthetic_stream(N, 1.0, 0.02, 1.0, rng)
    valid = np.ones(N, dtype=bool)

    acf, _ = masked_autocorrelation(stream, valid, MAX_LAG)
    freq, spectrum = spectrum_from_autocorrelation(acf)
    _amplitude, alpha, white = fit_power_law(freq, spectrum)

    assert abs(alpha - 1.0) < 0.1
    assert abs(white - 1.0) < 0.1


def test_power_law_fit_with_mask_matches_unmasked():
    """fit_power_law on the same alpha=1.3 process with and without 40 percent masking agrees in alpha within 0.1."""
    rng = np.random.default_rng(6)
    stream = synthetic_stream(N, 1.3, 0.01, 1.0, rng)
    valid_all = np.ones(N, dtype=bool)
    valid_gap = rng.random(N) > 0.4

    acf_all, _ = masked_autocorrelation(stream, valid_all, MAX_LAG)
    acf_gap, _ = masked_autocorrelation(stream, valid_gap, MAX_LAG)
    _, alpha_all, _ = fit_power_law(*spectrum_from_autocorrelation(acf_all))
    _, alpha_gap, _ = fit_power_law(*spectrum_from_autocorrelation(acf_gap))

    assert abs(alpha_all - alpha_gap) < 0.1


def test_power_law_fit_exact_model():
    """fit_power_law on an exact model spectrum returns the model parameters to within the binning bias."""
    freq = np.fft.rfftfreq(2 * MAX_LAG, d=1.0)
    spectrum = power_law(freq, 0.05, 0.8, 2.0)

    amplitude, alpha, white = fit_power_law(freq, spectrum)

    assert abs(amplitude / 0.05 - 1.0) < 0.05
    assert abs(alpha - 0.8) < 0.01
    assert abs(white / 2.0 - 1.0) < 0.01


def test_power_law_fit_fixed_alpha():
    """fit_power_law with fixed_alpha fits only amplitude and white."""
    freq = np.fft.rfftfreq(2 * MAX_LAG, d=1.0)
    spectrum = power_law(freq, 0.05, 1.0, 2.0)

    amplitude, alpha, white = fit_power_law(freq, spectrum, fixed_alpha=1.0)

    assert alpha == 1.0
    assert abs(amplitude / 0.05 - 1.0) < 0.05
    assert abs(white / 2.0 - 1.0) < 0.01


def test_estimate_spectrum_median_over_streams():
    """estimate_spectrum over three white streams of sigma 1, 1 and 10 returns the sigma-1 level."""
    rng = np.random.default_rng(7)
    streams = [rng.normal(size=N), rng.normal(size=N), rng.normal(scale=10.0, size=N)]
    valids = [np.ones(N, dtype=bool)] * 3

    _freq, spectrum = estimate_spectrum(streams, valids, MAX_LAG)

    assert abs(spectrum.mean() - 1.0) < 0.1


def test_log_binned_spectrum_drops_zero_frequency():
    """log_binned_spectrum returns only positive bin frequencies in increasing order."""
    freq = np.fft.rfftfreq(2 * MAX_LAG, d=1.0)
    spectrum = np.ones_like(freq)

    bin_freq, bin_spectrum = log_binned_spectrum(freq, spectrum, nbins=20)

    assert bin_freq.min() > 0
    assert np.all(np.diff(bin_freq) > 0)
    np.testing.assert_allclose(bin_spectrum, 1.0)


def test_smooth_spectrum_tracks_model():
    """smooth_spectrum of a power law with 30 percent multiplicative scatter stays within 15 percent of the model."""
    rng = np.random.default_rng(8)
    freq = np.fft.rfftfreq(2 * MAX_LAG, d=1.0)
    model = power_law(freq, 0.05, 1.0, 2.0)
    noisy = model * rng.gamma(10.0, 0.1, size=freq.size)

    smooth = smooth_spectrum(freq, noisy)

    ratio = smooth[1:] / model[1:]
    assert abs(np.median(ratio) - 1.0) < 0.15


def test_wiener_gain_zero_at_white_floor():
    """wiener_gain is 0 where the spectrum equals the white level and 0.9 where it is ten times the level."""
    spectrum = np.array([2.0, 20.0, 1.0, 0.0, np.nan])

    gain = wiener_gain(spectrum, 2.0)

    np.testing.assert_allclose(gain, [0.0, 0.9, 0.0, 0.0, 0.0])


def test_taper_low_frequencies():
    """taper_low_frequencies zeros the gain below half the cutoff and leaves it unchanged above the cutoff."""
    freq = np.array([0.0, 0.0004, 0.001, 0.002, 0.1])
    gain = np.ones_like(freq)

    tapered = taper_low_frequencies(freq, gain, cutoff_period=1000.0)

    np.testing.assert_allclose(tapered[[0, 1]], 0.0)
    np.testing.assert_allclose(tapered[[3, 4]], 1.0)
    assert 0.0 < tapered[2] <= 1.0


def test_interpolate_gain_onto_fft_grid():
    """interpolate_gain maps a gain known on a coarse grid onto the rfft grid of a longer transform."""
    freq = np.linspace(0.0, 0.5, 6)
    gain = freq * 2.0

    fine = interpolate_gain(freq, gain, 64)

    assert fine.size == 33
    np.testing.assert_allclose(fine, np.fft.rfftfreq(64) * 2.0)


def test_fit_power_law_rejects_non_positive_spectrum():
    """fit_power_law on an all-zero spectrum raises ValueError."""
    freq = np.fft.rfftfreq(2 * MAX_LAG, d=1.0)

    with pytest.raises(ValueError, match="positive power"):
        fit_power_law(freq, np.zeros_like(freq))


def test_taper_high_frequencies():
    """taper_high_frequencies zeros the gain above 1/min_period and leaves it unchanged below half that frequency."""
    freq = np.array([0.0, 0.002, 0.0075, 0.02, 0.5])
    gain = np.ones_like(freq)

    tapered = taper_high_frequencies(freq, gain, min_period=100.0)

    np.testing.assert_allclose(tapered[[0, 1]], 1.0)
    np.testing.assert_allclose(tapered[[3, 4]], 0.0)
    assert 0.0 < tapered[2] < 1.0

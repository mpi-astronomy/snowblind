import numpy as np

from snowblind import readout
from snowblind.metrics import (
    aperture_scatter,
    channel_psd,
    d0_chisq,
    injected_flux_bias,
    recovered_fraction,
    robust_sigma,
    row_banding_power,
)

NY, NX = 256, 256


def test_robust_sigma_gaussian():
    """robust_sigma of 100000 unit Gaussian values is 1 within 2 percent and ignores NaN."""
    rng = np.random.default_rng(1)
    values = rng.normal(size=100000)
    values[:10] = np.nan

    assert abs(robust_sigma(values) - 1.0) < 0.02


def test_channel_psd_white_noise_is_flat_at_variance():
    """channel_psd of white noise with sigma 2 is flat at 4 in every channel within 10 percent."""
    rng = np.random.default_rng(2)
    diffs = rng.normal(scale=2.0, size=(3, NY, NX))
    mask = np.ones((NY, NX), dtype=bool)

    freq, spectra = channel_psd(diffs, mask, -1, 2, max_lag=512)

    assert spectra.shape[0] == 4
    for spectrum in spectra:
        assert abs(spectrum.mean() / 4.0 - 1.0) < 0.1


def test_channel_psd_sees_row_stripes_at_low_frequency():
    """channel_psd of random row stripes in channel 0 puts far more power below 1/524 cycles per clock than above."""
    rng = np.random.default_rng(3)
    det = np.zeros((1, NY, NX))
    det[0, :, : NX // 4] = rng.normal(scale=5.0, size=(NY, 1))
    diffs = np.ascontiguousarray(readout.to_science_frame(det, -1, 2))
    mask = np.ones((NY, NX), dtype=bool)

    freq, spectra = channel_psd(diffs, mask, -1, 2, max_lag=1024)

    low = spectra[0][(freq > 0) & (freq < 1.0 / (NX // 4 + readout.LINE_OVERHEAD))].mean()
    high = spectra[0][freq > 0.1].mean()
    assert low > 20 * high


def test_row_banding_power_recovers_sinusoid():
    """row_banding_power of row means following a 16-row sinusoid peaks at 1/16 cycles per row in all channels."""
    rows = np.arange(NY)
    det = np.repeat(np.sin(2 * np.pi * rows / 16.0)[:, np.newaxis], NX, axis=1)
    image = np.ascontiguousarray(readout.to_science_frame(det, 1, -2))
    mask = np.ones((NY, NX), dtype=bool)

    freq, power, row_means = row_banding_power(image, mask, 1, -2)

    assert row_means.shape == (4, NY)
    for channel_power in power:
        assert abs(freq[np.argmax(channel_power)] - 1.0 / 16.0) < 1e-6


def test_row_banding_power_mean_is_variance():
    """row_banding_power of unit white noise per row averages to 1 / ncols_per_channel over bins."""
    rng = np.random.default_rng(4)
    image = rng.normal(size=(NY, NX))
    mask = np.ones((NY, NX), dtype=bool)

    _freq, power, _row_means = row_banding_power(image, mask, -1, 2)

    expected = 1.0 / (NX // 4)
    assert abs(power.mean() / expected - 1.0) < 0.15


def test_aperture_scatter_white_noise_follows_prediction():
    """aperture_scatter on unit white noise matches sigma / size within 20 percent for sizes 1, 3, 9."""
    rng = np.random.default_rng(5)
    image = rng.normal(size=(1024, 1024))
    mask = np.ones(image.shape, dtype=bool)

    result = aperture_scatter(image, mask, [1, 3, 9], pixel_sigma=1.0)

    ratio = result["scatter"] / result["white_prediction"]
    np.testing.assert_allclose(ratio, 1.0, atol=0.2)


def test_aperture_scatter_skips_masked_apertures():
    """aperture_scatter counts no 5-pixel apertures when every fifth column is masked (80 percent usable)."""
    rng = np.random.default_rng(6)
    image = rng.normal(size=(NY, NX))
    mask = np.ones(image.shape, dtype=bool)
    mask[:, ::5] = False

    result = aperture_scatter(image, mask, [5])

    assert result["n_apertures"][0] == 0
    assert np.isnan(result["scatter"][0])


def test_d0_chisq_counts_only_used_finite_pixels():
    """d0_chisq over residuals [1, 2, NaN, 3] with the last one unused returns mean of 1 and 4 over 2 pixels."""
    resid = np.array([[1.0, 2.0, np.nan, 3.0]])
    used = np.array([[True, True, True, False]])

    result = d0_chisq(resid, used)

    assert result["chisq"] == 2.5
    assert result["n"] == 2
    assert result["used_fraction"] == 0.75


def test_recovered_fraction_counts_newly_finite_pixels():
    """recovered_fraction with rate [1, 2, NaN, 4] against reference [1, NaN, NaN, 4] reports one recovered, none lost."""
    rate = np.array([1.0, 2.0, np.nan, 4.0])
    reference = np.array([1.0, np.nan, np.nan, 4.0])

    result = recovered_fraction(rate, reference)

    assert result["n_finite"] == 3
    assert result["n_finite_reference"] == 2
    assert result["n_recovered"] == 1
    assert result["n_lost"] == 0


def gaussian_sources(shape, x, y, flux, sigma=1.3):
    """Image of Gaussian sources with the given total fluxes."""
    yy, xx = np.mgrid[0: shape[0], 0: shape[1]]
    image = np.zeros(shape)
    for xi, yi, fi in zip(x, y, flux):
        image += fi * np.exp(-((xx - xi) ** 2 + (yy - yi) ** 2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    return image


def test_injected_flux_bias_zero_for_identical_images():
    """injected_flux_bias of an image against itself is zero for every source."""
    rng = np.random.default_rng(1)
    x, y, flux = np.array([30.0, 90.0]), np.array([40.0, 100.0]), np.array([100.0, 200.0])
    image = gaussian_sources((128, 128), x, y, flux) + rng.normal(size=(128, 128))

    result = injected_flux_bias(image, image, x, y, flux, np.ones((128, 128), dtype=bool))

    np.testing.assert_allclose(result["relative_bias"], 0.0)
    assert result["n_measured"] == 2


def test_injected_flux_bias_ignores_constant_offset():
    """injected_flux_bias is zero when the cleaned image differs by a constant, which the annulus removes."""
    x, y, flux = np.array([30.0, 90.0]), np.array([40.0, 100.0]), np.array([100.0, 200.0])
    image = gaussian_sources((128, 128), x, y, flux)

    result = injected_flux_bias(image - 5.0, image, x, y, flux, np.ones((128, 128), dtype=bool))

    np.testing.assert_allclose(result["relative_bias"], 0.0, atol=1e-9)


def test_injected_flux_bias_measures_one_percent_loss():
    """injected_flux_bias reports about -0.01 when the cleaned image holds 99 percent of the source flux."""
    x, y, flux = np.array([30.0, 90.0]), np.array([40.0, 100.0]), np.array([100.0, 200.0])
    sources = gaussian_sources((128, 128), x, y, flux)
    sky = 3.0

    result = injected_flux_bias(sky + 0.99 * sources, sky + sources, x, y, flux, np.ones((128, 128), dtype=bool))

    np.testing.assert_allclose(result["relative_bias"], -0.01 * 0.93, atol=0.002)


def test_injected_flux_bias_density_and_neighbors():
    """injected_flux_bias tags a source in a fully masked box with density 1 and reports the neighbour distance."""
    x, y, flux = np.array([30.0, 90.0]), np.array([40.0, 40.0]), np.array([100.0, 200.0])
    image = gaussian_sources((128, 128), x, y, flux)
    mask = np.ones((128, 128), dtype=bool)
    mask[20:61, 10:51] = False

    result = injected_flux_bias(image, image, x, y, flux, mask)

    assert result["density"][0] == 1.0
    assert result["density"][1] == 0.0
    np.testing.assert_allclose(result["neighbor_distance"], 60.0)


def test_injected_flux_bias_by_flux_bins_cover_all_sources():
    """injected_flux_bias returns by_flux bins whose counts sum to the number of sources."""
    x, y, flux = np.array([30.0, 90.0, 60.0]), np.array([40.0, 100.0, 70.0]), np.array([10.0, 100.0, 1000.0])
    image = gaussian_sources((128, 128), x, y, flux)

    result = injected_flux_bias(image, image, x, y, flux, np.ones((128, 128), dtype=bool))

    assert sum(row[2] for row in result["by_flux"]) == 3

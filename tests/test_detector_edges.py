import numpy as np

from snowblind import readout
from snowblind.detector_edges import (
    coupling_profiles,
    coupling_sums,
    edge_columns,
    edge_profiles,
    edge_reach,
    edge_bin_gains,
    edge_stream_gain,
    edge_stream_sums,
    filter_edge_series,
    far_half,
    weighted_edge_series,
)
from snowblind.stream_clean import clean_ramp, estimate_channel_gains, remove_edge_noise

NROWS, NCOLS = 256, 256  # four channels of 64 columns: bins of 16 give two near and two far bins per channel


def slow_row_signal(rng, nrows, amplitude=5.0):
    """A smooth signal along the rows: a random walk, mean removed."""
    signal = np.cumsum(rng.normal(size=nrows))
    return (amplitude * (signal - signal.mean()) / signal.std()).astype(np.float64)


def coupled_image(rng, edge_gain=0.5, edge_columns_count=16, noise=0.3):
    """Detector-frame image: each channel carries its own row signal, the first columns of channel 0 at ``edge_gain``."""
    image = np.zeros((NROWS, NCOLS))
    for k, columns in enumerate(readout.channel_slices(NCOLS)):
        image[:, columns] = slow_row_signal(rng, NROWS)[:, np.newaxis]
    image[:, :edge_columns_count] *= edge_gain
    return image + rng.normal(scale=noise, size=image.shape)


def test_far_half_is_the_high_side_for_the_first_channels():
    """far_half(0, 4) is bins 2 to 4 and far_half(3, 4) bins 0 to 2."""
    assert far_half(0, 4) == slice(2, 4)
    assert far_half(3, 4) == slice(0, 2)


def test_coupling_profile_measures_the_edge_gain_of_channel_0():
    """coupling_profiles on an image whose first 16 columns carry half the channel's row signal gives 0.5 there."""
    rng = np.random.default_rng(1)
    image = coupled_image(rng)
    xy, xx = coupling_sums(image, np.ones(image.shape, dtype=bool))

    profiles = coupling_profiles(xy, xx, readout.channel_width(NCOLS))

    assert abs(profiles[0][0] - 0.5) < 0.1
    assert abs(profiles[0][-1] - 1.0) < 1e-6


def test_coupling_profile_is_none_for_a_channel_that_couples_alike():
    """coupling_profiles on the same image returns None for channels 1 to 3, whose columns all carry the signal alike."""
    rng = np.random.default_rng(1)
    image = coupled_image(rng)
    xy, xx = coupling_sums(image, np.ones(image.shape, dtype=bool))

    profiles = coupling_profiles(xy, xx, readout.channel_width(NCOLS))

    assert profiles[1] is None and profiles[2] is None and profiles[3] is None


def test_coupling_sums_add_over_images():
    """coupling_profiles on the sums of two images equals the profile of one image when both are the same image."""
    rng = np.random.default_rng(2)
    image = coupled_image(rng)
    mask = np.ones(image.shape, dtype=bool)
    xy, xx = coupling_sums(image, mask)

    one = coupling_profiles(xy, xx, readout.channel_width(NCOLS))[0]
    two = coupling_profiles(2 * xy, 2 * xx, readout.channel_width(NCOLS))[0]

    assert np.allclose(one, two)


def test_edge_reach_shrinks_to_half_a_narrow_frame():
    """edge_reach(256) with the default border 4 and bins of 16 returns 112; edge_reach(2048) returns 256."""
    assert edge_reach(256) == 112
    assert edge_reach(2048) == 256


def test_edge_columns_are_mirrored():
    """edge_columns(2048, 256) returns columns 4 to 260 and 1788 to 2044."""
    assert edge_columns(2048, 256) == [slice(4, 260), slice(1788, 2044)]


def test_edge_series_is_the_window_mean_of_the_residual():
    """edge_stream_sums on an image with a row signal over the low edge returns that signal as the low edge's series."""
    rng = np.random.default_rng(3)
    signal = slow_row_signal(rng, NROWS)
    reach = edge_reach(NCOLS)
    image = np.zeros((NROWS, NCOLS))
    image[:, 4:4 + reach] = signal[:, np.newaxis]

    series = edge_stream_sums(image, np.ones(image.shape, dtype=bool))[0]

    assert np.allclose(series[0], signal)
    assert np.allclose(series[1], 0.0)


def test_edge_profile_follows_where_the_edge_noise_is_carried():
    """edge_profiles from an image whose edge noise covers the 32 columns next to the border gives 1 there and 0 beyond."""
    rng = np.random.default_rng(4)
    signal = slow_row_signal(rng, NROWS)
    image = rng.normal(scale=0.1, size=(NROWS, NCOLS))
    image[:, 4:36] += signal[:, np.newaxis]
    _series, xy, xx, _means, _counts = edge_stream_sums(image, np.ones(image.shape, dtype=bool), window=32)

    profiles = edge_profiles(xy, xx)

    assert abs(profiles[0][0] - 1.0) < 0.1
    assert abs(profiles[0][-1]) < 0.1


def test_edge_stream_gain_is_none_for_white_noise():
    """edge_stream_gain on 8 series of white noise returns None."""
    rng = np.random.default_rng(5)
    series = rng.normal(size=(8, NROWS))

    assert edge_stream_gain(series, 2 * NROWS) is None


def test_edge_stream_gain_passes_a_slow_signal():
    """edge_stream_gain on 8 series of a random walk plus small white noise is above 0.8 at a period of 32 rows."""
    rng = np.random.default_rng(6)
    series = np.array([slow_row_signal(rng, NROWS) for _ in range(8)]) + rng.normal(scale=0.3, size=(8, NROWS))
    nfft = 2 * NROWS

    gain = edge_stream_gain(series, nfft)

    freq = np.fft.rfftfreq(nfft)
    assert gain[np.argmin(np.abs(freq - 1 / 32))] > 0.8


def test_filter_edge_series_with_unit_gain_returns_the_mean_removed_series():
    """filter_edge_series with a gain of 1 everywhere but zero frequency returns the series minus its mean."""
    rng = np.random.default_rng(7)
    series = rng.normal(size=(2, NROWS)) + 3.0
    nfft = 2 * NROWS
    gain = np.ones(nfft // 2 + 1)
    gain[0] = 0.0

    filtered = filter_edge_series(series, gain, nfft)

    assert np.allclose(filtered, series - series.mean(axis=1, keepdims=True), atol=1e-5)


def test_remove_edge_noise_takes_the_edge_noise_out_of_the_groups():
    """remove_edge_noise on groups whose low-edge columns carry a slow per-difference noise lowers it by over 70 percent."""
    rng = np.random.default_rng(8)
    ndiff = 8
    reach = edge_reach(NCOLS)
    low = edge_columns(NCOLS, reach)[0]
    differences = rng.normal(scale=0.1, size=(ndiff, NROWS, NCOLS))
    for j in range(ndiff):
        differences[j, :, low] += slow_row_signal(rng, NROWS)[:, np.newaxis]
    groups = np.concatenate([np.zeros((1, NROWS, NCOLS)), np.cumsum(differences, axis=0)]).astype(np.float32)
    mask = np.ones((NROWS, NCOLS), dtype=bool)
    bins_by_group, xy, xx = {}, 0.0, 0.0
    for j in range(ndiff):
        _series, dxy, dxx, means, counts = edge_stream_sums(differences[j], mask)
        bins_by_group[j + 1] = (means, counts)
        xy, xx = xy + dxy, xx + dxx
    before = np.std(groups[-1][:, low].mean(axis=1))

    remove_edge_noise(groups, bins_by_group, xy, xx, fastaxis=1, slowaxis=2)

    after = np.std(groups[-1][:, low].mean(axis=1))
    assert after < 0.3 * before


def test_remove_edge_noise_leaves_the_interior_alone():
    """remove_edge_noise on the same groups leaves the columns between the two edge reaches (116 to 140) unchanged."""
    rng = np.random.default_rng(8)
    ndiff = 8
    reach = edge_reach(NCOLS)
    low = edge_columns(NCOLS, reach)[0]
    differences = rng.normal(scale=0.1, size=(ndiff, NROWS, NCOLS))
    for j in range(ndiff):
        differences[j, :, low] += slow_row_signal(rng, NROWS)[:, np.newaxis]
    groups = np.concatenate([np.zeros((1, NROWS, NCOLS)), np.cumsum(differences, axis=0)]).astype(np.float32)
    mask = np.ones((NROWS, NCOLS), dtype=bool)
    bins_by_group, xy, xx = {}, 0.0, 0.0
    for j in range(ndiff):
        _series, dxy, dxx, means, counts = edge_stream_sums(differences[j], mask)
        bins_by_group[j + 1] = (means, counts)
        xy, xx = xy + dxy, xx + dxx
    interior_before = groups[:, :, 116:140].copy()

    remove_edge_noise(groups, bins_by_group, xy, xx, fastaxis=1, slowaxis=2)

    assert np.array_equal(groups[:, :, 116:140], interior_before)


def test_remove_edge_noise_returns_none_without_edge_noise():
    """remove_edge_noise on groups of white noise returns None and leaves the groups unchanged."""
    rng = np.random.default_rng(9)
    ndiff = 8
    differences = rng.normal(scale=0.1, size=(ndiff, NROWS, NCOLS))
    groups = np.concatenate([np.zeros((1, NROWS, NCOLS)), np.cumsum(differences, axis=0)]).astype(np.float32)
    mask = np.ones((NROWS, NCOLS), dtype=bool)
    bins_by_group, xy, xx = {}, 0.0, 0.0
    for j in range(ndiff):
        _series, dxy, dxx, means, counts = edge_stream_sums(differences[j], mask)
        bins_by_group[j + 1] = (means, counts)
        xy, xx = xy + dxy, xx + dxx
    before = groups.copy()

    result = remove_edge_noise(groups, bins_by_group, xy, xx, fastaxis=1, slowaxis=2)

    assert result is None
    assert np.array_equal(groups, before)


def striped_noisy_ramp(rng, ngroups=4):
    """Ramp of 256 by 256 pixels, fastaxis 1 and slowaxis 2, whose differences carry slow 1/f-like stripes along the
    readout rows of each channel plus white noise."""
    data = np.zeros((1, ngroups, NROWS, NCOLS), dtype=np.float32)
    for j in range(1, ngroups):
        difference = rng.normal(scale=2.0, size=(NROWS, NCOLS))
        for columns in readout.channel_slices(NCOLS):
            difference[:, columns] += slow_row_signal(rng, NROWS)[:, np.newaxis]
        data[0, j] = data[0, j - 1] + difference
    return data


def test_clean_ramp_with_edge_coupling_changes_nothing_without_edge_effects():
    """clean_ramp with edge_coupling=True on a ramp whose stripes are carried alike everywhere equals clean_ramp
    without it."""
    rng = np.random.default_rng(10)
    data = striped_noisy_ramp(rng)
    groupdq = np.zeros(data.shape, dtype=np.uint32)
    mask = np.ones((NROWS, NCOLS), dtype=bool)

    plain = clean_ramp(data, groupdq, mask, 1, 2, fit_method="wiener", edge_coupling=False)
    coupled = clean_ramp(data, groupdq, mask, 1, 2, fit_method="wiener", edge_coupling=True)

    assert np.allclose(plain, coupled, atol=1e-4)


def test_estimate_channel_gains_measures_the_coupling_of_half_coupled_edge_columns():
    """estimate_channel_gains with edge_coupling=True on a ramp whose columns 4 to 36 carry half of channel 0's
    stripes gives channel 0 a coupling near 0.5 there and the other channels None."""
    rng = np.random.default_rng(11)
    data = striped_noisy_ramp(rng, ngroups=6)
    data[:, :, :, 4:36] *= 0.5
    groupdq = np.zeros(data.shape, dtype=np.uint32)
    mask = np.ones((NROWS, NCOLS), dtype=bool)

    spectra = estimate_channel_gains(data, groupdq, mask, 1, 2, edge_coupling=True)

    coupling = spectra[0][0]["coupling"]
    assert abs(np.mean(coupling[8:32]) - 0.5) < 0.15
    assert spectra[0][1]["coupling"] is None
    assert spectra[0][2]["coupling"] is None
    assert spectra[0][3]["coupling"] is None


def test_weighted_edge_series_is_the_gain_weighted_mean_of_the_bins():
    """weighted_edge_series with gains 1, 0.5 and 0 on three bins of equal counts returns (m0 + 0.5 m1) / 1.25."""
    means = np.array([[[2.0, 4.0], [1.0, 2.0], [9.0, 9.0]]])
    counts = np.ones((1, 3, 2))
    gains = np.array([[1.0, 0.5, 0.0]])

    series = weighted_edge_series(means, counts, gains)

    assert np.allclose(series[0], [2.5 / 1.25, 5.0 / 1.25])


def test_weighted_edge_series_is_nan_without_pixels():
    """weighted_edge_series with zero counts in a row returns NaN for that row."""
    means = np.zeros((1, 2, 3))
    counts = np.array([[[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]])
    gains = np.ones((1, 2))

    series = weighted_edge_series(means, counts, gains)

    assert np.isnan(series[0, 1]) and np.isfinite(series[0, [0, 2]]).all()


def test_edge_bin_gains_is_nan_without_data():
    """edge_bin_gains with xx of 0 in a bin returns NaN there and the ratio elsewhere."""
    xy = np.array([[2.0, 3.0]])
    xx = np.array([[4.0, 0.0]])

    gains = edge_bin_gains(xy, xx)

    assert gains[0, 0] == 0.5 and np.isnan(gains[0, 1])

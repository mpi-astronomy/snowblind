import numpy as np
from stdatamodels.jwst import datamodels

from snowblind import StreamCleanStep, readout
from snowblind.noise_spectrum import notch_period, synthetic_stream
from snowblind.stream_clean import (
    channel_gain,
    clean_ramp,
    estimate_channel_gains,
)

NY, NX = 256, 256
NGROUPS = 6
RATE = 10.0
WHITE = 1.0
MAX_LAG = 2048


def flicker_difference(rng, alpha=1.0, amplitude=0.22, fastaxis=-1, slowaxis=2):
    """Science-frame image of per-channel 1/f noise laid out in clock order, colored rms about 2."""
    width = readout.channel_width(NX)
    det = np.zeros((NY, NX), dtype=np.float32)
    for k, columns in enumerate(readout.channel_slices(NX)):
        direction = readout.channel_direction(k, 1)
        stream = synthetic_stream(readout.stream_length(NY, width), alpha, amplitude, 0.0, rng)
        det[:, columns] = readout.from_stream(stream, NY, width, direction)
    return np.ascontiguousarray(readout.to_science_frame(det, fastaxis, slowaxis))


def source_mask(rng, nsources=40):
    """Boolean image True on 5x5 source footprints; about 4 percent of pixels."""
    sources = np.zeros((NY, NX), dtype=bool)
    for _ in range(nsources):
        y, x = rng.integers(3, NY - 3), rng.integers(3, NX - 3)
        sources[y - 2: y + 3, x - 2: x + 3] = True
    return sources


def noisy_ramp(rng, with_sources=False):
    """Ramp whose differences are RATE plus flicker plus white noise; returns data, flicker and white cubes, sources."""
    data = np.zeros((1, NGROUPS, NY, NX), dtype=np.float32)
    flicker = np.zeros((NGROUPS, NY, NX), dtype=np.float32)
    white = np.zeros((NGROUPS, NY, NX), dtype=np.float32)
    sources = source_mask(rng) if with_sources else np.zeros((NY, NX), dtype=bool)
    for j in range(NGROUPS):
        flicker[j] = flicker_difference(rng)
        white[j] = rng.normal(scale=WHITE, size=(NY, NX))
        diff = RATE + flicker[j] + white[j] + 500.0 * sources
        data[0, j] = diff if j == 0 else data[0, j - 1] + diff
    return data, flicker, white, sources


def residual_rms(cleaned, data, flicker, mask):
    """RMS over unmasked pixels of (cleaned difference minus truth without flicker), groups 1 and up."""
    total = []
    for j in range(1, NGROUPS):
        cleaned_diff = cleaned[0, j] - cleaned[0, j - 1]
        truth_diff = (data[0, j] - data[0, j - 1]) - flicker[j]
        total.append(((cleaned_diff - truth_diff)[mask]) ** 2)
    return float(np.sqrt(np.mean(np.concatenate([t.ravel() for t in total]))))


def clean_with(method, data, mask, **kwargs):
    groupdq = np.zeros(data.shape, dtype=np.uint8)
    return clean_ramp(
        data, groupdq, mask, -1, 2, fit_method=method, background_method="median",
        max_lag=MAX_LAG, **kwargs
    )


def test_channel_gain_recovers_white_floor_and_slope():
    """estimate_channel_gains on flicker (alpha 1) plus unit white noise returns white near 1 and alpha within 0.15."""
    rng = np.random.default_rng(1)
    data, _flicker, _white, _sources = noisy_ramp(rng)
    groupdq = np.zeros(data.shape, dtype=np.uint8)
    mask = np.ones((NY, NX), dtype=bool)

    spectra = estimate_channel_gains(data, groupdq, mask, -1, 2, background_method="median", max_lag=MAX_LAG)

    for channel in spectra[0]:
        assert abs(channel["white"] - 1.0) < 0.15
        assert abs(channel["alpha"] - 1.0) < 0.15
        assert channel["gain"][0] == 1.0


def test_wiener_beats_row_median():
    """clean_ramp with wiener leaves a smaller residual than row_median on flicker plus white noise."""
    rng = np.random.default_rng(2)
    data, flicker, _white, _sources = noisy_ramp(rng)
    mask = np.ones((NY, NX), dtype=bool)

    wiener = residual_rms(clean_with("wiener", data, mask), data, flicker, mask)
    row_median = residual_rms(clean_with("row_median", data, mask), data, flicker, mask)

    assert wiener < row_median


def test_wiener_residual_below_flicker_rms():
    """clean_ramp with wiener removes most of the flicker: residual rms below half the injected flicker rms."""
    rng = np.random.default_rng(3)
    data, flicker, _white, _sources = noisy_ramp(rng)
    mask = np.ones((NY, NX), dtype=bool)

    residual = residual_rms(clean_with("wiener", data, mask), data, flicker, mask)

    assert residual < 0.5 * float(np.sqrt(np.mean(flicker[1:] ** 2)))


def test_wiener_interpolates_under_sources():
    """clean_ramp with wiener changes masked source pixels by the interpolated flicker, within 1.5 white sigma rms."""
    rng = np.random.default_rng(4)
    data, flicker, _white, sources = noisy_ramp(rng, with_sources=True)
    mask = ~sources

    cleaned = clean_with("wiener", data, mask)

    errors = []
    for j in range(1, NGROUPS):
        removed = (data[0, j] - data[0, j - 1]) - (cleaned[0, j] - cleaned[0, j - 1])
        errors.append((removed - flicker[j])[sources])
    assert float(np.sqrt(np.mean(np.concatenate(errors) ** 2))) < 1.5 * WHITE


def test_wiener_source_flux_bias_small():
    """clean_ramp with wiener biases the summed flux of 25-pixel sources at 500 DN per pixel by less than 0.5 percent."""
    rng = np.random.default_rng(5)
    data, _flicker, _white, sources = noisy_ramp(rng, with_sources=True)
    mask = ~sources

    cleaned = clean_with("wiener", data, mask)

    diff = cleaned[0, 3] - cleaned[0, 2]
    truth = data[0, 3] - data[0, 2]
    bias = (diff[sources] - truth[sources]).mean() / 500.0
    assert abs(bias) < 0.005


def test_clean_ramp_returns_spectra_for_wiener():
    """clean_ramp with return_spectra=True and wiener returns one list of four channel dicts per integration."""
    rng = np.random.default_rng(6)
    data, _flicker, _white, _sources = noisy_ramp(rng)
    mask = np.ones((NY, NX), dtype=bool)

    _cleaned, spectra = clean_with("wiener", data, mask, return_spectra=True)

    assert len(spectra) == 1
    assert len(spectra[0]) == 4
    assert set(spectra[0][0]) >= {"freq", "spectrum", "smooth", "gain", "alpha", "white", "amplitude"}


def test_clean_ramp_returns_no_spectra_for_row_median():
    """clean_ramp with return_spectra=True and row_median returns None for the spectra."""
    rng = np.random.default_rng(7)
    data, _flicker, _white, _sources = noisy_ramp(rng)
    mask = np.ones((NY, NX), dtype=bool)

    _cleaned, spectra = clean_with("row_median", data, mask, return_spectra=True)

    assert spectra is None


def test_channel_gain_fixed_alpha():
    """channel_gain with fixed_alpha keeps that slope."""
    freq = np.fft.rfftfreq(4096)
    spectrum = 0.01 / np.maximum(freq, freq[1]) + 1.0

    result = channel_gain(freq, spectrum, fixed_alpha=1.2)

    assert result["alpha"] == 1.2


def make_ramp_model(data):
    model = datamodels.RampModel(data)
    model.meta.instrument.name = "NIRCAM"
    model.meta.instrument.detector = "NRCA1"
    model.meta.exposure.type = "NRC_IMAGE"
    model.meta.exposure.nframes = 1
    model.meta.exposure.groupgap = 0
    model.meta.subarray.name = "FULL"
    model.meta.subarray.fastaxis = -1
    model.meta.subarray.slowaxis = 2
    model.meta.filename = "jw00000000000_00000_00000_nrca1_jump.fits"
    return model


def test_step_wiener_with_user_mask_saves_spectra(tmp_path, tmp_cwd):
    """StreamCleanStep.call with wiener and save_spectra writes a spectra ASDF next to the output and completes."""
    rng = np.random.default_rng(8)
    data, _flicker, _white, _sources = noisy_ramp(rng)
    model = make_ramp_model(data)
    mask_model = datamodels.ImageModel(np.ones((NY, NX), dtype=np.float32))
    mask_path = tmp_path / "mask.fits"
    mask_model.save(mask_path)

    result = StreamCleanStep.call(
        model, apply_flat_field=False, fit_method="wiener", user_mask=str(mask_path), background_method="median",
        max_lag=MAX_LAG, save_spectra=True,
    )

    assert result.meta.cal_step.stream_clean == "COMPLETE"
    assert len(list(tmp_cwd.glob("*_spectra.asdf"))) == 1


def test_channel_gain_min_period_zeroes_high_frequencies():
    """channel_gain with min_period 100 gives zero gain at periods under 100 clocks and gain 1 at zero frequency."""
    freq = np.fft.rfftfreq(4096)
    spectrum = 0.05 / np.maximum(freq, freq[1]) + 1.0

    result = channel_gain(freq, spectrum, min_period=100)

    assert result["gain"][0] == 1.0
    np.testing.assert_array_equal(result["gain"][freq > 0.0101], 0.0)
    assert result["gain"][(freq > 0.001) & (freq < 0.004)].min() > 0.5


def test_notch_period_is_zero_at_the_period_and_one_far_from_it():
    """notch_period on a unit gain with period 524 returns 0 at 1/524 cycles per clock and 1 at half that frequency."""
    freq = np.linspace(0.0, 0.5, 100001)
    gain = np.ones_like(freq)

    out = notch_period(freq, gain, 524.0)

    assert out[np.argmin(np.abs(freq - 1.0 / 524.0))] < 0.01
    assert abs(out[np.argmin(np.abs(freq - 0.5 / 524.0))] - 1.0) < 1e-6


def test_channel_gain_with_notch_period_cuts_the_gain_at_that_period():
    """channel_gain on a 1/f spectrum with notch_period=524 gives a gain below 0.05 at 1/524 cycles per clock."""
    freq = np.fft.rfftfreq(65536)
    spectrum = 1.0 + 1000.0 / np.maximum(freq, freq[1])

    with_notch = channel_gain(freq, spectrum, notch_period=524.0)
    without = channel_gain(freq, spectrum)

    index = np.argmin(np.abs(freq - 1.0 / 524.0))
    assert with_notch["gain"][index] < 0.05
    assert without["gain"][index] > 0.5

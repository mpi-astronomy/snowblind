import numpy as np

from stdatamodels.jwst import datamodels

from snowblind import readout
from snowblind.reset_calib import (
    pedestal_cube,
    read_calibration,
    remove_channel_offsets,
    sigma_b_from_pedestals,
    write_calibration,
)

NY, NX = 32, 64


def test_remove_channel_offsets_subtracts_per_integration_channel_median():
    """remove_channel_offsets on planes that are constant per channel and integration returns zeros."""
    pedestals = np.zeros((3, NY, NX), dtype=np.float32)
    det = readout.to_detector_frame(pedestals, -1, 2)
    for i in range(3):
        for k, columns in enumerate(readout.channel_slices(NX)):
            det[i, :, columns] = 10.0 * i + k

    corrected = remove_channel_offsets(pedestals, -1, 2)

    np.testing.assert_allclose(corrected, 0.0, atol=1e-6)


def test_remove_channel_offsets_keeps_pixel_scatter():
    """remove_channel_offsets leaves a zero-median pixel pattern unchanged apart from the median."""
    rng = np.random.default_rng(1)
    pattern = rng.normal(size=(1, NY, NX)).astype(np.float32)
    pattern -= np.median(pattern)

    corrected = remove_channel_offsets(pattern + 7.0, -1, 2)

    np.testing.assert_allclose(corrected, pattern, atol=0.2)


def test_remove_channel_offsets_ignores_nan():
    """remove_channel_offsets treats NaN pixels as missing and keeps them NaN."""
    pedestals = np.full((1, NY, NX), 5.0, dtype=np.float32)
    pedestals[0, 3, 3] = np.nan

    corrected = remove_channel_offsets(pedestals, -1, 2)

    assert np.isnan(corrected[0, 3, 3])
    np.testing.assert_allclose(corrected[0, 0], 0.0, atol=1e-6)


def test_sigma_b_recovers_known_scatter():
    """sigma_b_from_pedestals on 200 Gaussian planes with scatter 4 returns about 4 per pixel."""
    rng = np.random.default_rng(2)
    pedestals = rng.normal(scale=4.0, size=(200, NY, NX)).astype(np.float32)

    sigma_b, _mean, count, _scatter = sigma_b_from_pedestals(pedestals)

    assert abs(np.median(sigma_b) - 4.0) < 0.2
    assert count.min() == 200


def test_sigma_b_mean_is_separate_from_scatter():
    """sigma_b_from_pedestals reports a 3 DN offset in the mean and not in sigma_b."""
    rng = np.random.default_rng(3)
    pedestals = (3.0 + rng.normal(scale=1.0, size=(200, NY, NX))).astype(np.float32)

    sigma_b, mean, _count, _scatter = sigma_b_from_pedestals(pedestals)

    assert abs(np.median(mean) - 3.0) < 0.1
    assert abs(np.median(sigma_b) - 1.0) < 0.1


def test_sigma_b_nan_where_too_few_values():
    """sigma_b_from_pedestals returns NaN and the finite count for a pixel with only two values."""
    pedestals = np.ones((5, NY, NX), dtype=np.float32)
    pedestals[2:, 1, 1] = np.nan

    sigma_b, _mean, count, _scatter = sigma_b_from_pedestals(pedestals)

    assert np.isnan(sigma_b[1, 1])
    assert count[1, 1] == 2


def test_sigma_b_clips_outliers():
    """sigma_b_from_pedestals ignores one 1000 DN outlier among 50 unit-scatter values."""
    rng = np.random.default_rng(4)
    pedestals = rng.normal(size=(50, NY, NX)).astype(np.float32)
    pedestals[7, 2, 2] = 1000.0

    sigma_b, _mean, _count, _scatter = sigma_b_from_pedestals(pedestals)

    assert sigma_b[2, 2] < 2.0


def test_sigma_b_removes_first_group_read_noise():
    """sigma_b_from_pedestals with read_noise 3 on planes of scatter 5 returns sigma_b 4 and scatter 5."""
    rng = np.random.default_rng(5)
    pedestals = rng.normal(scale=5.0, size=(400, NY, NX)).astype(np.float32)
    read_noise = np.full((NY, NX), 3.0, dtype=np.float32)

    sigma_b, _mean, _count, scatter = sigma_b_from_pedestals(pedestals, read_noise=read_noise)

    assert abs(np.median(scatter) - 5.0) < 0.2
    assert abs(np.median(sigma_b) - 4.0) < 0.25


def test_sigma_b_read_noise_divided_by_nframes():
    """sigma_b_from_pedestals with read_noise 4 and nframes 4 on planes of scatter 5 returns about sqrt(21)."""
    rng = np.random.default_rng(6)
    pedestals = rng.normal(scale=5.0, size=(400, NY, NX)).astype(np.float32)
    read_noise = np.full((NY, NX), 4.0, dtype=np.float32)

    sigma_b, _mean, _count, _scatter = sigma_b_from_pedestals(pedestals, read_noise=read_noise, nframes=4)

    assert abs(np.median(sigma_b) - np.sqrt(21.0)) < 0.25


def test_sigma_b_clipped_at_zero_when_read_noise_dominates():
    """sigma_b_from_pedestals returns 0, not NaN, where the read noise exceeds the pedestal scatter."""
    pedestals = np.ones((5, NY, NX), dtype=np.float32) + np.arange(5, dtype=np.float32)[:, None, None] * 0.01
    read_noise = np.full((NY, NX), 10.0, dtype=np.float32)

    sigma_b, _mean, _count, _scatter = sigma_b_from_pedestals(pedestals, read_noise=read_noise)

    np.testing.assert_array_equal(sigma_b, 0.0)


def test_calibration_round_trip(tmp_path):
    """write_calibration then read_calibration returns the same arrays, median, scale and detector."""
    sigma_b = np.full((NY, NX), 2.5, dtype=np.float32)
    mean = np.zeros((NY, NX), dtype=np.float32)
    count = np.full((NY, NX), 6, dtype=np.int32)
    path = tmp_path / "nrca1_reset_calib.asdf"

    write_calibration(path, sigma_b, mean, count, "NRCA1", ["a_jump.fits", "b_jump.fits"], t1_scale=0.98)
    tree = read_calibration(path)

    np.testing.assert_array_equal(tree["sigma_b"], sigma_b)
    np.testing.assert_array_equal(tree["pedestal_scatter"], sigma_b)
    assert tree["sigma_b_median"] == 2.5
    assert tree["t1_scale"] == 0.98
    assert tree["detector"] == "NRCA1"
    assert tree["provenance"]["input_files"] == ["a_jump.fits", "b_jump.fits"]


def test_read_calibration_tolerates_older_file(tmp_path):
    """read_calibration on an ASDF without pedestal_scatter still returns sigma_b and the median."""
    import asdf

    sigma_b = np.full((NY, NX), 2.0, dtype=np.float32)
    path = tmp_path / "old_calib.asdf"
    with asdf.AsdfFile({"detector": "NRCA1", "sigma_b": sigma_b, "sigma_b_median": 2.0, "t1_scale": 1.0}) as handle:
        handle.write_to(path)

    tree = read_calibration(path)

    np.testing.assert_array_equal(tree["sigma_b"], sigma_b)
    assert "pedestal_scatter" not in tree


def small_jump_model(ngroups=6, reset=20.0, rate=1.0):
    """A tiny NIRCam-like ramp of one integration with a known reset level and rate."""
    model = datamodels.RampModel((1, ngroups, NY, NX))
    model.meta.instrument.name = "NIRCAM"
    model.meta.instrument.detector = "NRCA1"
    model.meta.exposure.type = "NRC_DARK"
    model.meta.exposure.nframes = 1
    model.meta.exposure.groupgap = 0
    model.meta.exposure.frame_time = 10.0
    model.meta.exposure.group_time = 10.0
    model.meta.exposure.ngroups = ngroups
    model.meta.exposure.nints = 1
    t = 10.0 * np.arange(1, ngroups + 1)
    model.data[0] = (reset + rate * t)[:, None, None]
    return model


def test_pedestal_cube_returns_reset_level_and_unscaled_read_noise():
    """pedestal_cube on a flat ramp at 20 DN returns pedestals of 20 and read noise equal to the input over sqrt 2."""
    model = small_jump_model(rate=0.0)
    readnoise = np.full((NY, NX), 7.0, dtype=np.float32)
    gain = np.full((NY, NX), 2.0, dtype=np.float32)

    pedestal, read_noise = pedestal_cube(model, readnoise_2d=readnoise, gain_2d=gain)

    np.testing.assert_allclose(pedestal[0], 20.0, atol=1e-3)
    np.testing.assert_allclose(read_noise, 7.0 / np.sqrt(2.0))
    np.testing.assert_array_equal(readnoise, 7.0)


def test_update_timing_keeps_sigma_b(tmp_path):
    """update_timing rewrites t1_scale, reset_offset and the per-channel lists and leaves sigma_b untouched."""
    from snowblind.reset_calib import update_timing

    sigma_b = np.full((NY, NX), 2.5, dtype=np.float32)
    path = tmp_path / "calib.asdf"
    write_calibration(path, sigma_b, np.zeros_like(sigma_b), np.full((NY, NX), 6, dtype=np.int32), "NRCA1", [])

    update_timing(path, 0.9, -3.5, t1_scale_channels=[0.9, 0.91, 0.89, 0.9], reset_offset_channels=[-3.0, -4.0, -3.5, -3.5])
    tree = read_calibration(path)

    np.testing.assert_array_equal(tree["sigma_b"], sigma_b)
    assert tree["t1_scale"] == 0.9
    assert tree["reset_offset"] == -3.5
    assert tree["t1_scale_channels"] == [0.9, 0.91, 0.89, 0.9]


def test_write_calibration_adds_history_entry_naming_snowblind(tmp_path):
    """write_calibration writes one ASDF history entry whose software name is snowblind and whose text names the detector."""
    import asdf

    calib = tmp_path / "nrca1_reset_calib.asdf"
    write_calibration(
        calib, np.full((8, 8), 12.0, dtype=np.float32), np.zeros((8, 8), dtype=np.float32),
        np.full((8, 8), 10, dtype=np.int32), "NRCA1", ["dark_uncal.fits"],
    )

    with asdf.open(str(calib)) as handle:
        entries = handle.get_history_entries()
    assert len(entries) == 1
    assert entries[0]["software"]["name"] == "snowblind"
    assert "NRCA1" in entries[0]["description"]


def test_update_timing_keeps_history_and_adds_entry(tmp_path):
    """update_timing on a calibration with one history entry leaves two, the second naming the timing source file."""
    import asdf

    from snowblind.reset_calib import update_timing

    calib = tmp_path / "nrca1_reset_calib.asdf"
    write_calibration(
        calib, np.full((8, 8), 12.0, dtype=np.float32), np.zeros((8, 8), dtype=np.float32),
        np.full((8, 8), 10, dtype=np.int32), "NRCA1", ["dark_uncal.fits"],
    )

    update_timing(calib, 0.9, -3.0, provenance={"file": "field_uncal.fits"})

    with asdf.open(str(calib)) as handle:
        entries = handle.get_history_entries()
    assert len(entries) == 2
    assert "field_uncal.fits" in entries[1]["description"]


def test_reference_superbias_picks_the_latest_delivery():
    """reference_superbias on names ending 0252 and 0257 returns the 0257 file; on an empty list returns None."""
    from snowblind.reset_calib import reference_superbias

    chosen = reference_superbias(["jwst_niriss_superbias_0252.fits", "jwst_niriss_superbias_0257.fits", None])

    assert chosen == "jwst_niriss_superbias_0257.fits"
    assert reference_superbias([None]) is None


def test_run_detector1_to_jump_accepts_the_snowblind_flag():
    """run_detector1_to_jump has a snowblind keyword, default False, so the panel scripts can request SnowblindStep."""
    import inspect

    from snowblind.reset_calib import run_detector1_to_jump

    parameter = inspect.signature(run_detector1_to_jump).parameters["snowblind"]

    assert parameter.default is False


def test_run_detector1_to_jump_accepts_jump_params():
    """run_detector1_to_jump has a jump_params keyword, default None, for overriding the jump step's reference parameters."""
    import inspect

    from snowblind.reset_calib import run_detector1_to_jump

    parameter = inspect.signature(run_detector1_to_jump).parameters["jump_params"]

    assert parameter.default is None

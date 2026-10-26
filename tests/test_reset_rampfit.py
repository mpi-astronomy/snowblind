import numpy as np
import pytest
from stdatamodels.jwst import datamodels

from snowblind import ResetRampFitStep
from snowblind.reset_calib import write_calibration

SAT = datamodels.dqflags.group["SATURATED"]
NY, NX = 8, 8
FRAME_TIME = 10.73677
RATE = 20.0


def set_subarray(model):
    model.meta.subarray.name = "FULL"
    model.meta.subarray.xstart = 1
    model.meta.subarray.ystart = 1
    model.meta.subarray.xsize = NX
    model.meta.subarray.ysize = NY
    model.meta.subarray.fastaxis = -1
    model.meta.subarray.slowaxis = 2
    model.meta.instrument.name = "NIRCAM"
    model.meta.instrument.detector = "NRCA1"


def make_ramp(ngroups=5):
    """Linear ramp of RATE DN/s, one frame per group, first read one frame time after reset."""
    model = datamodels.RampModel((1, ngroups, NY, NX))
    set_subarray(model)
    model.meta.exposure.type = "NRC_IMAGE"
    model.meta.exposure.nframes = 1
    model.meta.exposure.groupgap = 0
    model.meta.exposure.frame_time = FRAME_TIME
    model.meta.exposure.group_time = FRAME_TIME
    model.meta.exposure.ngroups = ngroups
    model.meta.exposure.nints = 1
    model.meta.filename = "jw00000000000_00000_00000_nrca1_jump.fits"
    t = FRAME_TIME * np.arange(1, ngroups + 1)
    model.data[0] = (RATE * t)[:, None, None]
    return model


def reference_files(tmp_path):
    """Save a readnoise and a gain reference model and return their paths."""
    readnoise = datamodels.ReadnoiseModel((NY, NX))
    readnoise.data[:] = 7.0
    set_subarray(readnoise)
    gain = datamodels.GainModel((NY, NX))
    gain.data[:] = 2.0
    set_subarray(gain)
    rn_path, gain_path = tmp_path / "readnoise.fits", tmp_path / "gain.fits"
    readnoise.save(rn_path)
    gain.save(gain_path)
    return str(rn_path), str(gain_path)


def test_init():
    """ResetRampFitStep() has the class alias reset_rampfit."""
    step = ResetRampFitStep()

    assert step.class_alias == "reset_rampfit"


def test_call_scalar_sigma_b_recovers_rate(tmp_path):
    """ResetRampFitStep.call with sigma_b 15 on a clean linear ramp returns the rate everywhere and marks COMPLETE."""
    rn, gain = reference_files(tmp_path)
    model = make_ramp()

    rate, rateints = ResetRampFitStep.call(model, sigma_b=15.0, override_readnoise=rn, override_gain=gain)

    np.testing.assert_allclose(rate.data, RATE, rtol=1e-5)
    assert rate.meta.cal_step.reset_rampfit == "COMPLETE"
    assert rate.meta.cal_step.ramp_fit == "COMPLETE"
    assert rateints.data.shape == (1, NY, NX)


def test_call_one_good_group_pixel_gets_rate(tmp_path):
    """ResetRampFitStep.call gives a pixel saturated from group 2 on the rate R1 / t1."""
    rn, gain = reference_files(tmp_path)
    model = make_ramp()
    model.groupdq[0, 1:, 3, 3] = SAT

    rate, _ = ResetRampFitStep.call(model, sigma_b=15.0, override_readnoise=rn, override_gain=gain)

    assert rate.data[3, 3] == pytest.approx(model.data[0, 0, 3, 3] / FRAME_TIME, rel=1e-5)
    assert np.isfinite(rate.err[3, 3])


def test_call_with_calibration_file_per_pixel(tmp_path):
    """ResetRampFitStep.call with a calibration file, per_pixel_sigma_b and t1_scale 0.95 scales the one-group rate."""
    rn, gain = reference_files(tmp_path)
    model = make_ramp()
    model.groupdq[0, 1:, 2, 5] = SAT
    sigma_b = np.full((NY, NX), 12.0, dtype=np.float32)
    sigma_b[0, 0] = np.nan
    calib = tmp_path / "nrca1_reset_calib.asdf"
    write_calibration(calib, sigma_b, np.zeros_like(sigma_b), np.full((NY, NX), 10, dtype=np.int32), "NRCA1", [], t1_scale=0.95)

    rate, _ = ResetRampFitStep.call(
        model, reset_calib=str(calib), per_pixel_sigma_b=True, override_readnoise=rn, override_gain=gain
    )

    assert rate.data[2, 5] == pytest.approx(model.data[0, 0, 2, 5] / (0.95 * FRAME_TIME), rel=1e-5)
    assert np.isfinite(rate.data[0, 0])


def test_call_rejects_calibration_for_other_detector(tmp_path):
    """ResetRampFitStep.call raises ValueError when the calibration file names a different detector."""
    rn, gain = reference_files(tmp_path)
    model = make_ramp()
    calib = tmp_path / "nrcb1_reset_calib.asdf"
    sigma_b = np.full((NY, NX), 12.0, dtype=np.float32)
    write_calibration(calib, sigma_b, np.zeros_like(sigma_b), np.full((NY, NX), 10, dtype=np.int32), "NRCB1", [])

    with pytest.raises(ValueError, match="NRCB1"):
        ResetRampFitStep.call(model, reset_calib=str(calib), override_readnoise=rn, override_gain=gain)


def test_call_requires_sigma_b_for_a_detector_without_packaged_calibration(tmp_path):
    """ResetRampFitStep.call without reset_calib or sigma_b on a detector the packaged table lacks raises ValueError."""
    rn, gain = reference_files(tmp_path)
    model = make_ramp()
    model.meta.instrument.detector = "NRS1"

    with pytest.raises(ValueError, match="No packaged reset calibration"):
        ResetRampFitStep.call(model, override_readnoise=rn, override_gain=gain)


def test_call_skips_two_groups(tmp_path):
    """ResetRampFitStep.call on a two-group ramp returns (None, None), as the jwst step does when it cannot fit."""
    rn, gain = reference_files(tmp_path)
    model = make_ramp(ngroups=2)

    rate, rateints = ResetRampFitStep.call(model, sigma_b=15.0, override_readnoise=rn, override_gain=gain)

    assert rate is None
    assert rateints is None


def test_call_rejects_zero_sigma_b(tmp_path):
    """ResetRampFitStep.call with sigma_b=0 raises ValueError instead of dividing by zero."""
    rn, gain = reference_files(tmp_path)
    model = make_ramp()

    with pytest.raises(ValueError, match="positive"):
        ResetRampFitStep.call(model, sigma_b=0.0, override_readnoise=rn, override_gain=gain)


def test_save_opt_writes_resetfit(tmp_path, tmp_cwd):
    """ResetRampFitStep.call with save_opt writes a _resetfit.asdf holding used_reset and d0_resid planes."""
    import asdf

    rn, gain = reference_files(tmp_path)
    model = make_ramp()

    ResetRampFitStep.call(model, sigma_b=15.0, save_opt=True, override_readnoise=rn, override_gain=gain)

    files = list(tmp_cwd.glob("*_resetfit.asdf"))
    assert len(files) == 1
    with asdf.open(files[0]) as handle:
        assert handle.tree["used_reset"].shape == (1, NY, NX)
        assert handle.tree["d0_resid"].shape == (1, NY, NX)
        assert handle.tree["used_reset_fraction"] == 1.0


def test_call_with_pedestal_mean_and_offset(tmp_path):
    """ResetRampFitStep.call with use_pedestal_mean and reset_offset -2 subtracts mean plus offset from the one-group pixel."""
    rn, gain = reference_files(tmp_path)
    model = make_ramp()
    model.groupdq[0, 1:, 4, 4] = SAT
    sigma_b = np.full((NY, NX), 12.0, dtype=np.float32)
    mean = np.zeros((NY, NX), dtype=np.float32)
    mean[4, 4] = 6.0
    calib = tmp_path / "nrca1_reset_calib.asdf"
    write_calibration(calib, sigma_b, mean, np.full((NY, NX), 10, dtype=np.int32), "NRCA1", [])

    rate, _ = ResetRampFitStep.call(
        model, reset_calib=str(calib), use_pedestal_mean=True, reset_offset=-2.0, prior_highpass=False,
        override_readnoise=rn, override_gain=gain,
    )

    assert rate.data[4, 4] == pytest.approx((model.data[0, 0, 4, 4] - 4.0) / FRAME_TIME, rel=1e-5)


def test_call_use_pedestal_mean_requires_calibration(tmp_path):
    """ResetRampFitStep.call with use_pedestal_mean but no reset_calib raises ValueError."""
    rn, gain = reference_files(tmp_path)
    model = make_ramp()

    with pytest.raises(ValueError, match="use_pedestal_mean"):
        ResetRampFitStep.call(model, sigma_b=15.0, use_pedestal_mean=True, override_readnoise=rn, override_gain=gain)


def test_call_uses_per_channel_timing_from_file(tmp_path):
    """ResetRampFitStep.call with per-channel t1_scale in the file gives a one-group pixel its own channel's scale."""
    from snowblind import readout
    from snowblind.reset_calib import update_timing

    rn, gain = reference_files(tmp_path)
    model = make_ramp()
    blocks = readout.channel_blocks((NY, NX), -1, 2)
    y, x = 3, blocks[2][1].start
    model.groupdq[0, 1:, y, x] = SAT
    sigma_b = np.full((NY, NX), 12.0, dtype=np.float32)
    calib = tmp_path / "nrca1_reset_calib.asdf"
    write_calibration(calib, sigma_b, np.zeros_like(sigma_b), np.full((NY, NX), 10, dtype=np.int32), "NRCA1", [])
    update_timing(calib, 0.9, -1.0, t1_scale_channels=[1.0, 1.0, 0.8, 1.0], reset_offset_channels=[0.0, 0.0, -2.0, 0.0])

    rate, _ = ResetRampFitStep.call(model, reset_calib=str(calib), override_readnoise=rn, override_gain=gain)

    assert rate.data[y, x] == pytest.approx((model.data[0, 0, y, x] + 2.0) / (0.8 * FRAME_TIME), rel=1e-5)


def test_call_t1_scale_override_disables_per_channel_timing(tmp_path):
    """ResetRampFitStep.call with an explicit t1_scale ignores the file's per-channel scales."""
    from snowblind.reset_calib import update_timing

    rn, gain = reference_files(tmp_path)
    model = make_ramp()
    model.groupdq[0, 1:, 2, 2] = SAT
    sigma_b = np.full((NY, NX), 12.0, dtype=np.float32)
    calib = tmp_path / "nrca1_reset_calib.asdf"
    write_calibration(calib, sigma_b, np.zeros_like(sigma_b), np.full((NY, NX), 10, dtype=np.int32), "NRCA1", [])
    update_timing(calib, 0.9, 0.0, t1_scale_channels=[0.5, 0.5, 0.5, 0.5])

    rate, _ = ResetRampFitStep.call(model, reset_calib=str(calib), t1_scale=1.0, override_readnoise=rn, override_gain=gain)

    assert rate.data[2, 2] == pytest.approx(model.data[0, 0, 2, 2] / FRAME_TIME, rel=1e-5)


def test_highpass_prior_removes_row_pattern_and_keeps_channel_mean():
    """highpass_prior on a map of 5 DN plus a zero-mean 2 DN row pattern returns 5 DN in every row to 1e-4."""
    from snowblind.reset_rampfit import highpass_prior

    rng = np.random.default_rng(11)
    rows = rng.normal(0.0, 2.0, 64)
    rows -= rows.mean()
    pedestal = (5.0 + rows[:, None] + np.zeros((64, 64))).astype(np.float32)

    cleaned = highpass_prior(pedestal, -1, 2)

    assert np.abs(cleaned - 5.0).max() < 1e-4


def test_highpass_prior_keeps_isolated_pixel_features():
    """highpass_prior on a flat map with four isolated +100 DN pixels keeps them within 3 DN of 100."""
    from snowblind.reset_rampfit import highpass_prior

    pedestal = np.zeros((64, 64), dtype=np.float32)
    for y, x in ((5, 7), (20, 40), (33, 12), (50, 60)):
        pedestal[y, x] = 100.0

    cleaned = highpass_prior(pedestal, -1, 2)

    assert np.abs(cleaned[5, 7] - 100.0) < 3.0
    assert np.abs(cleaned[20, 40] - 100.0) < 3.0
    assert np.abs(cleaned[50, 60] - 100.0) < 3.0


def test_reset_prior_map_adds_channel_offsets_to_highpassed_map(tmp_path):
    """reset_prior_map on a flat 3 DN map with per-channel offsets 1..4 returns 3 plus the offset in each channel block."""
    import asdf

    from snowblind import readout
    from snowblind.reset_rampfit import reset_prior_map

    mean = np.full((64, 64), 3.0, dtype=np.float32)
    calib = tmp_path / "nrca1_reset_calib.asdf"
    write_calibration(calib, np.full((64, 64), 12.0, dtype=np.float32), mean, np.full((64, 64), 10, dtype=np.int32), "NRCA1", [])
    with asdf.open(str(calib), mode="rw") as handle:
        handle.tree["t1_scale_channels"] = [0.9, 0.9, 0.9, 0.9]
        handle.tree["reset_offset_channels"] = [1.0, 2.0, 3.0, 4.0]
        handle.update()

    level = reset_prior_map(str(calib), (64, 64), -1, 2)

    for k, (rows, cols) in enumerate(readout.channel_blocks((64, 64), -1, 2)):
        np.testing.assert_allclose(level[rows, cols], 3.0 + k + 1, atol=1e-4)


def test_superbias_difference_is_calibration_minus_science(tmp_path):
    """superbias_difference with two 8 x 8 superbias files of 10 and 3 DN returns +7 everywhere."""
    from astropy.io import fits

    from snowblind.reset_rampfit import superbias_difference

    paths = {}
    for name, value in (("jwst_nircam_superbias_0001.fits", 10.0), ("jwst_nircam_superbias_0002.fits", 3.0)):
        path = tmp_path / name
        fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.full((8, 8), value, dtype=np.float32), name="SCI")]).writeto(path)
        paths[name] = str(path)

    delta = superbias_difference(
        "crds://jwst_nircam_superbias_0001.fits", "jwst_nircam_superbias_0002.fits", (8, 8), locate=paths.get
    )

    np.testing.assert_allclose(delta, 7.0)


def test_superbias_difference_is_none_for_the_same_file():
    """superbias_difference with the same superbias name on both sides returns None without touching any file."""
    from snowblind.reset_rampfit import superbias_difference

    delta = superbias_difference("crds://jwst_nircam_superbias_0454.fits", "jwst_nircam_superbias_0454.fits", (8, 8), locate=None)

    assert delta is None


def test_write_calibration_records_superbias(tmp_path):
    """write_calibration with superbias='jwst_nircam_superbias_0454.fits' stores that name in the provenance."""
    import asdf

    calib = tmp_path / "nrca1_reset_calib.asdf"
    write_calibration(
        calib, np.full((8, 8), 12.0, dtype=np.float32), np.zeros((8, 8), dtype=np.float32),
        np.full((8, 8), 10, dtype=np.int32), "NRCA1", [], superbias="jwst_nircam_superbias_0454.fits",
    )

    with asdf.open(str(calib)) as handle:
        assert handle.tree["provenance"]["superbias"] == "jwst_nircam_superbias_0454.fits"


def superbias_warnings(caplog):
    """The messages of the captured WARNING (or higher) records that mention the superbias."""
    import logging

    messages = []
    for record in caplog.records:
        if record.levelno >= logging.WARNING and "superbias" in record.getMessage():
            messages.append(record.getMessage())
    return messages


def test_call_warns_when_data_superbias_differs_from_best_reference(tmp_path, caplog):
    """ResetRampFitStep.call on data recorded with superbias 0001 and override_superbias naming 0002 warns naming both."""
    import logging

    rn, gain = reference_files(tmp_path)
    model = make_ramp()
    model.meta.ref_file.superbias.name = "crds://jwst_nircam_superbias_0001.fits"
    other = tmp_path / "jwst_nircam_superbias_0002.fits"
    other.write_bytes(b"")

    with caplog.at_level(logging.WARNING):
        ResetRampFitStep.call(
            model, sigma_b=15.0, override_readnoise=rn, override_gain=gain, override_superbias=str(other)
        )

    message = superbias_warnings(caplog)[0]
    assert "jwst_nircam_superbias_0001.fits" in message
    assert "jwst_nircam_superbias_0002.fits" in message


def test_call_no_warning_when_data_superbias_matches_best_reference(tmp_path, caplog):
    """ResetRampFitStep.call on data recorded with superbias 0002 and override_superbias naming 0002 logs no warning."""
    import logging

    rn, gain = reference_files(tmp_path)
    model = make_ramp()
    model.meta.ref_file.superbias.name = "crds://jwst_nircam_superbias_0002.fits"
    same = tmp_path / "jwst_nircam_superbias_0002.fits"
    same.write_bytes(b"")

    with caplog.at_level(logging.WARNING):
        ResetRampFitStep.call(model, sigma_b=15.0, override_readnoise=rn, override_gain=gain, override_superbias=str(same))

    assert superbias_warnings(caplog) == []


def test_calibrated_pedestal_map_is_unchanged_for_another_superbias(caplog):
    """calibrated_pedestal_map with a data superbias other than the calibration's returns the map unchanged and warns."""
    import logging

    from snowblind.reset_rampfit import calibrated_pedestal_map

    mean = np.full((8, 8), 3.0, dtype=np.float32)
    tree = {"pedestal_mean": mean, "provenance": {"superbias": "jwst_nircam_superbias_0443.fits"}}

    with caplog.at_level(logging.WARNING):
        out = calibrated_pedestal_map(
            tree, (8, 8), -1, 2, highpass=False, science_superbias="crds://jwst_nircam_superbias_0220.fits"
        )

    np.testing.assert_array_equal(out, mean)
    assert any("0220" in r.getMessage() and "0443" in r.getMessage() for r in caplog.records)


def test_rescale_t1_keeps_the_delay_in_seconds():
    """rescale_t1 with 0.943 measured at 10.74 s gives a 0.612 s delay, which at 48.3 s is a scale of 0.9873."""
    from snowblind.reset_rampfit import rescale_t1

    scale = rescale_t1(0.943, 10.74, 48.3)

    assert scale == pytest.approx(1.0 - (1.0 - 0.943) * 10.74 / 48.3, rel=1e-9)
    assert scale == pytest.approx(0.9873, abs=5e-4)


def test_call_anchors_the_reset_offset_to_the_sky(tmp_path):
    """ResetRampFitStep.call with anchor_offset_to_sky on a ramp whose reset level is 7 DN above the file's gives RATE."""
    rn, gain = reference_files(tmp_path)
    model = make_ramp()
    model.data[0] += 7.0
    model.groupdq[0, 1:, 4, 4] = SAT
    calib = tmp_path / "nrca1_reset_calib.asdf"
    write_calibration(
        calib, np.full((NY, NX), 12.0, dtype=np.float32), np.zeros((NY, NX), dtype=np.float32),
        np.full((NY, NX), 10, dtype=np.int32), "NRCA1", [],
    )

    rate, _ = ResetRampFitStep.call(
        model, reset_calib=str(calib), use_pedestal_mean=True, prior_highpass=False, anchor_offset_to_sky=True,
        anchor_max_signal=1e9, anchor_min_pixels=10, override_readnoise=rn, override_gain=gain,
    )

    assert rate.data[4, 4] == pytest.approx(RATE, rel=1e-4)


def test_call_without_anchoring_is_biased_by_a_wrong_offset(tmp_path):
    """ResetRampFitStep.call without anchoring, reset 7 DN above the file's prior: the one-group pixel is 7/t1 too high."""
    rn, gain = reference_files(tmp_path)
    model = make_ramp()
    model.data[0] += 7.0
    model.groupdq[0, 1:, 4, 4] = SAT
    calib = tmp_path / "nrca1_reset_calib.asdf"
    write_calibration(
        calib, np.full((NY, NX), 12.0, dtype=np.float32), np.zeros((NY, NX), dtype=np.float32),
        np.full((NY, NX), 10, dtype=np.int32), "NRCA1", [],
    )

    rate, _ = ResetRampFitStep.call(
        model, reset_calib=str(calib), use_pedestal_mean=True, prior_highpass=False, anchor_offset_to_sky=False,
        override_readnoise=rn, override_gain=gain,
    )

    assert rate.data[4, 4] == pytest.approx(RATE + 7.0 / FRAME_TIME, rel=1e-4)


def test_star_anchored_scales_recovers_a_short_first_read():
    """star_anchored_scales on ramps whose first read holds 80 percent of its nominal signal returns 0.8 per channel."""
    from snowblind import readout
    from snowblind.reset_rampfit import star_anchored_scales

    model = make_ramp()
    rng = np.random.default_rng(21)
    rates = rng.uniform(5.0, 150.0, (NY, NX))  # nominal first-read signal 54 to 1600 DN
    t = FRAME_TIME * np.arange(1, 6)
    model.data[0] = rates[None] * t[:, None, None]
    model.data[0, 0] = rates * (t[0] - 0.2 * FRAME_TIME)
    blocks = readout.channel_blocks((NY, NX), -1, 2)

    scales = star_anchored_scales(model, 0.0, blocks, use_zeroframe=False, min_bright=5, fallback=[1.0] * 4, min_per_bin=1)

    np.testing.assert_allclose(scales, 0.8, atol=2e-3)


def test_default_calibration_has_nrca1_timing_per_channel():
    """default_calibration('nrca1') returns a positive sigma_b and four per-channel first-read scales."""
    from snowblind.reset_calib import default_calibration

    tree = default_calibration("nrca1")

    assert tree["detector"] == "NRCA1"
    assert 10.0 < tree["sigma_b_median"] < 30.0
    assert len(tree["t1_scale_channels"]) == 4


def test_default_calibration_unknown_detector_raises():
    """default_calibration('MIRIMAGE') raises a ValueError naming the detector."""
    from snowblind.reset_calib import default_calibration

    with pytest.raises(ValueError, match="MIRIMAGE"):
        default_calibration("MIRIMAGE")


def test_call_without_calibration_uses_the_packaged_defaults(tmp_path):
    """ResetRampFitStep.call on an NRCA1 ramp with neither reset_calib nor sigma_b completes with a finite rate."""
    rn, gain = reference_files(tmp_path)
    model = make_ramp()

    out, _ints = ResetRampFitStep.call(model, override_readnoise=rn, override_gain=gain)

    assert out.meta.cal_step.reset_rampfit == "COMPLETE"
    assert np.isfinite(out.data).all()


def test_call_with_save_results_writes_rate_and_rateints(tmp_path):
    """ResetRampFitStep.call with save_results and an output_dir writes a _rate.fits and a _rateints.fits there."""
    rn, gain = reference_files(tmp_path)
    model = make_ramp()

    ResetRampFitStep.call(
        model, sigma_b=15.0, override_readnoise=rn, override_gain=gain, save_results=True, output_dir=str(tmp_path)
    )

    assert len(list(tmp_path.glob("*_rate.fits"))) == 1
    assert len(list(tmp_path.glob("*_rateints.fits"))) == 1

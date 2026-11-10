import logging
import asdf
import numpy as np
import pytest
from astropy.io import fits
from stdatamodels.jwst import datamodels

from snowblind import SnowblindStep
from snowblind.snowblind import group_difference_times, longest_packaged_snowball_times, packaged_snowball_times


JUMP_DET = datamodels.dqflags.group['JUMP_DET']
GOOD = datamodels.dqflags.group['GOOD']
SATURATED = datamodels.dqflags.group['SATURATED']
REFERENCE_PIXEL = datamodels.dqflags.pixel['REFERENCE_PIXEL']


def test_init():
    snow = SnowblindStep(growth_factor=2)

    assert snow.growth_factor == 2.0


def snowball_data():
    # RampModel, i.e. from Detector1Pipeline
    im1 = datamodels.RampModel((1, 6, 40, 40))
    im1.groupdq[0, 1, 15:26, 15:26] = JUMP_DET
    im1.groupdq[0, 1, 20, 20] = SATURATED
    im1.groupdq[0, 1, 5, 5] = JUMP_DET

    # 2D ImageModel, e.g., _rate products
    im2 = datamodels.ImageModel((40, 40))
    im2.dq = np.zeros((40, 40), dtype=np.uint32)
    im2.dq[15:26, 15:26] = JUMP_DET
    im2.dq[20, 20] = SATURATED
    im2.dq[5, 5] = JUMP_DET

    # CubeModel, e.g., _rateints
    im3 = datamodels.CubeModel((6, 40, 40))
    im3.dq = np.zeros((6, 40, 40), dtype=np.uint32)
    im3.dq[1, 15:26, 15:26] = JUMP_DET
    im3.dq[1, 20, 20] = SATURATED
    im3.dq[1, 5, 5] = JUMP_DET

    return im1, im2, im3


def test_ramp():
    im = datamodels.RampModel((1, 6, 40, 40))
    im.groupdq[0, 1, 15:26, 15:26] = JUMP_DET
    im.groupdq[0, 1, 20, 20] = SATURATED
    im.groupdq[0, 1, 5, 5] = JUMP_DET

    result = SnowblindStep.call(im)

    # Verify large area got expanded
    assert result.groupdq[0, 1, 14, 14] == JUMP_DET

    # Verify single pixel area did not get expanded
    assert result.groupdq[0, 1, 6, 6] == GOOD

    # Verify that the ring around the saturated core is flagged to the end of the integration
    assert result.groupdq[0, 5, 22, 20] == JUMP_DET


def test_rate():
    # Image mode, e.g., rate products
    im = datamodels.ImageModel((40, 40))
    im.dq = np.zeros((40, 40), dtype=np.uint32)
    im.dq[15:26, 15:26] = JUMP_DET
    im.dq[20, 20] = SATURATED
    im.dq[5, 5] = JUMP_DET

    result = SnowblindStep.call(im)

    # Verify large area got expanded
    assert result.dq[14, 14] == JUMP_DET

    # Verify single pixel area did not get expanded
    assert result.dq[6, 6] == GOOD

    # Set different flag value
    result = SnowblindStep.call(im, new_jump_flag=4096)
    assert result.dq[14, 14] & 4096 == 4096


def test_rateints():
    # Cube mode, e.g., rateints
    im = datamodels.CubeModel((6, 40, 40))
    im.dq = np.zeros((6, 40, 40), dtype=np.uint32)
    im.dq[1, 15:26, 15:26] = JUMP_DET
    im.dq[1, 20, 20] = SATURATED
    im.dq[1, 5, 5] = JUMP_DET

    result = SnowblindStep.call(im)

    # Verify large area got expanded
    assert result.dq[1, 14, 14] == JUMP_DET

    # Verify single pixel area did not get expanded
    assert result.dq[1, 6, 6] == GOOD


@pytest.mark.parametrize("im", snowball_data())
def test_step_complete(im, tmp_path):
    result = SnowblindStep.call(im)

    assert result.meta.cal_step.snowblind == "COMPLETE"

    filename = tmp_path / "jw001234_blah_blah_00001_snowblind.fits"
    result.save(filename)

    with datamodels.open(filename) as result_reopened:
        assert result_reopened.meta.cal_step.snowblind == "COMPLETE"


def test_ramp_event_rate():
    """SnowblindStep.call on a ramp with one large event around a saturated core in one group sets EXTNCRS.

    The rate is 1e6 * 1 event / (group_time * (6 groups - 1 first group) * 40 * 40 pixels).
    """
    im = datamodels.RampModel((1, 6, 40, 40))
    im.meta.exposure.group_time = 10.0
    im.groupdq[0, 1, 15:26, 15:26] = JUMP_DET
    im.groupdq[0, 1, 20, 20] = SATURATED
    im.groupdq[0, 1, 5, 5] = JUMP_DET

    result = SnowblindStep.call(im)

    assert result.meta.exposure.extended_emission_events == pytest.approx(1e6 / (10.0 * 5 * 1600))


def test_ramp_logs_total_snowballs(caplog):
    """SnowblindStep.call on a ramp with a snowball in each of groups 1 and 3 logs "Total snowballs = 2"."""
    im = datamodels.RampModel((1, 6, 40, 40))
    im.groupdq[0, 1, 15:26, 15:26] = JUMP_DET
    im.groupdq[0, 1, 20, 20] = SATURATED
    im.groupdq[0, 3, 15:26, 15:26] = JUMP_DET
    im.groupdq[0, 3, 20, 20] = SATURATED

    with caplog.at_level(logging.INFO, logger="snowblind.snowblind"):
        SnowblindStep.call(im)

    assert "Total snowballs = 2" in caplog.messages


def test_ramp_event_rate_no_events():
    """SnowblindStep.call on a ramp with only a single-pixel jump sets EXTNCRS to 0."""
    im = datamodels.RampModel((1, 6, 40, 40))
    im.meta.exposure.group_time = 10.0
    im.groupdq[0, 1, 5, 5] = JUMP_DET

    result = SnowblindStep.call(im)

    assert result.meta.exposure.extended_emission_events == 0


def test_ramp_event_rate_without_group_time():
    """SnowblindStep.call on a ramp with no group_time leaves EXTNCRS unset."""
    im = datamodels.RampModel((1, 6, 40, 40))
    im.groupdq[0, 1, 15:26, 15:26] = JUMP_DET

    result = SnowblindStep.call(im)

    assert result.meta.exposure.extended_emission_events is None


def test_rate_event_rate_untouched():
    """SnowblindStep.call on a rate image keeps the EXTNCRS value it came with."""
    im = datamodels.ImageModel((40, 40))
    im.dq = np.zeros((40, 40), dtype=np.uint32)
    im.dq[15:26, 15:26] = JUMP_DET
    im.meta.exposure.extended_emission_events = 3.5

    result = SnowblindStep.call(im)

    assert result.meta.exposure.extended_emission_events == 3.5


@pytest.mark.parametrize("im", snowball_data())
def test_step_complete_in_fits_header(im, tmp_path):
    """SnowblindStep.call on a ramp, rate image or cube, then save: the primary header has S_SNOWBL = COMPLETE."""
    result = SnowblindStep.call(im)
    filename = tmp_path / "jw001234_blah_blah_00001_snowblind.fits"
    result.save(filename)

    assert fits.getheader(filename)["S_SNOWBL"] == "COMPLETE"


def test_step_complete_in_fits_header_after_resave(tmp_path):
    """Saving a reopened SnowblindStep product keeps S_SNOWBL = COMPLETE once, without a duplicate card."""
    result = SnowblindStep.call(snowball_data()[1])
    first = tmp_path / "first.fits"
    second = tmp_path / "second.fits"
    result.save(first)
    with datamodels.open(first) as reopened:
        SnowblindStep.call(reopened).save(second)

    assert list(fits.getheader(second)).count("S_SNOWBL") == 1


def hole_slice(hole_area):
    """A 80x80 jump slice with a 60x60 block of JUMP_DET and a hole of ``hole_area`` pixels in it."""
    jump = np.zeros((80, 80), dtype=bool)
    jump[10:70, 10:70] = True
    rows, extra = divmod(hole_area, 10)
    jump[20:20 + rows, 20:30] = False
    jump[20 + rows, 20:20 + extra] = False
    return jump


def test_dilate_jump_slice_fills_a_hole_of_200_pixels():
    """SnowblindStep.dilate_jump_slice with growth_factor=0 on a block with a 200 pixel hole holding one saturated pixel
    fills the hole."""
    jump = hole_slice(200)
    sat = np.zeros_like(jump)
    sat[25, 25] = True

    dilated, cores = SnowblindStep(growth_factor=0).dilate_jump_slice(jump, sat)

    assert dilated[22, 22]


def test_dilate_jump_slice_keeps_a_hole_of_201_pixels():
    """SnowblindStep.dilate_jump_slice with growth_factor=0 on a block with a 201 pixel hole leaves the hole open.

    A saturated 1 pixel hole elsewhere in the block is its core, so the block itself is kept.
    """
    jump = hole_slice(201)
    jump[50, 50] = False
    sat = np.zeros_like(jump)
    sat[50, 50] = True

    dilated, cores = SnowblindStep(growth_factor=0).dilate_jump_slice(jump, sat)

    assert dilated[40, 40]
    assert not dilated[25, 25]


def test_dilate_jump_slice_keeps_a_hole_of_225_pixels_with_one_saturated_pixel():
    """SnowblindStep.dilate_jump_slice with growth_factor=0 on a block with a 225 pixel hole holding one saturated pixel
    leaves the hole open.

    Most of the hole is more than 1 pixel from a saturated pixel, so it is sky rather than a core.  A saturated 1 pixel
    hole elsewhere in the block is its core, so the block itself is kept.
    """
    jump = np.zeros((80, 80), dtype=bool)
    jump[10:70, 10:70] = True
    jump[20:35, 20:35] = False
    jump[50, 50] = False
    sat = np.zeros_like(jump)
    sat[27, 27] = True
    sat[50, 50] = True

    dilated, cores = SnowblindStep(growth_factor=0).dilate_jump_slice(jump, sat)

    assert dilated[40, 40]
    assert not dilated[22, 22]


def test_dilate_jump_slice_fills_a_hole_of_225_pixels_holding_a_saturated_core():
    """SnowblindStep.dilate_jump_slice with growth_factor=0 on a block with a 225 pixel saturated hole fills the hole."""
    jump = np.zeros((80, 80), dtype=bool)
    jump[10:70, 10:70] = True
    jump[20:35, 20:35] = False
    sat = np.zeros_like(jump)
    sat[20:35, 20:35] = True

    dilated, cores = SnowblindStep(growth_factor=0).dilate_jump_slice(jump, sat)

    assert dilated[25, 25]


def test_ramp_without_saturated_core():
    """SnowblindStep.call on a ramp with a large jump region and no saturated pixel inside leaves it ungrown."""
    im = datamodels.RampModel((1, 6, 40, 40))
    im.groupdq[0, 1, 15:26, 15:26] = JUMP_DET

    result = SnowblindStep.call(im)

    assert result.groupdq[0, 1, 14, 14] == GOOD


def test_ramp_core_saturating_in_the_next_group():
    """SnowblindStep.call on a ramp whose large jump region in group 1 holds a pixel that saturates in group 2 grows it."""
    im = datamodels.RampModel((1, 6, 40, 40))
    im.groupdq[0, 1, 15:26, 15:26] = JUMP_DET
    im.groupdq[0, 2:, 20, 20] = SATURATED

    result = SnowblindStep.call(im)

    assert result.groupdq[0, 1, 14, 14] == JUMP_DET


def test_ramp_ring_leaves_out_earlier_saturation():
    """SnowblindStep.call on a ramp with a pixel saturated from group 0 inside a grown halo puts no ring around it.

    Only the core of the event, which saturates in the jump group, is carried into the next groups.
    """
    im = datamodels.RampModel((1, 6, 40, 40))
    im.groupdq[0, 1, 15:26, 15:26] = JUMP_DET
    im.groupdq[0, 1:, 20, 20] = SATURATED
    im.groupdq[0, :, 10, 30] = SATURATED

    result = SnowblindStep.call(im)

    assert result.groupdq[0, 3, 11, 30] == GOOD


def test_dilate_saturated_cores_stops_at_the_end_of_the_integration():
    """SnowblindStep.dilate_saturated_cores with after_jumps=10 on a core in group 3 of 6 flags groups 3 to 5 only."""
    cores = np.zeros((1, 6, 9, 9), dtype=bool)
    cores[0, 3, 4, 4] = True

    ring = SnowblindStep(after_jumps=10).dilate_saturated_cores(cores)

    assert ring[0].any(axis=(1, 2)).tolist() == [False, False, False, True, True, True]


def test_dilate_saturated_cores_flags_the_ring_to_the_end_of_the_integration():
    """SnowblindStep().dilate_saturated_cores on a core in group 1 of 6 flags the ring in groups 1 to 5."""
    cores = np.zeros((1, 6, 9, 9), dtype=bool)
    cores[0, 1, 4, 4] = True

    ring = SnowblindStep().dilate_saturated_cores(cores)

    assert ring[0].any(axis=(1, 2)).tolist() == [False, True, True, True, True, True]


def test_dilate_saturated_cores_with_after_jumps_flags_that_many_groups_after_the_jump():
    """SnowblindStep(after_jumps=1).dilate_saturated_cores on a core in group 1 of 6 flags the ring in groups 1 and 2."""
    cores = np.zeros((1, 6, 9, 9), dtype=bool)
    cores[0, 1, 4, 4] = True

    ring = SnowblindStep(after_jumps=1).dilate_saturated_cores(cores)

    assert ring[0].any(axis=(1, 2)).tolist() == [False, True, True, False, False, False]


def test_ramp_with_after_jumps_warns_that_it_is_deprecated():
    """SnowblindStep(after_jumps=2).process on a ramp raises a DeprecationWarning.

    process, not call: stpipe's call turns warnings into log records.
    """
    im = datamodels.RampModel((1, 6, 40, 40))

    with pytest.warns(DeprecationWarning, match="after_jumps is deprecated"):
        SnowblindStep(after_jumps=2).process(im)


def test_ramp_without_jumps_flags_nothing():
    """SnowblindStep.call on a ramp with no flags in groupdq leaves groupdq all GOOD.

    The ring dilation once set JUMP_DET at pixels (0, 0), (0, 1) and (1, 0) of every group.
    """
    im = datamodels.RampModel((1, 5, 20, 20))

    result = SnowblindStep.call(im)

    assert (result.groupdq == GOOD).all()


def ramp_with_snowball_in_first_integration(nints):
    """A ramp of nints integrations of 6 groups of one 10 s frame, with one snowball in group 2 of the first.

    Its core is the pixel [20, 20].  Each integration lasts 60 s and the next starts after a
    10 s reset frame, so the differences of integration 2 start 10, 20, ..., 60 s after the
    end of integration 1, and those of integration 3 80, 90, ..., 130 s after it.
    """
    im = datamodels.RampModel((nints, 6, 40, 40))
    im.meta.instrument.detector = "NRCA1"
    im.meta.exposure.frame_time = 10.0
    im.meta.exposure.group_time = 10.0
    im.meta.exposure.nframes = 1
    im.meta.exposure.groupgap = 0
    im.groupdq[0, 2, 15:26, 15:26] = JUMP_DET
    im.groupdq[0, 2:, 20, 20] = SATURATED
    return im


def int_times_with_gap(gap):
    """An int_times table of two 60 s integrations, the second starting gap seconds after the first ends."""
    names = ["integration_number", "int_start_MJD_UTC", "int_mid_MJD_UTC", "int_end_MJD_UTC",
             "int_start_BJD_TDB", "int_mid_BJD_TDB", "int_end_BJD_TDB"]
    table = np.zeros(2, dtype=[(names[0], "i4")] + [(name, "f8") for name in names[1:]])
    table["integration_number"] = [1, 2]
    table["int_start_MJD_UTC"] = np.array([0.0, 60.0 + gap]) / 86400 + 60000.0
    table["int_end_MJD_UTC"] = table["int_start_MJD_UTC"] + 60.0 / 86400
    return table


def test_ramp_core_flagged_in_the_next_integration_for_the_core_persistence_time():
    """SnowblindStep.call with core_persistence_time=35 s flags the core in groups 0 to 2 of integration 2.

    Their differences start 10, 20 and 30 s after the end of integration 1.
    """
    result = SnowblindStep.call(
        ramp_with_snowball_in_first_integration(2), core_persistence_time=35.0, neighbor_persistence_time=15.0
    )

    assert result.groupdq[1, :, 20, 20].tolist() == [JUMP_DET] * 3 + [GOOD] * 3


def test_ramp_core_neighbor_flagged_in_the_next_integration_for_the_neighbor_persistence_time():
    """SnowblindStep.call with neighbor_persistence_time=15 s flags the pixel next to the core in group 0 of integration 2."""
    result = SnowblindStep.call(
        ramp_with_snowball_in_first_integration(2), core_persistence_time=35.0, neighbor_persistence_time=15.0
    )

    assert result.groupdq[1, :, 21, 21].tolist() == [JUMP_DET] + [GOOD] * 5


def test_ramp_pixel_two_pixels_from_the_core_not_flagged_in_the_next_integration():
    """SnowblindStep.call with both persistence times at 1000 s leaves a pixel 2 px from the core GOOD in integration 2."""
    result = SnowblindStep.call(
        ramp_with_snowball_in_first_integration(2), core_persistence_time=1000.0, neighbor_persistence_time=1000.0
    )

    assert (result.groupdq[1, :, 20, 22] == GOOD).all()


def test_ramp_core_flagged_two_integrations_later_for_the_core_persistence_time():
    """SnowblindStep.call with core_persistence_time=100 s flags the core in groups 0 and 1 of integration 3.

    Their differences start 80 and 90 s after the end of integration 1.
    """
    result = SnowblindStep.call(
        ramp_with_snowball_in_first_integration(3), core_persistence_time=100.0, neighbor_persistence_time=15.0
    )

    assert result.groupdq[2, :, 20, 20].tolist() == [JUMP_DET] * 2 + [GOOD] * 4


def test_ramp_core_hit_in_two_integrations_flagged_from_the_later_hit():
    """SnowblindStep.call with core_persistence_time=35 s on a ramp whose core is hit in integrations 1 and 2 flags it
    in groups 0 to 2 of integration 3.

    Their differences start 10, 20 and 30 s after the end of integration 2, and 80 s or more after that of
    integration 1.
    """
    im = ramp_with_snowball_in_first_integration(3)
    im.groupdq[1, 2, 15:26, 15:26] = JUMP_DET
    im.groupdq[1, 2:, 20, 20] = SATURATED

    result = SnowblindStep.call(im, core_persistence_time=35.0, neighbor_persistence_time=15.0)

    assert result.groupdq[2, :, 20, 20].tolist() == [JUMP_DET] * 3 + [GOOD] * 3


def test_ramp_core_persistence_timed_from_int_times():
    """SnowblindStep.call with core_persistence_time=500 s on integrations 1000 s apart in int_times leaves the core GOOD."""
    im = ramp_with_snowball_in_first_integration(2)
    im.int_times = int_times_with_gap(1000.0)

    result = SnowblindStep.call(im, core_persistence_time=500.0, neighbor_persistence_time=500.0)

    assert (result.groupdq[1, :, 20, 20] == GOOD).all()


def test_ramp_takes_the_packaged_core_time_of_the_detector():
    """SnowblindStep.call on an NRCA1 ramp, integrations 1000 s apart, flags the core in integration 2 (packaged 1935 s)."""
    im = ramp_with_snowball_in_first_integration(2)
    im.int_times = int_times_with_gap(1000.0)

    result = SnowblindStep.call(im)

    assert (result.groupdq[1, :, 20, 20] == JUMP_DET).all()


def test_ramp_takes_the_packaged_neighbor_time_of_the_detector():
    """SnowblindStep.call on an NRCA1 ramp, integrations 1000 s apart, leaves the core's neighbor GOOD (packaged 701 s)."""
    im = ramp_with_snowball_in_first_integration(2)
    im.int_times = int_times_with_gap(1000.0)

    result = SnowblindStep.call(im)

    assert (result.groupdq[1, :, 20, 21] == GOOD).all()


def package_times_of_other_detectors(directory):
    """Packaged times for NRCA2 (core 500 s, neighbors 50 s) and NRCB4 (core 2000 s, neighbors 30 s) in directory."""
    for detector, core_time, neighbor_time in [("nrca2", 500.0, 50.0), ("nrcb4", 2000.0, 30.0)]:
        tree = {"core_time": core_time, "neighbor_time": neighbor_time}
        asdf.AsdfFile(tree).write_to(directory / f"snowball_persistence_{detector}.asdf")


def test_longest_packaged_snowball_times_of_two_files(monkeypatch, tmp_path):
    """longest_packaged_snowball_times over NRCA2 (500 s, 50 s) and NRCB4 (2000 s, 30 s) returns (2000 s, 50 s)."""
    package_times_of_other_detectors(tmp_path)
    monkeypatch.setattr("snowblind.snowblind.PACKAGED_DIR", tmp_path)

    assert longest_packaged_snowball_times() == (2000.0, 50.0)


def test_ramp_without_its_own_packaged_times_logs_a_warning(monkeypatch, tmp_path, caplog):
    """SnowblindStep.call on an NRCA1 ramp with times packaged for other detectors only logs a warning."""
    package_times_of_other_detectors(tmp_path)
    monkeypatch.setattr("snowblind.snowblind.PACKAGED_DIR", tmp_path)

    with caplog.at_level(logging.WARNING, logger="snowblind.snowblind"):
        SnowblindStep.call(ramp_with_snowball_in_first_integration(2))

    assert "No snowball persistence times are packaged for detector NRCA1" in caplog.text


def test_ramp_without_its_own_packaged_times_flags_the_core_for_the_longest_core_time(monkeypatch, tmp_path):
    """SnowblindStep.call on an NRCA1 ramp, longest packaged core time 2000 s, integrations 1000 s apart, flags the core."""
    package_times_of_other_detectors(tmp_path)
    monkeypatch.setattr("snowblind.snowblind.PACKAGED_DIR", tmp_path)
    im = ramp_with_snowball_in_first_integration(2)
    im.int_times = int_times_with_gap(1000.0)

    result = SnowblindStep.call(im)

    assert (result.groupdq[1, :, 20, 20] == JUMP_DET).all()


def test_ramp_without_its_own_packaged_times_flags_the_neighbor_for_the_longest_neighbor_time(monkeypatch, tmp_path):
    """SnowblindStep.call on an NRCA1 ramp, longest packaged neighbor time 50 s, flags the neighbor in group 0 only.

    With integrations 40 s apart, the differences of integration 2 start 40, 50, ..., 90 s after the first ends.
    """
    package_times_of_other_detectors(tmp_path)
    monkeypatch.setattr("snowblind.snowblind.PACKAGED_DIR", tmp_path)
    im = ramp_with_snowball_in_first_integration(2)
    im.int_times = int_times_with_gap(40.0)

    result = SnowblindStep.call(im)

    assert result.groupdq[1, :, 21, 21].tolist() == [JUMP_DET] + [GOOD] * 5


def test_ramp_without_any_packaged_times_leaves_later_integrations(monkeypatch, tmp_path):
    """SnowblindStep.call on a ramp with no packaged times at all leaves integration 2 GOOD."""
    monkeypatch.setattr("snowblind.snowblind.PACKAGED_DIR", tmp_path)

    result = SnowblindStep.call(ramp_with_snowball_in_first_integration(2))

    assert (result.groupdq[1] == GOOD).all()


@pytest.mark.parametrize(
    "detector",
    [
        "NRCA1", "NRCA2", "NRCA3", "NRCA4", "NRCALONG", "NRCB1", "NRCB2", "NRCB3", "NRCB4", "NRCBLONG", "NIS",
        "NRS1", "NRS2",
    ],
)
def test_packaged_snowball_times_has_a_core_time_longer_than_the_neighbor_time(detector):
    """packaged_snowball_times for each NIRCam, NIRISS and NIRSpec detector returns a core time longer than the neighbor time."""
    core_time, neighbor_time = packaged_snowball_times(detector)

    assert core_time > neighbor_time > 0


def test_group_difference_times_from_the_frame_counts():
    """group_difference_times on 2 integrations of 3 groups of 4 frames plus 1 dropped, 10 s frames, gives their starts.

    An integration spans 15 frames (150 s) and the next starts after one reset frame, at 160 s.
    """
    im = datamodels.RampModel((2, 3, 4, 4))
    im.meta.exposure.frame_time = 10.0
    im.meta.exposure.nframes = 4
    im.meta.exposure.groupgap = 1

    starts, ends = group_difference_times(im)

    assert starts.tolist() == [[0.0, 40.0, 90.0], [160.0, 200.0, 250.0]]
    assert ends.tolist() == [150.0, 310.0]


def test_group_difference_times_without_a_frame_time_is_none():
    """group_difference_times on a ramp with no frame_time returns None."""
    assert group_difference_times(datamodels.RampModel((2, 3, 4, 4))) is None


def test_rate_without_hole():
    """SnowblindStep.call on a rate image with a large jump region and no hole in it leaves it ungrown."""
    im = datamodels.ImageModel((40, 40))
    im.dq = np.zeros((40, 40), dtype=np.uint32)
    im.dq[15:26, 15:26] = JUMP_DET

    result = SnowblindStep.call(im)

    assert result.dq[14, 14] == GOOD


def test_rate_with_a_hole_that_is_not_saturated():
    """SnowblindStep.call on a rate image with a large jump region around a 1 pixel hole that is not SATURATED leaves
    it ungrown."""
    im = datamodels.ImageModel((40, 40))
    im.dq = np.zeros((40, 40), dtype=np.uint32)
    im.dq[15:26, 15:26] = JUMP_DET
    im.dq[20, 20] = GOOD

    result = SnowblindStep.call(im)

    assert result.dq[14, 14] == GOOD


def test_rate_with_a_saturated_core_of_253_pixels():
    """SnowblindStep.call on a rate image with a JUMP_DET halo of radius 18 around a SATURATED core of 253 pixels
    grows the halo."""
    im = datamodels.ImageModel((100, 150))
    halo, core = round_event((100, 150), (50, 75), 18, 9)
    im.dq = np.zeros((100, 150), dtype=np.uint32)
    im.dq[halo] = JUMP_DET
    im.dq[core] = SATURATED

    result = SnowblindStep.call(im)

    assert int(core.sum()) == 253
    assert result.dq[50, 75 + 25] & JUMP_DET


def test_rate_with_a_jump_track_across_it():
    """SnowblindStep.call on a rate image with a JUMP_DET track across it and a SATURATED pixel on its narrow side
    leaves that side unflagged.

    The narrow side is sky cut off by the track, not a hole around a core.
    """
    im = datamodels.ImageModel((64, 512))
    im.dq = np.zeros((64, 512), dtype=np.uint32)
    im.dq[:, 100:103] = JUMP_DET
    im.dq[30, 50] = SATURATED

    result = SnowblindStep.call(im)

    assert not result.dq[10, 10] & JUMP_DET


def round_event(shape, center, radius, core_radius):
    """Masks of a round halo and of its core, centered at ``center``, which may lie off the array."""
    distance = np.hypot(*(np.indices(shape) - np.array(center)[:, None, None]))
    return (distance <= radius) & (distance > core_radius), distance <= core_radius


def ramp_with_event(halo, core, refpix):
    """A ramp of 1 integration and 3 groups: the halo JUMP_DET in group 1 and the core SATURATED from group 1.

    The halo and core are left off the reference pixels, which get REFERENCE_PIXEL in pixeldq.
    """
    im = datamodels.RampModel((1, 3) + halo.shape)
    im.groupdq[0, 1][halo & ~refpix] = JUMP_DET
    im.groupdq[0, 1:][:, core & ~refpix] = SATURATED
    im.pixeldq[refpix] = REFERENCE_PIXEL
    return im


def bottom_reference_rows(shape):
    """Reference pixels in the first 4 rows."""
    refpix = np.zeros(shape, dtype=bool)
    refpix[:4] = True
    return refpix


def test_ramp_snowball_centered_on_the_first_light_sensitive_row():
    """SnowblindStep.call on a ramp with a snowball of radius 8 centered on the first row after 4 reference rows
    flags the row 22 pixels along from its center.

    The event is completed past the edge, so it grows by twice its full radius, to about 24 pixels.
    """
    refpix = bottom_reference_rows((40, 80))
    im = ramp_with_event(*round_event((40, 80), (4, 30), 8, 2), refpix)

    result = SnowblindStep.call(im, min_radius=3)

    assert result.groupdq[0, 1, 4, 52] == JUMP_DET


def test_ramp_snowball_centered_on_the_array_border():
    """SnowblindStep.call on a ramp with no reference pixels and a snowball of radius 8 centered on row 0
    flags the row 22 pixels along from its center.
    """
    no_refpix = np.zeros((40, 80), dtype=bool)
    im = ramp_with_event(*round_event((40, 80), (0, 30), 8, 2), no_refpix)

    result = SnowblindStep.call(im, min_radius=3)

    assert result.groupdq[0, 1, 0, 52] == JUMP_DET


def test_ramp_snowball_centered_on_a_corner():
    """SnowblindStep.call on a ramp with a snowball of radius 8 centered on the corner of the light-sensitive area,
    after 4 reference rows and 4 reference columns, flags the first row 22 pixels along from its center.
    """
    refpix = bottom_reference_rows((60, 60))
    refpix[:, :4] = True
    im = ramp_with_event(*round_event((60, 60), (4, 4), 8, 2), refpix)

    result = SnowblindStep.call(im, min_radius=3)

    assert result.groupdq[0, 1, 4, 26] == JUMP_DET


def test_ramp_snowball_with_its_core_past_the_edge():
    """SnowblindStep.call on a ramp with a snowball of radius 20 centered 8 rows into the reference pixels, its core
    of radius 5 out of view, grows the visible part.
    """
    refpix = bottom_reference_rows((60, 80))
    im = ramp_with_event(*round_event((60, 80), (-4, 40), 20, 5), refpix)

    result = SnowblindStep.call(im, min_radius=3)

    assert result.groupdq[0, 1, 20, 40] == JUMP_DET


def test_distance_to_empty_without_an_empty_pixel_counts_from_past_the_edge():
    """SnowblindStep._distance_to_empty on a 5x5 block with no empty pixel gives 3 at its center, the distance to just
    past its edge."""
    distance = SnowblindStep._distance_to_empty(np.ones((5, 5), dtype=bool))

    assert distance[2, 2] == 3


def test_distance_to_farthest_pixel_of_a_block():
    """SnowblindStep._distance_to_farthest_pixel on a 3x5 block gives hypot(2, 4) at its corner pixel (0, 0)."""
    event = np.ones((3, 5), dtype=bool)
    rows, cols = np.nonzero(event)

    farthest = SnowblindStep._distance_to_farthest_pixel(event, rows, cols)

    assert farthest[0] == pytest.approx(np.hypot(2, 4))


def test_distance_to_farthest_pixel_of_a_line():
    """SnowblindStep._distance_to_farthest_pixel on a 1x5 line, which has no convex hull, gives 4 at its end pixel."""
    event = np.ones((1, 5), dtype=bool)
    rows, cols = np.nonzero(event)

    farthest = SnowblindStep._distance_to_farthest_pixel(event, rows, cols)

    assert farthest[0] == 4


def test_events_at_edge_adds_nothing_to_an_event_covering_its_window():
    """SnowblindStep._events_at_edge on a 9x9 region that fills its window, with a 3x3 core at its middle, adds no
    pixels past the window to the event it finds.

    Nothing in view is known to be empty, so the event is taken to end at the window's edge.
    """
    region = np.ones((9, 9), dtype=bool)
    cores = np.zeros((9, 9), dtype=bool)
    cores[3:6, 3:6] = True

    [(event, completed)] = SnowblindStep()._events_at_edge(
        region, (slice(0, 9), slice(0, 9)), cores, np.zeros((9, 9), dtype=bool)
    )

    assert len(completed[0]) == len(event[0])


def test_ramp_small_jump_region_at_the_edge_without_core():
    """SnowblindStep.call on a ramp with a 3 by 7 jump region on the first 3 rows after the reference pixels, and no
    saturated core, leaves it ungrown.

    A disk of min_radius=2 fits only by hanging over the edge, so the region may not do without a core.
    """
    refpix = bottom_reference_rows((40, 40))
    halo = np.zeros((40, 40), dtype=bool)
    halo[4:7, 17:24] = True
    im = ramp_with_event(halo, np.zeros_like(halo), refpix)

    result = SnowblindStep.call(im, min_radius=2)

    assert result.groupdq[0, 1, 8, 20] == GOOD


def test_rate_snowball_centered_on_the_first_light_sensitive_row():
    """SnowblindStep.call on a rate image with a snowball of radius 8 centered on the first row after 4 reference
    rows, its unflagged core open to the reference pixels, flags the row 22 pixels along from its center.
    """
    refpix = bottom_reference_rows((40, 80))
    halo, core = round_event((40, 80), (4, 30), 8, 2)
    im = datamodels.ImageModel((40, 80))
    im.dq = np.zeros((40, 80), dtype=np.uint32)
    im.dq[halo & ~refpix] = JUMP_DET
    im.dq[refpix] = REFERENCE_PIXEL

    result = SnowblindStep.call(im, min_radius=3)

    assert result.dq[4, 52] & JUMP_DET

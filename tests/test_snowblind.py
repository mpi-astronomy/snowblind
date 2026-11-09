import logging
import numpy as np
import pytest
from astropy.io import fits
from stdatamodels.jwst import datamodels

from snowblind import SnowblindStep


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

    # Verify that the saturated core grew for only 2 subsequent groups
    assert result.groupdq[0, 2, 21, 20] == JUMP_DET
    assert result.groupdq[0, 3, 22, 20] == JUMP_DET
    assert result.groupdq[0, 4, 22, 20] == GOOD


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
    """SnowblindStep.dilate_jump_slice with growth_factor=0 on a block with a 200 pixel hole fills the hole."""
    dilated, cores = SnowblindStep(growth_factor=0).dilate_jump_slice(hole_slice(200))

    assert dilated[25, 25]


def test_dilate_jump_slice_keeps_a_hole_of_201_pixels():
    """SnowblindStep.dilate_jump_slice with growth_factor=0 on a block with a 201 pixel hole leaves the hole open.

    A 1 pixel hole elsewhere in the block is its core, as in a rate image, so the block itself is kept.
    """
    jump = hole_slice(201)
    jump[50, 50] = False

    dilated, cores = SnowblindStep(growth_factor=0).dilate_jump_slice(jump)

    assert dilated[40, 40]
    assert not dilated[25, 25]


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


def test_ramp_without_jumps_flags_nothing():
    """SnowblindStep.call on a ramp with no flags in groupdq leaves groupdq all GOOD.

    The ring dilation once set JUMP_DET at pixels (0, 0), (0, 1) and (1, 0) of every group.
    """
    im = datamodels.RampModel((1, 5, 20, 20))

    result = SnowblindStep.call(im)

    assert (result.groupdq == GOOD).all()


def ramp_with_snowball_in_first_integration(nints):
    """A ramp of nints integrations with one snowball in group 2 of the first, its core the pixel [20, 20]."""
    im = datamodels.RampModel((nints, 6, 40, 40))
    im.groupdq[0, 2, 15:26, 15:26] = JUMP_DET
    im.groupdq[0, 2:, 20, 20] = SATURATED
    return im


def test_ramp_core_flagged_in_every_group_of_the_next_integration():
    """SnowblindStep.call on a ramp with a snowball in integration 1 sets JUMP_DET on its core in all groups of integration 2."""
    result = SnowblindStep.call(ramp_with_snowball_in_first_integration(2))

    assert (result.groupdq[1, :, 20, 20] == JUMP_DET).all()


def test_ramp_core_neighbor_not_flagged_in_the_next_integration():
    """SnowblindStep.call on a ramp with a snowball in integration 1 leaves a pixel 3 px from its core GOOD in integration 2."""
    result = SnowblindStep.call(ramp_with_snowball_in_first_integration(2))

    assert (result.groupdq[1, :, 20, 23] == GOOD).all()


def test_ramp_core_not_flagged_two_integrations_later():
    """SnowblindStep.call on a ramp with a snowball in integration 1 leaves its core GOOD in integration 3."""
    result = SnowblindStep.call(ramp_with_snowball_in_first_integration(3))

    assert (result.groupdq[2, :, 20, 20] == GOOD).all()


def test_rate_without_hole():
    """SnowblindStep.call on a rate image with a large jump region and no hole in it leaves it ungrown."""
    im = datamodels.ImageModel((40, 40))
    im.dq = np.zeros((40, 40), dtype=np.uint32)
    im.dq[15:26, 15:26] = JUMP_DET

    result = SnowblindStep.call(im)

    assert result.dq[14, 14] == GOOD


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

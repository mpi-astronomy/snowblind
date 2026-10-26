import numpy as np
import pytest
from astropy.io import fits
from stdatamodels.jwst import datamodels

from snowblind import SnowblindStep


JUMP_DET = datamodels.dqflags.group['JUMP_DET']
GOOD = datamodels.dqflags.group['GOOD']
SATURATED = datamodels.dqflags.group['SATURATED']


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
    """SnowblindStep.call on a ramp with one large event in one group sets EXTNCRS.

    The rate is 1e6 * 1 event / (group_time * (6 groups - 1 first group) * 40 * 40 pixels).
    """
    im = datamodels.RampModel((1, 6, 40, 40))
    im.meta.exposure.group_time = 10.0
    im.groupdq[0, 1, 15:26, 15:26] = JUMP_DET
    im.groupdq[0, 1, 5, 5] = JUMP_DET

    result = SnowblindStep.call(im)

    assert result.meta.exposure.extended_emission_events == pytest.approx(1e6 / (10.0 * 5 * 1600))


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
    result = SnowblindStep(growth_factor=0).dilate_jump_slice(hole_slice(200))

    assert result[25, 25]


def test_dilate_jump_slice_keeps_a_hole_of_201_pixels():
    """SnowblindStep.dilate_jump_slice with growth_factor=0 on a block with a 201 pixel hole leaves the hole open."""
    result = SnowblindStep(growth_factor=0).dilate_jump_slice(hole_slice(201))

    assert not result[25, 25]

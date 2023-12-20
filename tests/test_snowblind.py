import pytest
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
    im2.dq[15:26, 15:26] = JUMP_DET
    im2.dq[20, 20] = SATURATED
    im2.dq[5, 5] = JUMP_DET

    # CubeModel, e.g., _rateints
    im3 = datamodels.CubeModel((6, 40, 40))
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

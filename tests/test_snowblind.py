from stdatamodels.jwst import datamodels

from snowblind import SnowblindStep


JUMP_DET = datamodels.dqflags.group['JUMP_DET']
GOOD = datamodels.dqflags.group['GOOD']
SATURATED = datamodels.dqflags.group['SATURATED']


def test_init():
    snow = SnowblindStep(growth_factor=2)

    assert snow.growth_factor == 2.0


def test_call():
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

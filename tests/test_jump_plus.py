import numpy as np
from stdatamodels.jwst import datamodels

from snowblind import JumpPlusStep


JUMP_DET = datamodels.dqflags.group['JUMP_DET']
GOOD = datamodels.dqflags.group['GOOD']
SATURATED = datamodels.dqflags.group["SATURATED"]


def test_init():
    step = JumpPlusStep()

    assert step.class_alias == "jump_plus"


def test_call():
    im = datamodels.RampModel((1, 3, 40, 40))
    im.meta.exposure.nframes = 8
    im.groupdq[0, 1, 15, 15] = JUMP_DET
    im.groupdq[0, 2, 5, 5] = JUMP_DET
    im.groupdq[0, 2, 20, 20] = SATURATED

    result = JumpPlusStep.call(im)

    # Verify that 2nd slice did not change in :19 area
    np.testing.assert_equal(result.groupdq[0, 1, :19, :19], im.groupdq[0, 1, :19, :19])

    # Verify jumps at group N are now in N+1
    assert result.groupdq[0, 2, 15, 15] == JUMP_DET

    # Verify that for saturated group at N, that N-1 is now jump
    assert result.groupdq[0, 1, 20, 20] == SATURATED

    # Verify that jumps in last slice didn't end up in the first slice
    assert result.groupdq[0, 0, 5, 5] == GOOD

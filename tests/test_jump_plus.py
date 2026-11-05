import numpy as np
import pytest
from stdatamodels.jwst import datamodels

from snowblind import JumpPlusStep

pytestmark = pytest.mark.filterwarnings("ignore:JumpPlusStep is deprecated:DeprecationWarning")

JUMP_DET = datamodels.dqflags.group['JUMP_DET']
GOOD = datamodels.dqflags.group['GOOD']
SATURATED = datamodels.dqflags.group["SATURATED"]


def test_process_warns_deprecated():
    """JumpPlusStep().process on a ramp with nframes=8 raises a DeprecationWarning.

    process is called directly because stpipe sends warnings raised during a run to its log.
    """
    im = datamodels.RampModel((1, 3, 4, 4))
    im.meta.exposure.nframes = 8

    with pytest.warns(DeprecationWarning, match="JumpPlusStep is deprecated"):
        JumpPlusStep().process(im)


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


def test_no_frame_averaging_leaves_skip_unset():
    """JumpPlusStep().run on a ramp with nframes=1 leaves the step's skip False, so a later run is not skipped."""
    step = JumpPlusStep()
    im = datamodels.RampModel((1, 3, 4, 4))
    im.meta.exposure.nframes = 1

    step.run(im)

    assert step.skip is False


def test_no_frame_averaging_records_skipped():
    """JumpPlusStep.call on a ramp with nframes=1 returns meta.cal_step.jump_plus SKIPPED."""
    im = datamodels.RampModel((1, 3, 4, 4))
    im.meta.exposure.nframes = 1

    result = JumpPlusStep.call(im)

    assert result.meta.cal_step.jump_plus == "SKIPPED"

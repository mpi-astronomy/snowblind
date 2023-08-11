import numpy as np
from stdatamodels.jwst import datamodels
from jwst.stpipe import Step


JUMP_DET = datamodels.dqflags.group["JUMP_DET"]


class JumpPlusStep(Step):
    """Updates groupdq by copying jumps in group N to group N+1

    For NIR readout modes that average together more than one frames
    per group, i.e NFRAMES > 1, this propagates the detected JUMPS to
    the successive group, as CR hit within a group effects the slope of
    the one before and the one after.
    """
    spec = ""

    class_alias = "jump_plus"

    def process(self, input_data):
        with datamodels.open(input_data) as jump:
            # If there is more than one frame averaged into a group,
            # then jumps events are to be flagged for 2 frames
            if jump.meta.exposure.nframes > 1:
                frame_averaging = True
            else:
                frame_averaging = False

            result = jump.copy()
        
        if not frame_averaging:
            # Update step meta
            self.log.info("No frame averaging in this readout mode")
            self.skip = True
            setattr(result.meta.cal_step, self.class_alias, "SKIPPED")

            return result

        for integration_dq in result.groupdq:
            # Make jump DQ for the integration where flagged jumps in group N
            # propagate to the group N+1
            after_jump_dq = np.roll(integration_dq & JUMP_DET, axis=0, shift=1)

            # np.roll() wraps the last image slice back to the first.  Restore
            # the original first slice
            after_jump_dq[0] = integration_dq[0] & JUMP_DET

            # Add the new JUMP_DET flags for the integration
            integration_dq |= after_jump_dq
        
        setattr(result.meta.cal_step, self.class_alias, "COMPLETE")

        return result

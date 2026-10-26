import logging
import numpy as np
from jwst import datamodels
from .step import Step

log = logging.getLogger(__name__)


JUMP_DET = datamodels.dqflags.group["JUMP_DET"]
SATURATED = datamodels.dqflags.group["SATURATED"]


class JumpPlusStep(Step):
    """Updates groupdq by propagating jumps in group N to group N+1

    For NIR readout modes that average together more than one frames
    per group, i.e NFRAMES > 1, this propagates the detected JUMPS to
    the successive group, as a cosmic ray arriving within a group affects the slope of
    the one before and potentially the one after.

    Frame-averaged groups that are saturated often got saturated in a
    frame in a previous group, so flag those as jumps as well.
    """
    spec = """
    """

    class_alias = "jump_plus"

    def process(self, input_data):
        """Propagate JUMP_DET flags to the next group and SATURATED flags to the previous one.

        Parameters
        ----------
        input_data : str or RampModel
            Filename or ramp datamodel after jump detection, with a ``groupdq`` array
            of shape (nints, ngroups, nrows, ncols).  When ``meta.exposure.nframes``
            is 1 the step is skipped.

        Returns
        -------
        RampModel
            Copy of the input.  With frame averaging, each group's JUMP_DET flags are
            also set in the following group, each group's SATURATED flags are also set
            in the preceding group, and ``meta.cal_step.jump_plus`` is ``"COMPLETE"``.
            Without frame averaging the copy is unchanged, ``self.skip`` is set and
            ``meta.cal_step.jump_plus`` is ``"SKIPPED"``.
        """
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
            log.info("No frame averaging in this readout mode")
            self.skip = True
            setattr(result.meta.cal_step, self.class_alias, "SKIPPED")

            return result

        for integration_dq in result.groupdq:
            # Make jump DQ for the integration where flagged jumps in group N
            # propagate to the group N+1
            after_jump_dq = np.roll(integration_dq & JUMP_DET, axis=0, shift=1)
            # Restore the original first slice
            after_jump_dq[0] = integration_dq[0] & JUMP_DET

            # Add the new JUMP_DET flags for the integration
            integration_dq |= after_jump_dq

            # Flag saturated groups as jumps in the group before
            before_sat_dq = np.roll(integration_dq & SATURATED, axis=0, shift=-1)
            # Restore the original last slice
            before_sat_dq[-1] = integration_dq[-1] & SATURATED

            # Add the new SATURADED flags for the integration
            integration_dq |= before_sat_dq

        setattr(result.meta.cal_step, self.class_alias, "COMPLETE")

        return result

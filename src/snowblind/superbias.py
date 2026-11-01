"""The jwst superbias step, without turning valid zeroframe values into its "unusable" marker.

The ZEROFRAME has no DQ array.  The saturation step marks a zeroframe pixel
that saturated or sat at the A/D floor by setting it to zero, and every later
step leaves zeros alone: refpix, linearity and the stream clean do not
correct them, and the ramp fit does not use them.  The jwst superbias step
keeps the zeros it is given, but a pixel whose raw zeroframe equals the
superbias exactly also comes out at zero, and the later steps then take it
for unusable.  The raw data are whole DN, so with superbias references that
hold only whole and half-integer values, as recent NIRCam ones do, this can
be up to about 1% of the pixels, more the closer the bias level sits to the
superbias.

`SuperBiasStep` takes the ``superbias`` slot of ``snowblind_detector1``.  It
notes the zeros it is given, runs the jwst step, and gives any other
zeroframe pixel left at zero the value `NOT_ZERO`, which is 0 DN to every
later step but not the marker.
"""
import logging

import numpy as np
from jwst.superbias import SuperBiasStep as JwstSuperBiasStep

from .step import Step

__all__ = ["SuperBiasStep"]

log = logging.getLogger(__name__)

# Not zero, so not the unusable marker, and a normal float32, so no arithmetic flushes it to zero
NOT_ZERO = np.finfo(np.float32).tiny


class SuperBiasStep(Step, JwstSuperBiasStep):
    """The jwst superbias step, keeping the zeroframe values that the subtraction makes exactly zero.

    Only for the ``superbias`` slot of `snowblind.Detector1Pipeline`; it is
    not registered with stpipe.  Its parameters, reference file and CRDS
    parameter reference are those of the jwst step.
    """

    def process(self, step_input):
        """Subtract the superbias, then lift the zeroframe zeros that the subtraction made.

        Parameters
        ----------
        step_input : str or `~stdatamodels.jwst.datamodels.RampModel`
            As for the jwst step.

        Returns
        -------
        `~stdatamodels.jwst.datamodels.RampModel`
            The jwst step's result, with `NOT_ZERO` in the zeroframe pixels
            that are zero after the subtraction but were not zero before it.
        """
        # only read here; the jwst step makes the copy a standalone run needs
        result = self.prepare_output(step_input, make_copy=False, open_as_ramp=True)
        unusable = result.zeroframe == 0 if result.meta.exposure.zero_frame else None
        result = super().process(result)
        if unusable is None or result.meta.cal_step.superbias != "COMPLETE":
            return result
        made = (result.zeroframe == 0) & ~unusable
        result.zeroframe[made] = NOT_ZERO
        log.info(
            f"{made.sum()} zeroframe pixels came out of the subtraction at exactly 0 DN, the pipeline's marker for "
            f"unusable data; they are set to {NOT_ZERO:.3g} DN so that the later steps correct and use them"
        )
        self.set_cal_step(result, "COMPLETE")
        return result

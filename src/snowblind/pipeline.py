"""The Detector1 pipeline with the snowblind steps in place of the pipeline's own.

``calwebb_detector1`` cleans 1/f noise with ``clean_flicker_noise`` and
fits ramps with ``ramp_fit``.  This pipeline keeps every other step and
parameter of the jwst pipeline, including the parameters CRDS holds for
it, and makes two substitutions: `StreamCleanStep` runs as a post-hook of
the jump step with the jwst flicker step skipped, and `ResetRampFitStep`
takes the ramp-fit slot.  It runs as ``strun snowblind_detector1`` or
through `Detector1Pipeline.call`.
"""
import logging

from jwst.pipeline import Detector1Pipeline as JwstDetector1Pipeline

from .reset_rampfit import ResetRampFitStep
from .stream_clean import StreamCleanStep

__all__ = ["Detector1Pipeline"]

log = logging.getLogger(__name__)


class Detector1Pipeline(JwstDetector1Pipeline):
    """``calwebb_detector1`` with the stream clean after the jump step and the reset-anchored ramp fit.

    The step slots and their parameters are the jwst pipeline's (``--steps.jump...``,
    ``--steps.refpix.refpix_algorithm=sirs`` and so on), and so are the
    parameters CRDS provides for ``calwebb_detector1``.  The differences:
    ``jump`` gets `StreamCleanStep` as a post-hook (``--steps.stream_clean``
    is not a slot; its parameters go through the hook, e.g.
    ``--steps.jump.post_hooks="snowblind.StreamCleanStep(background_method='median')",``),
    ``clean_flicker_noise`` is skipped, and ``ramp_fit`` is `ResetRampFitStep`,
    whose parameters are set as ``--steps.ramp_fit.reset_calib=...``.
    Either substitution is switched off with ``--stream_clean=False`` or
    ``--reset_fit=False``, which gives the jwst step back.
    """

    class_alias = "snowblind_detector1"

    spec = JwstDetector1Pipeline.spec + """
    stream_clean = boolean(default=True)  # hook StreamCleanStep after the jump step and skip the jwst flicker step
    reset_fit = boolean(default=True)  # fit ramps with ResetRampFitStep instead of the jwst ramp fit
    """

    step_defs = {**JwstDetector1Pipeline.step_defs, "ramp_fit": ResetRampFitStep}

    def __init__(self, *args, **kwargs):
        """Create the pipeline with the stream clean hooked after jump and the jwst flicker step skipped.

        Parameters
        ----------
        *args
            Passed to the jwst pipeline.
        **kwargs
            Passed to the jwst pipeline.  With ``stream_clean`` (the
            default) the ``jump`` entry of the ``steps`` mapping gets
            `StreamCleanStep` appended to ``post_hooks`` unless one is there
            already, and the ``clean_flicker_noise`` entry gets ``skip``
            set, whatever CRDS or the caller asked for, since the stream
            clean replaces it.  With ``reset_fit`` (the default) the
            ``ramp_fit`` slot holds `ResetRampFitStep`; otherwise the jwst
            ramp fit.
        """
        if not kwargs.get("reset_fit", True):
            jwst_ramp_fit = JwstDetector1Pipeline.step_defs["ramp_fit"]
            self.step_defs = {**self.step_defs, "ramp_fit": jwst_ramp_fit}
            # Step.call builds the ramp_fit section from this class's spec, which is ResetRampFitStep's; the jwst step
            # would reject the parameters it does not know, so they are dropped from the section here
            section = (kwargs.get("steps") or {}).get("ramp_fit")
            if section is not None:
                for key in _parameters_unknown_to(jwst_ramp_fit, section):
                    section.pop(key, None)
        if kwargs.get("stream_clean", True):
            # edited in place: from strun this is a ConfigObj section whose sub-sections stpipe reads as such
            steps = kwargs.get("steps")
            if steps is None:
                steps = kwargs["steps"] = {}
            jump = steps.setdefault("jump", {})
            hooks = jump.get("post_hooks") or []
            hooks = [hooks] if isinstance(hooks, str) else list(hooks)
            if not any(_is_stream_clean(hook) for hook in hooks):
                hooks.append(StreamCleanStep)
            jump["post_hooks"] = hooks
            steps.setdefault("clean_flicker_noise", {})["skip"] = True
        super().__init__(*args, **kwargs)

    @classmethod
    def get_config_reftype(cls):
        """The CRDS parameter reference of the jwst pipeline, so that its instrument settings apply here too.

        Returns
        -------
        str
            ``"pars-detector1pipeline"``.
        """
        return JwstDetector1Pipeline.get_config_reftype()


def _parameters_unknown_to(step_class, section):
    """The keys of a config section that a step class's spec does not define.

    Parameters
    ----------
    step_class : type
        A `stpipe.Step` subclass; its full spec (own parameters, the generic
        step parameters and the ``override_*`` entries of its reference
        types) is loaded with ``load_spec_file``.
    section : mapping
        A config section meant for that step.

    Returns
    -------
    list of str
        The keys of ``section`` the spec does not know, which the step's
        validation would reject.
    """
    known = set(step_class.load_spec_file().keys())
    return [key for key in section if key not in known]


def _is_stream_clean(hook):
    """Whether a post-hook entry already is the stream clean, as a class, an instance or a dotted name.

    Parameters
    ----------
    hook : type or Step or str
        One entry of a ``post_hooks`` list.

    Returns
    -------
    bool
        True for `StreamCleanStep`, an instance of it, or a string naming it.
    """
    if isinstance(hook, str):
        return hook.split("(")[0].strip().endswith("StreamCleanStep")
    if isinstance(hook, type):
        return issubclass(hook, StreamCleanStep)
    return isinstance(hook, StreamCleanStep)

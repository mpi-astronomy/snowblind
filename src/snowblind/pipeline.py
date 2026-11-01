"""The Detector1 pipeline with the snowblind steps in place of the pipeline's own.

``calwebb_detector1`` cleans 1/f noise with ``clean_flicker_noise`` and
fits ramps with ``ramp_fit``.  This pipeline keeps every other step and
parameter of the jwst pipeline, including the parameters CRDS holds for
it, and makes two substitutions: `StreamCleanStep` runs as a post-hook of
the jump step with the jwst flicker step skipped, and `ResetRampFitStep`
takes the ramp-fit slot.  The jump step gets its own defaults for these
two (`JUMP_DEFAULTS`, with `SnowblindStep` and `JumpPlusStep` as its first
post-hooks).  The superbias slot holds `snowblind.superbias.SuperBiasStep`,
the jwst step without the zeroframe zeros its subtraction makes.  It runs
as ``strun snowblind_detector1`` or through `Detector1Pipeline.call`.
"""
import logging

from jwst.pipeline import Detector1Pipeline as JwstDetector1Pipeline

from .jump_plus import JumpPlusStep
from .reset_rampfit import ResetRampFitStep
from .snowblind import SnowblindStep
from .stream_clean import StreamCleanStep
from .superbias import SuperBiasStep

__all__ = ["Detector1Pipeline"]

log = logging.getLogger(__name__)

# The jump step's parameters ahead of the stream clean and the reset fit: at 7 sigma it flags what would disturb
# the stream clean, SnowblindStep and JumpPlusStep take over its snowball expansion and after-jump flagging, and
# the reset fit's own jump search (4.5 sigma) finds the fainter jumps on the cleaned ramp.  They take precedence
# over the CRDS parameter references and give way to the caller's parameters.
JUMP_DEFAULTS = {
    "rejection_threshold": 7.0,
    "three_group_rejection_threshold": 7.0,
    "four_group_rejection_threshold": 7.0,
    "flag_4_neighbors": True,
    "min_jump_to_flag_neighbors": 20.0,
    "after_jump_flag_time1": 0.0,
    "expand_large_events": False,
}

# The jump step's post-hooks in place of its expand_large_events, in their order
SNOWBALL_HOOKS = (
    (SnowblindStep, "snowblind.SnowblindStep(min_radius=3, after_jumps=5)"),
    (JumpPlusStep, "snowblind.JumpPlusStep"),
)


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

    The jump step runs with `JUMP_DEFAULTS` over the CRDS parameters (7
    sigma, no large-event expansion or after-jump flagging), and with
    ``expand_large_events`` off `SnowblindStep` and `JumpPlusStep` are its
    first post-hooks; the caller's ``--steps.jump...`` parameters still
    win, and ``--steps.jump.expand_large_events=True`` gives the jwst
    snowball flagging back instead of the two hooks.

    The ``superbias`` slot holds `snowblind.superbias.SuperBiasStep`, the
    jwst step, which also keeps the zeroframe pixels its subtraction makes
    exactly zero from reading as unusable to the later steps.
    """

    class_alias = "snowblind_detector1"

    spec = JwstDetector1Pipeline.spec + """
    stream_clean = boolean(default=True)  # hook StreamCleanStep after the jump step and skip the jwst flicker step
    reset_fit = boolean(default=True)  # fit ramps with ResetRampFitStep instead of the jwst ramp fit
    """

    step_defs = {**JwstDetector1Pipeline.step_defs, "superbias": SuperBiasStep, "ramp_fit": ResetRampFitStep}

    def __init__(self, *args, **kwargs):
        """Create the pipeline with the stream clean hooked after jump and the jwst flicker step skipped.

        Parameters
        ----------
        *args
            Passed to the jwst pipeline.
        **kwargs
            Passed to the jwst pipeline.  The ``jump`` entry of the
            ``steps`` mapping gets each `JUMP_DEFAULTS` parameter it does
            not hold and, while ``expand_large_events`` is off, a
            `SnowblindStep` and a `JumpPlusStep` in ``post_hooks`` unless
            one is there already, ahead of a stream clean listed there.
            With ``stream_clean`` (the default) it also gets
            `StreamCleanStep` appended to ``post_hooks`` unless one is there
            already, and the ``clean_flicker_noise`` entry gets ``skip``
            set, whatever CRDS or the caller asked for, since the stream
            clean replaces it.  With ``reset_fit`` (the default) the
            ``ramp_fit`` slot holds `ResetRampFitStep`; otherwise the jwst
            ramp fit, and a warning when the jump step is still at the
            snowblind threshold.
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
        # edited in place: from strun this is a ConfigObj section whose sub-sections stpipe reads as such
        steps = kwargs.get("steps")
        if steps is None:
            steps = kwargs["steps"] = {}
        jump = steps.setdefault("jump", {})
        # without CRDS (no input, or its parameter references switched off) the defaults are not in the section yet
        for key, value in JUMP_DEFAULTS.items():
            jump.setdefault(key, value)
        hooks = jump.get("post_hooks") or []
        hooks = [hooks] if isinstance(hooks, str) else list(hooks)
        if not _as_bool(jump["expand_large_events"]):
            for k, (step_class, hook) in enumerate(SNOWBALL_HOOKS):
                if any(_is_hook_of(listed, step_class) for listed in hooks):
                    continue
                later = [later_class for later_class, _ in SNOWBALL_HOOKS[k + 1:]] + [StreamCleanStep]
                at = next(
                    (i for i, listed in enumerate(hooks) if any(_is_hook_of(listed, c) for c in later)), len(hooks)
                )
                hooks.insert(at, hook)
        if kwargs.get("stream_clean", True):
            if not any(_is_hook_of(hook, StreamCleanStep) for hook in hooks):
                hooks.append(StreamCleanStep)
            steps.setdefault("clean_flicker_noise", {})["skip"] = True
        jump["post_hooks"] = hooks
        if not kwargs.get("reset_fit", True) and float(jump["rejection_threshold"]) == JUMP_DEFAULTS["rejection_threshold"]:
            log.warning(
                f"The jump step runs at {JUMP_DEFAULTS['rejection_threshold']:g} sigma, the snowblind default, which "
                "leaves the fainter jumps to the reset fit's own search; the jwst ramp fit may not search for them, "
                "so set --steps.jump.rejection_threshold for it"
            )
        super().__init__(*args, **kwargs)

    @classmethod
    def get_config_from_reference(cls, dataset, disable=None, crds_observatory=None):
        """The CRDS parameters of the jwst pipeline, with `JUMP_DEFAULTS` over those of the jump step.

        stpipe merges the caller's parameters onto this, so they still win
        over the snowblind defaults.

        Parameters
        ----------
        dataset : AbstractDataModel or dict
            The input, or its CRDS parameters.
        disable : bool or None
            Do not retrieve parameters from CRDS; None checks the global
            setting.
        crds_observatory : str or None
            Observatory name, needed when ``dataset`` is a dict.

        Returns
        -------
        configobj.ConfigObj
            The parameters for ``steps`` and the pipeline.
        """
        refcfg = super().get_config_from_reference(dataset, disable=disable, crds_observatory=crds_observatory)
        jump = refcfg.setdefault("steps", {}).setdefault("jump", {})
        for key, value in JUMP_DEFAULTS.items():
            jump[key] = value
        return refcfg

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


def _is_hook_of(hook, step_class):
    """Whether a post-hook entry is a given step, as a class, an instance or a dotted name.

    Parameters
    ----------
    hook : type or Step or str
        One entry of a ``post_hooks`` list.
    step_class : type
        The `stpipe.Step` subclass to look for.

    Returns
    -------
    bool
        True for ``step_class``, a subclass or an instance of it, or a
        string naming it (with or without arguments).
    """
    if isinstance(hook, str):
        return hook.split("(")[0].strip().endswith(step_class.__name__)
    if isinstance(hook, type):
        return issubclass(hook, step_class)
    return isinstance(hook, step_class)


def _as_bool(value):
    """A boolean parameter as Python gives it, or as a string from strun or a config file.

    Parameters
    ----------
    value : bool or str
        The parameter value.

    Returns
    -------
    bool
        The value; a string is true when it reads true, yes, on or 1.
    """
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "on", "1")
    return bool(value)

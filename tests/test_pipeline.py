import logging

from configobj import ConfigObj
from jwst.pipeline import Detector1Pipeline as JwstDetector1Pipeline
from jwst.ramp_fitting import RampFitStep

from snowblind import Detector1Pipeline, ResetRampFitStep, StreamCleanStep

SNOWBLIND_HOOK = "snowblind.SnowblindStep(min_radius=3, after_jumps=5)"
JUMP_PLUS_HOOK = "snowblind.JumpPlusStep"


def crds_jump_threshold_of_4(cls, dataset, disable=None, crds_observatory=None):
    """Stand-in for the jwst pipeline's CRDS parameters: a jump section with rejection_threshold 4."""
    return ConfigObj({"steps": {"jump": {"rejection_threshold": 4.0}}})


def no_crds_parameters(cls, dataset):
    """Stand-in for the CRDS lookup of an input file: no parameters, for jwst."""
    return {}, "jwst"


def test_pipeline_fits_ramps_with_the_reset_step():
    """Detector1Pipeline() has a ResetRampFitStep in its ramp_fit slot."""
    pipeline = Detector1Pipeline()

    assert isinstance(pipeline.ramp_fit, ResetRampFitStep)


def test_pipeline_hooks_the_stream_clean_after_jump_and_skips_the_jwst_flicker_step():
    """Detector1Pipeline() lists StreamCleanStep among the jump step's post hooks and has clean_flicker_noise skipped."""
    pipeline = Detector1Pipeline()

    assert StreamCleanStep in pipeline.jump.post_hooks
    assert pipeline.clean_flicker_noise.skip is True


def test_pipeline_keeps_a_stream_clean_hook_the_caller_gave():
    """Detector1Pipeline(steps=...) with a stream clean hook string keeps it after the snowball hooks, with no second one."""
    hook = "snowblind.StreamCleanStep(background_method='median')"

    pipeline = Detector1Pipeline(steps={"jump": {"post_hooks": [hook]}})

    assert pipeline.jump.post_hooks == [SNOWBLIND_HOOK, JUMP_PLUS_HOOK, hook]


def test_pipeline_uses_the_jwst_pipeline_parameter_reference():
    """Detector1Pipeline.get_config_reftype returns pars-detector1pipeline."""
    assert Detector1Pipeline.get_config_reftype() == "pars-detector1pipeline"


def test_pipeline_without_reset_fit_keeps_the_jwst_ramp_fit():
    """Detector1Pipeline(reset_fit=False) has the jwst RampFitStep in its ramp_fit slot."""

    pipeline = Detector1Pipeline(reset_fit=False)

    assert type(pipeline.ramp_fit) is RampFitStep


def test_pipeline_without_stream_clean_adds_no_stream_clean_hook():
    """Detector1Pipeline(stream_clean=False) with the flicker step on gives jump the snowball hooks alone and the step on."""
    pipeline = Detector1Pipeline(stream_clean=False, steps={"clean_flicker_noise": {"skip": False}})

    assert pipeline.jump.post_hooks == [SNOWBLIND_HOOK, JUMP_PLUS_HOOK]
    assert pipeline.clean_flicker_noise.skip is False


def test_pipeline_accepts_a_single_hook_string():
    """Detector1Pipeline(steps=...) with post_hooks given as one string keeps it whole and adds no second stream clean."""
    hook = "snowblind.StreamCleanStep"

    pipeline = Detector1Pipeline(steps={"jump": {"post_hooks": hook}})

    assert pipeline.jump.post_hooks == [SNOWBLIND_HOOK, JUMP_PLUS_HOOK, hook]


def test_pipeline_without_reset_fit_drops_the_reset_parameters_from_the_ramp_fit_section():
    """Detector1Pipeline(reset_fit=False) with a ramp_fit section holding reset_calib builds a jwst RampFitStep with OLS_C."""
    pipeline = Detector1Pipeline(reset_fit=False, steps={"ramp_fit": {"reset_calib": None, "algorithm": "OLS_C"}})

    assert type(pipeline.ramp_fit) is RampFitStep
    assert pipeline.ramp_fit.algorithm == "OLS_C"


def test_pipeline_runs_the_jump_step_at_7_sigma():
    """Detector1Pipeline() has the jump step's rejection_threshold at 7."""
    pipeline = Detector1Pipeline()

    assert pipeline.jump.rejection_threshold == 7.0


def test_pipeline_switches_off_the_jump_steps_large_event_expansion():
    """Detector1Pipeline() has the jump step's expand_large_events off."""
    pipeline = Detector1Pipeline()

    assert pipeline.jump.expand_large_events is False


def test_pipeline_hooks_snowblind_and_jump_plus_ahead_of_the_stream_clean():
    """Detector1Pipeline() has SnowblindStep(min_radius=3, after_jumps=5), JumpPlusStep, StreamCleanStep as jump post hooks."""
    pipeline = Detector1Pipeline()

    assert pipeline.jump.post_hooks == [SNOWBLIND_HOOK, JUMP_PLUS_HOOK, StreamCleanStep]


def test_pipeline_puts_a_missing_snowblind_hook_ahead_of_the_callers_jump_plus():
    """Detector1Pipeline(steps=...) with post_hooks of JumpPlusStep alone puts the SnowblindStep hook before it."""
    pipeline = Detector1Pipeline(steps={"jump": {"post_hooks": [JUMP_PLUS_HOOK]}})

    assert pipeline.jump.post_hooks == [SNOWBLIND_HOOK, JUMP_PLUS_HOOK, StreamCleanStep]


def test_pipeline_keeps_a_jump_threshold_the_caller_gave():
    """Detector1Pipeline(steps=...) with the jump step's rejection_threshold at 5 keeps 5."""
    pipeline = Detector1Pipeline(steps={"jump": {"rejection_threshold": 5.0}})

    assert pipeline.jump.rejection_threshold == 5.0


def test_pipeline_with_large_event_expansion_adds_no_snowball_hooks():
    """Detector1Pipeline(steps=...) with expand_large_events on gives the jump step the stream clean as its only post hook."""
    pipeline = Detector1Pipeline(steps={"jump": {"expand_large_events": True}})

    assert pipeline.jump.post_hooks == [StreamCleanStep]


def test_config_from_reference_puts_the_jump_defaults_over_crds(monkeypatch):
    """Detector1Pipeline.get_config_from_reference over CRDS parameters with a 4 sigma jump step returns 7 sigma."""
    monkeypatch.setattr(JwstDetector1Pipeline, "get_config_from_reference", classmethod(crds_jump_threshold_of_4))

    config = Detector1Pipeline.get_config_from_reference({}, crds_observatory="jwst")

    assert config["steps"]["jump"]["rejection_threshold"] == 7.0


def test_build_config_keeps_the_callers_jump_threshold_over_the_defaults(monkeypatch):
    """Detector1Pipeline.build_config with CRDS at 4 sigma and the caller's jump threshold of 5 returns 5."""
    monkeypatch.setattr(JwstDetector1Pipeline, "get_config_from_reference", classmethod(crds_jump_threshold_of_4))
    monkeypatch.setattr(Detector1Pipeline, "_get_crds_parameters", classmethod(no_crds_parameters))

    config, _ = Detector1Pipeline.build_config(
        "jw00000000000_00000_00000_nrca1_uncal.fits", steps={"jump": {"rejection_threshold": 5.0}}
    )

    assert config["steps"]["jump"]["rejection_threshold"] == 5.0


def test_pipeline_without_reset_fit_warns_about_the_7_sigma_jump_step(caplog):
    """Detector1Pipeline(reset_fit=False) logs a warning that the jump step is at the snowblind 7 sigma."""
    with caplog.at_level(logging.WARNING, logger="snowblind.pipeline"):
        Detector1Pipeline(reset_fit=False)

    assert any("7 sigma, the snowblind default" in record.getMessage() for record in caplog.records)

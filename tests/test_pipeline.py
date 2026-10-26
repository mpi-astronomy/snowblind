from snowblind import Detector1Pipeline, ResetRampFitStep, StreamCleanStep


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
    """Detector1Pipeline(steps=...) with a configured stream clean hook string keeps it and adds no second one."""
    hook = "snowblind.StreamCleanStep(background_method='median')"

    pipeline = Detector1Pipeline(steps={"jump": {"post_hooks": [hook]}})

    assert pipeline.jump.post_hooks == [hook]


def test_pipeline_uses_the_jwst_pipeline_parameter_reference():
    """Detector1Pipeline.get_config_reftype returns pars-detector1pipeline."""
    assert Detector1Pipeline.get_config_reftype() == "pars-detector1pipeline"

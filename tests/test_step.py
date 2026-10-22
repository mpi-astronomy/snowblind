import logging

from snowblind import JumpPlusStep, OpenPixelStep, PersistenceFlagStep, ResetRampFitStep, SnowblindStep, StreamCleanStep
from snowblind.step import Step


def test_step_names_the_snowblind_logger_after_the_jwst_ones():
    """Step.get_stpipe_loggers returns the jwst step's logger names followed by 'snowblind'."""
    names = Step.get_stpipe_loggers()

    assert names[-1] == "snowblind"
    assert "jwst" in names


def test_every_step_uses_the_shared_base_class():
    """All six snowblind steps are subclasses of snowblind.step.Step."""
    for cls in (SnowblindStep, JumpPlusStep, OpenPixelStep, PersistenceFlagStep, StreamCleanStep, ResetRampFitStep):
        assert issubclass(cls, Step)


def test_module_loggers_are_children_of_the_snowblind_logger():
    """logging.getLogger('snowblind.jump_plus') propagates to the 'snowblind' logger that stpipe is told to configure."""
    child = logging.getLogger("snowblind.jump_plus")

    assert child.parent is logging.getLogger("snowblind")

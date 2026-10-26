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


class LoggingStep(Step):
    """A step whose process logs one line through a snowblind module logger and returns its input."""

    class_alias = "logging_step"
    spec = """
    """

    def process(self, value):
        logging.getLogger("snowblind.logging_step").info("hello from a hook")
        return value


class ListHandler(logging.Handler):
    """A handler that keeps the messages it receives."""

    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def test_run_sends_package_messages_to_the_stpipe_handlers():
    """LoggingStep().run(1), with a handler and INFO level on the 'stpipe' logger only, delivers the step's message there."""
    handler = ListHandler()
    stpipe_logger = logging.getLogger("stpipe")
    previous_level = stpipe_logger.level
    stpipe_logger.addHandler(handler)
    stpipe_logger.setLevel(logging.INFO)
    try:
        LoggingStep().run(1)
    finally:
        stpipe_logger.removeHandler(handler)
        stpipe_logger.setLevel(previous_level)

    assert "hello from a hook" in handler.messages


def test_run_returns_the_borrowed_handlers_afterwards():
    """After LoggingStep().run(1) the 'snowblind' logger no longer holds the handler borrowed from 'stpipe'."""
    handler = ListHandler()
    stpipe_logger = logging.getLogger("stpipe")
    previous_level = stpipe_logger.level
    stpipe_logger.addHandler(handler)
    stpipe_logger.setLevel(logging.INFO)
    try:
        LoggingStep().run(1)
    finally:
        stpipe_logger.removeHandler(handler)
        stpipe_logger.setLevel(previous_level)

    assert handler not in logging.getLogger("snowblind").handlers

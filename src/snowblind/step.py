"""The base class of the snowblind steps.

stpipe attaches its handlers only to the loggers a step names in
``get_stpipe_loggers`` (the jwst step names its own packages).  Naming
``snowblind`` here lets every module logger of this package, obtained with
``logging.getLogger(__name__)``, reach the step log, the step's recorded
log and a pipeline's log file like jwst's own messages.
"""
from jwst.stpipe import Step as JwstStep

__all__ = ["Step"]


class Step(JwstStep):
    """A jwst step whose package loggers are routed into the stpipe log."""

    @classmethod
    def get_stpipe_loggers(cls):
        """The jwst loggers plus this package's."""
        return super().get_stpipe_loggers() + ("snowblind",)

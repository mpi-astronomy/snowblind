"""The base class of the snowblind steps.

stpipe attaches its handlers only to the loggers a step names in
``get_stpipe_loggers`` (the jwst step names its own packages).  Naming
``snowblind`` here lets every module logger of this package, obtained with
``logging.getLogger(__name__)``, reach the step log, the step's recorded
log and a pipeline's log file like jwst's own messages.

That covers a snowblind step run on its own.  Run as a hook inside a jwst
pipeline, the loggers configured are the pipeline's, so `Step.run` also
lends the ``snowblind`` logger the handlers of the ``stpipe`` logger for
the duration of the run.
"""
import logging

from jwst.stpipe import Step as JwstStep

__all__ = ["Step"]

PACKAGE_LOGGER = "snowblind"
PIPELINE_LOGGER = "stpipe"


class Step(JwstStep):
    """A jwst step whose package loggers are routed into the stpipe log."""

    @classmethod
    def get_stpipe_loggers(cls):
        """The jwst loggers plus this package's.

        Returns
        -------
        tuple of str
            The logger names from ``jwst.stpipe.Step.get_stpipe_loggers`` followed
            by ``"snowblind"``.
        """
        return super().get_stpipe_loggers() + (PACKAGE_LOGGER,)

    def run(self, *args):
        """Run the step, with this package's log messages going wherever stpipe's go.

        Inside another package's pipeline only that pipeline's loggers are
        configured, so for the duration of the run the ``snowblind`` logger
        gets the handlers of the ``stpipe`` logger that it does not have
        already, and the same level.  Run on its own, the step has those
        handlers already and nothing changes.

        Parameters
        ----------
        *args
            Passed to ``jwst.stpipe.Step.run``: the step input.

        Returns
        -------
        object
            Whatever ``process`` returns.
        """
        ours = logging.getLogger(PACKAGE_LOGGER)
        theirs = logging.getLogger(PIPELINE_LOGGER)
        borrowed = [handler for handler in theirs.handlers if handler not in ours.handlers]
        previous_level = ours.level
        for handler in borrowed:
            ours.addHandler(handler)
        if borrowed and theirs.level != logging.NOTSET:
            ours.setLevel(theirs.level)
        try:
            return super().run(*args)
        finally:
            for handler in borrowed:
                ours.removeHandler(handler)
            ours.setLevel(previous_level)

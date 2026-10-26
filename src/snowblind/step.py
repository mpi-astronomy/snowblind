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

The jwst schema has no FITS keyword for the status of these steps, so a value set in
``meta.cal_step`` is saved only in the ASDF extension and is missing from the FITS
header.  `Step.set_cal_step` also writes it as an ``S_`` header card through the
model's ``extra_fits`` record, which survives copying, saving and reopening.  It also
adds a HISTORY entry naming the step and the snowblind version, which is kept in both
the FITS header and the ASDF extension.
"""
import logging

from jwst.stpipe import Step as JwstStep

__all__ = ["Step", "carry_cal_step"]

PACKAGE_LOGGER = "snowblind"
PIPELINE_LOGGER = "stpipe"


def _set_card(model, keyword, value, comment):
    """Add or replace a card in the primary header record kept in ``model.extra_fits``.

    Parameters
    ----------
    model : DataModel
        Model to update.
    keyword : str
        FITS keyword, at most 8 characters.
    value : str
        Card value.
    comment : str
        Card comment.
    """
    primary = model._instance.setdefault("extra_fits", {}).setdefault("PRIMARY", {})
    cards = [card for card in primary.get("header", []) if card[0] != keyword]
    cards.append([keyword, value, comment])
    primary["header"] = cards


def _history(model):
    """The descriptions of the history entries of ``model``, as a list of str."""
    return [entry["description"] for entry in model._asdf.get_history_entries()]


def carry_cal_step(source, target):
    """Copy step status records that ``target`` lacks from ``source``.

    A model made from scratch (as the rate products of ramp fitting are) keeps only the
    ``meta.cal_step`` entries its schema knows.  This copies the others, with their
    ``S_`` header cards from ``source.extra_fits`` and the snowblind history entries,
    and leaves entries ``target`` already has alone.

    Parameters
    ----------
    source : DataModel
        Model holding the step records, e.g. the ramp.
    target : DataModel
        Model to add them to.
    """
    for key, value in source.meta.cal_step.instance.items():
        if key not in target.meta.cal_step.instance:
            setattr(target.meta.cal_step, key, value)
    for description in _history(source):
        if description.startswith("snowblind ") and description not in _history(target):
            target.add_history_entry(description)
    cards = source._instance.get("extra_fits", {}).get("PRIMARY", {}).get("header", [])
    present = {card[0] for card in target._instance.get("extra_fits", {}).get("PRIMARY", {}).get("header", [])}
    for keyword, value, comment in cards:
        if keyword.startswith("S_") and keyword not in present:
            _set_card(target, keyword, value, comment)


class Step(JwstStep):
    """A jwst step whose package loggers are routed into the stpipe log.

    Subclasses set ``fits_keyword`` (``S_`` plus up to six characters) and
    ``fits_comment`` for the header card that `set_cal_step` writes.
    """

    fits_keyword = None
    fits_comment = ""

    def set_cal_step(self, model, status):
        """Record the status of this step in ``meta.cal_step``, the FITS header and the history.

        The history entry reads ``snowblind <version> <ClassName>: <status>``; an
        identical entry already in the history is not repeated.

        Parameters
        ----------
        model : DataModel
            Model to update.
        status : str
            ``"COMPLETE"`` or ``"SKIPPED"``.
        """
        setattr(model.meta.cal_step, self.class_alias, status)
        if self.fits_keyword is not None:
            _set_card(model, self.fits_keyword, status, self.fits_comment)
        from . import __version__

        description = f"snowblind {__version__} {type(self).__name__}: {status}"
        if description not in _history(model):
            model.add_history_entry(description)

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

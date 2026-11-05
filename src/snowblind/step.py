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

What a step measures goes in ``meta.snowblind.<class_alias>``, under the names
defined once for each step in `META_SCHEMA`, through `Step.record`; an entry with a
``fits_keyword`` is also written to that header card through ``extra_fits``.  Both
are kept in the ASDF extension, so they survive copying, saving and reopening, by
jwst steps as well.  The schema is not added to the models: the jwst steps open and
save them with their own schemas, which would leave out a card written from it, and a
card both in the schema and in ``extra_fits`` is written twice.
"""
import logging

from jwst.stpipe import Step as JwstStep

__all__ = ["Step", "META_SCHEMA"]

PACKAGE_LOGGER = "snowblind"
PIPELINE_LOGGER = "stpipe"

# The schema of ``meta.snowblind``: one object per step, by ``class_alias``, with what
# the step measures.  An entry with a ``fits_keyword`` is also written to that card of
# the primary header, with its ``title`` (at most 47 characters) as the comment, or its
# ``fits_comment`` when it has one (empty for a long value such as a file name).
META_SCHEMA = {
    "type": "object",
    "properties": {
        "persist": {
            "type": "object",
            "properties": {
                "time": {
                    "type": "number",
                    "title": "persist: flagging time [s]",
                    "description": (
                        "Pixels that saturated in an earlier exposure of the detector which ended less than this many "
                        "seconds before the middle of this one are flagged."
                    ),
                    "fits_keyword": "PERSTIME",
                },
                "time_origin": {
                    "type": "string",
                    "title": "Where the time came from",
                    "description": (
                        "given: the step's time parameter; packaged: the time measured for the detector and packaged "
                        "with snowblind; default: 2500 s, for a detector without a packaged time."
                    ),
                },
                "flagged_pixels": {"type": "integer", "title": "Pixels flagged in this exposure"},
                "earlier_exposures": {"type": "integer", "title": "Earlier exposures the flags came from"},
            },
        },
        "reset_rampfit": {
            "type": "object",
            "properties": {
                "unit": {"type": "string", "title": "Unit of reset_offset and reset_level_scatter"},
                "reset_level_scatter": {
                    "type": "number",
                    "title": "Scatter of the reset level the fit assumed (median over pixels)",
                },
                "reset_offset": {
                    "type": "array",
                    "items": {"type": "number"},
                    "title": "Offset of the reset level over the prior, per output channel (or one for all)",
                    "description": (
                        "Added to the prior of the reset level (the detector's mean pedestal map, or zero) in each "
                        "output channel."
                    ),
                },
                "reset_offset_origin": {
                    "type": "string",
                    "title": "Where reset_offset came from",
                    "description": (
                        "sky: anchored to this exposure's sky pixels, keeping the prior's value in a channel "
                        "with too few; prior: the reset calibration and the step's reset_offset."
                    ),
                },
                "first_read_time_scale": {
                    "type": "array",
                    "items": {"type": "number"},
                    "title": "Effective time of the first read over its nominal, per output channel (or one for all)",
                },
                "first_read_time_origin": {
                    "type": "string",
                    "title": "Where first_read_time_scale came from",
                    "description": (
                        "stars: fitted to this exposure's bright pixels, keeping the prior's value in a channel "
                        "with too few; prior: the reset calibration or the step's t1_scale."
                    ),
                },
            },
        },
        "stream_clean": {
            "type": "object",
            "properties": {
                "unit": {"type": "string", "title": "Unit of the sky numbers"},
                "sky_mean_minus_median": {
                    "type": "number",
                    "title": "stream_clean sky mean minus median [DN/s]",
                    "description": (
                        "Masked mean minus masked median of the background pixels of the flat-fielded group "
                        "differences, averaged over the differences and integrations: the shift a scalar median "
                        "background imposes on the sky."
                    ),
                    "fits_keyword": "SKYMNMED",
                },
                "sky_change_by_cleaning": {
                    "type": "number",
                    "title": "stream_clean sky change by cleaning [DN/s]",
                    "description": (
                        "Mean of the cleaned minus the input group differences over the background pixels, "
                        "flat-fielded, averaged over the differences and integrations; negative when the cleaning "
                        "lowered the sky."
                    ),
                    "fits_keyword": "SKYCHG",
                },
                "sky_mean_minus_median_per_integration": {
                    "type": "array",
                    "items": {"type": "number"},
                    "title": "sky_mean_minus_median of each integration [DN/s]",
                },
                "sky_change_by_cleaning_per_integration": {
                    "type": "array",
                    "items": {"type": "number"},
                    "title": "sky_change_by_cleaning of each integration [DN/s]",
                },
            },
        },
        "sparse_resample": {
            "type": "object",
            "properties": {
                "variance": {
                    "type": "string",
                    "title": "Variance planes: exact or stcal",
                    "description": (
                        "exact: the variance of each output pixel, without the correlation between neighboring "
                        "output pixels, which the noise factor gives; ERR is the root of their sum times the noise "
                        "factor. stcal: the variance planes and ERR of the jwst resample step."
                    ),
                    "fits_keyword": "VARTYPE",
                },
                "noise_factor_file": {
                    "type": "string",
                    "title": "File of the noise factor of the variance",
                    "description": (
                        "The noise factor of each output pixel, about the variance of a sum over a large aperture "
                        "over the sum of the pixel variances in it, saved next to the mosaic."
                    ),
                    "fits_keyword": "NFACFILE",
                    # a product file name fills most of the card
                    "fits_comment": "",
                },
            },
        },
    },
}


def _set_card(model, keyword, value, comment):
    """Add or replace a card in the primary header record kept in ``model.extra_fits``.

    Parameters
    ----------
    model : DataModel
        Model to update.
    keyword : str
        FITS keyword, at most 8 characters.
    value : str or float
        Card value.
    comment : str
        Card comment.
    """
    primary = model._instance.setdefault("extra_fits", {}).setdefault("PRIMARY", {})
    cards = [card for card in primary.get("header", []) if card[0] != keyword]
    cards.append([keyword, value, comment])
    primary["header"] = cards


def _history(model):
    """The descriptions of the history entries of ``model``.

    Parameters
    ----------
    model : DataModel
        Model whose ASDF history is read.

    Returns
    -------
    list of str
        One description per history entry, in order.
    """
    return [entry["description"] for entry in model._asdf.get_history_entries()]


class Step(JwstStep):
    """A jwst step whose package loggers are routed into the stpipe log.

    Subclasses set ``fits_keyword`` (``S_`` plus up to six characters) and
    ``fits_comment`` for the header card that `set_cal_step` writes.
    """

    fits_keyword = None
    fits_comment = ""

    def record(self, model, **values):
        """Record what this step measured in ``meta.snowblind.<class_alias>``, and in its header cards.

        Parameters
        ----------
        model : DataModel
            Model to update.
        **values
            Each measurement by its name in `META_SCHEMA` for this step: a string,
            a number or a list of numbers.  One with a ``fits_keyword`` there is
            also written to that card of the primary header, through
            ``model.extra_fits``, with the entry's ``fits_comment``, or else its
            ``title``, as the comment.

        Raises
        ------
        KeyError
            When a name is not in `META_SCHEMA` for this step.
        """
        entries = META_SCHEMA["properties"][self.class_alias]["properties"]
        block = model.meta.instance.setdefault("snowblind", {}).setdefault(self.class_alias, {})
        for name, value in values.items():
            entry = entries[name]
            block[name] = value
            if "fits_keyword" in entry:
                _set_card(model, entry["fits_keyword"], value, entry.get("fits_comment", entry["title"]))

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

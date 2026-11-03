import logging

import pytest
from astropy.io import fits
from stdatamodels.jwst import datamodels

from snowblind import __version__
from snowblind import (
    JumpPlusStep, OpenPixelStep, PersistenceFlagStep, ResetRampFitStep, SnowblindStep, SparseOutlierDetectionStep,
    SparseResampleStep, StreamCleanStep, TiledOutlierStep,
)
from snowblind.step import META_SCHEMA, Step, _history


def test_step_names_the_snowblind_logger_after_the_jwst_ones():
    """Step.get_stpipe_loggers returns the jwst step's logger names followed by 'snowblind'."""
    names = Step.get_stpipe_loggers()

    assert names[-1] == "snowblind"
    assert "jwst" in names


def test_every_step_uses_the_shared_base_class():
    """All nine snowblind steps are subclasses of snowblind.step.Step."""
    for cls in (
        SnowblindStep, JumpPlusStep, OpenPixelStep, PersistenceFlagStep, StreamCleanStep, ResetRampFitStep,
        SparseResampleStep, SparseOutlierDetectionStep, TiledOutlierStep,
    ):
        assert issubclass(cls, Step)


def test_module_loggers_are_children_of_the_snowblind_logger():
    """logging.getLogger('snowblind.jump_plus') propagates to the 'snowblind' logger that stpipe is told to configure."""
    parent = logging.getLogger("snowblind")
    child = logging.getLogger("snowblind.jump_plus")

    assert child.parent is parent


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


STEPS = [SnowblindStep, JumpPlusStep, OpenPixelStep, PersistenceFlagStep, StreamCleanStep, ResetRampFitStep]


@pytest.mark.parametrize("cls", STEPS)
def test_fits_card_is_a_plain_card_with_whole_comment(cls):
    """The card from each step's fits_keyword and fits_comment with value COMPLETE is not HIERARCH and keeps its full comment."""
    card = fits.Card(cls.fits_keyword, "COMPLETE", cls.fits_comment)

    assert not str(card).startswith("HIERARCH")
    assert card.comment == cls.fits_comment
    assert str(card).rstrip().endswith(cls.fits_comment)


def test_fits_keywords_are_distinct():
    """The six steps have six different fits_keyword values."""
    assert len({cls.fits_keyword for cls in STEPS}) == 6


def test_set_cal_step_adds_version_history_entry():
    """SnowblindStep.set_cal_step with COMPLETE adds one history entry with the snowblind version, even when called twice."""
    model = datamodels.ImageModel((4, 4))
    step = SnowblindStep()
    step.set_cal_step(model, "COMPLETE")
    step.set_cal_step(model, "COMPLETE")

    assert _history(model) == [f"snowblind {__version__} SnowblindStep: COMPLETE"]


def test_history_entry_is_in_fits_header_and_survives_reopen(tmp_path):
    """Saving a model after SnowblindStep.set_cal_step puts the entry in the FITS HISTORY cards and the reopened history."""
    model = datamodels.ImageModel((4, 4))
    SnowblindStep().set_cal_step(model, "COMPLETE")
    model.save(tmp_path / "out.fits")
    expected = f"snowblind {__version__} SnowblindStep: COMPLETE"

    assert expected in list(fits.getheader(tmp_path / "out.fits")["HISTORY"])
    with datamodels.open(tmp_path / "out.fits") as reopened:
        assert _history(reopened) == [expected]


def primary_cards(model):
    """The primary header cards of a model's extra_fits record, by keyword."""
    return {card[0]: card[1] for card in model._instance.get("extra_fits", {}).get("PRIMARY", {}).get("header", [])}


def schema_keywords():
    """The fits_keyword of every entry in META_SCHEMA."""
    return [
        entry["fits_keyword"]
        for step in META_SCHEMA["properties"].values()
        for entry in step["properties"].values()
        if "fits_keyword" in entry
    ]


def test_record_writes_the_value_under_meta_snowblind():
    """StreamCleanStep.record with sky_mean_minus_median=0.5 sets meta.snowblind.stream_clean.sky_mean_minus_median to 0.5."""
    model = datamodels.RampModel((1, 3, 4, 4))
    StreamCleanStep().record(model, sky_mean_minus_median=0.5)

    assert model.meta.snowblind.stream_clean.sky_mean_minus_median == 0.5


def test_record_writes_the_header_card_of_the_schema():
    """StreamCleanStep.record with sky_mean_minus_median=0.5 adds the card SKYMNMED = 0.5 to the primary header record."""
    model = datamodels.RampModel((1, 3, 4, 4))
    StreamCleanStep().record(model, sky_mean_minus_median=0.5)

    assert primary_cards(model)["SKYMNMED"] == 0.5


def test_record_without_a_fits_keyword_writes_no_card():
    """StreamCleanStep.record with unit='DN/s', which has no fits_keyword in META_SCHEMA, adds no header card."""
    model = datamodels.RampModel((1, 3, 4, 4))
    StreamCleanStep().record(model, unit="DN/s")

    assert primary_cards(model) == {}


def test_record_of_a_name_not_in_the_schema_raises():
    """StreamCleanStep.record with a name META_SCHEMA does not define for stream_clean raises KeyError."""
    model = datamodels.RampModel((1, 3, 4, 4))

    with pytest.raises(KeyError):
        StreamCleanStep().record(model, sky_level=0.5)


def test_recorded_card_survives_a_reopen_and_save_by_jwst(tmp_path):
    """A model saved after StreamCleanStep.record, reopened with datamodels.open and saved again, has SKYMNMED in its header."""
    model = datamodels.RampModel((1, 3, 4, 4))
    StreamCleanStep().record(model, sky_mean_minus_median=0.5)
    model.save(tmp_path / "first.fits")
    with datamodels.open(tmp_path / "first.fits") as reopened:
        reopened.save(tmp_path / "second.fits")

    assert fits.getheader(tmp_path / "second.fits")["SKYMNMED"] == 0.5


def test_recorded_value_survives_a_reopen_and_save_by_jwst(tmp_path):
    """A model saved after StreamCleanStep.record, reopened with datamodels.open and saved again, keeps the value in meta."""
    model = datamodels.RampModel((1, 3, 4, 4))
    StreamCleanStep().record(model, sky_mean_minus_median=0.5)
    model.save(tmp_path / "first.fits")
    with datamodels.open(tmp_path / "first.fits") as reopened:
        reopened.save(tmp_path / "second.fits")

    with datamodels.open(tmp_path / "second.fits") as again:
        assert again.meta.snowblind.stream_clean.sky_mean_minus_median == 0.5


@pytest.mark.parametrize("keyword", schema_keywords())
def test_schema_keyword_is_not_a_jwst_keyword(keyword):
    """ImageModel.find_fits_keyword on each fits_keyword of META_SCHEMA finds nothing in the jwst schema."""
    assert datamodels.ImageModel((4, 4)).find_fits_keyword(keyword) == []

import logging
import os
from collections import defaultdict, deque
from importlib.resources import files

import asdf
from astropy.io import fits
import numpy as np
from jwst import datamodels
from jwst.associations import load_asn
from stdatamodels.jwst.datamodels import read_metadata

from .step import Step

log = logging.getLogger(__name__)


DO_NOT_USE = datamodels.dqflags.pixel["DO_NOT_USE"]
SATURATED = datamodels.dqflags.pixel["SATURATED"]
PERSISTENCE = datamodels.dqflags.pixel["PERSISTENCE"]
SECONDS_PER_DAY = 86400.0
DEFAULT_TIME = 2500.0  # seconds, for a detector without a packaged time
PACKAGED_TIME_PATTERN = "persistence_{detector}.asdf"
# the folder of the packaged times; tests point it elsewhere
PACKAGED_DIR = files("snowblind").joinpath("data")


class PersistenceFlagStep(Step):
    """
    Given a series of exposures, for any pixel flagged as SATURATED in one exposure,
    flag the pixel as DO_NOT_USE | PERSISTENCE in the later exposures of the same
    detector whose middle is less than ``time`` seconds after the end of that one.

    Persistence decays from the end of the saturated exposure, and the rate of a later
    exposure averages it over the exposure, which its value at the middle of the
    exposure approximates; so an exposure longer than ``time`` still flags the next.

    Without ``time``, each detector gets the time measured for it and packaged with
    snowblind (`packaged_time`), and `DEFAULT_TIME` (2500 s) when it has none.  The
    packaged times are long on purpose: persistence varies across a detector and with
    each pixel's history of illumination (how long and how often it saturated), so they
    are set for a pixel persisting twice as much as the detector's typical saturated
    star.  Each file's ``typical_time`` is the time of the typical star; pass it, or a
    time between the two, as ``time`` to flag less.
    """

    spec = """
        time = float(default=None)  # seconds, end of saturated exposure to middle of later one; None for the packaged time
        save_mask = boolean(default=False)  # write out persistence mask for each exposure
        output_use_model = boolean(default=True)
        output_use_index = boolean(default=False)
    """

    class_alias = "persist"
    # S_PERSIS is the jwst schema's keyword for its own persistence step
    fits_keyword = "S_PERSFL"
    fits_comment = "Persistence: snowblind.PersistenceFlagStep"

    def process(self, input_data):
        """Flag persistence from earlier saturated exposures in each image.

        The saturated pixels of an exposure are those with SATURATED in its ``dq``:
        ramp fitting sets it where any group of any integration saturated.

        Parameters
        ----------
        input_data : str, list of str, or ModelContainer
            Association file, list of file names, or container of image datamodels
            (``_rate`` or ``_cal`` products) with ``dq`` arrays of shape (nrows, ncols).
            Of an association, the members of its first product are read, whatever
            their exposure type, as `~jwst.datamodels.ModelContainer` reads them.
            The exposures are grouped by ``meta.instrument.detector`` and each group
            is handled on its own, in order of ``meta.exposure.start_time``, with
            ``meta.exposure.end_time`` giving the end and the middle of each; the
            ``meta.subarray`` start pixels place each exposure on the detector, so
            a subarray is flagged where a full-frame exposure saturated, and the
            other way round.  When file names are given and ``save_results`` is set,
            each exposure is opened, flagged, saved and closed in turn, so only the
            saturation masks of the last ``time`` seconds are held in memory;
            each is saved under its own file name with the step's suffix, whatever
            ``output_use_model`` and ``output_file`` say.

        Returns
        -------
        ModelContainer or list of str
            Copy of the input in which pixels saturated in an earlier exposure of the
            same detector, which ended less than ``time`` seconds before the middle
            of the exposure, have ``DO_NOT_USE | PERSISTENCE`` set in ``dq``; for file names with
            ``save_results`` set, the paths of the saved files instead.
        """
        paths = member_paths(input_data, self.make_input_path)
        if paths is not None and self.save_results:
            written = []
            exposures = []
            for path in paths:
                meta = read_metadata(path)
                exposures.append((
                    meta["meta.instrument.detector"], meta["meta.exposure.start_time"],
                    meta["meta.exposure.end_time"], path,
                ))

            def save(model):
                # stpipe finalizes and saves the result after process returns; these are saved one at a time here,
                # each under its own name, which one output_file cannot give several exposures
                self.finalize_result(model, [])
                written.append(self.save_model(model, output_file=model.meta.filename))
                model.close()

            self.flag_in_time_order(exposures, datamodels.open, save)
            return written

        if paths is not None and isinstance(input_data, (list, tuple)):
            # with bare names put in input_dir
            input_data = paths
        results = self.prepare_output(input_data)
        exposures = [
            (m.meta.instrument.detector, m.meta.exposure.start_time, m.meta.exposure.end_time, m) for m in results
        ]
        self.flag_in_time_order(exposures, lambda model: model, lambda model: None)
        return results

    def time_for(self, detector):
        """The time within which a detector's later exposures are flagged.

        Parameters
        ----------
        detector : str
            Detector name as in ``meta.instrument.detector``.

        Returns
        -------
        float
            ``self.time`` when it is set; otherwise the detector's `packaged_time`,
            or `DEFAULT_TIME` when there is none.  Seconds, from the end of the
            saturated exposure to the middle of the later one.
        """
        if self.time is not None:
            return self.time
        packaged = packaged_time(detector)
        if packaged is None:
            log.info(f"No persistence time is packaged for {detector}; flagging for {DEFAULT_TIME:g} s")
            return DEFAULT_TIME
        log.info(f"Flagging {detector} for {packaged:g} s, the persistence time measured for the detector")
        return packaged

    def flag_in_time_order(self, exposures, load, done):
        """Flag each exposure from the saturated pixels of the earlier ones, one detector at a time.

        Parameters
        ----------
        exposures : list of (str, float, float, object)
            Detector name, start and end time (MJD) and an item that ``load`` turns
            into the image model, for each exposure.
        load : callable
            Takes an item and returns its datamodel.
        done : callable
            Takes the datamodel once its flags are set.
        """
        by_detector = defaultdict(list)
        for detector, start, end, item in exposures:
            by_detector[detector].append((start, end, item))

        for detector, sequence in by_detector.items():
            limit = self.time_for(detector)
            sequence.sort(key=lambda entry: entry[0])
            starts = np.array([start for start, _, _ in sequence])
            ends = np.array([end for _, end, _ in sequence])
            log.info(
                f"Delays [sec] from the end of each {detector} exposure to the middle of the next are "
                f"{np.round(((starts[1:] + ends[1:]) / 2 - ends[:-1]) * SECONDS_PER_DAY, 1).tolist()}"
            )

            # start and end time [sec], file name, detector origin and saturation mask of the earlier exposures that
            # ended less than limit before the middle of the current one
            recent = deque()
            for start, end, item in sequence:
                seconds = start * SECONDS_PER_DAY
                middle = (start + end) / 2 * SECONDS_PER_DAY
                while recent and middle - recent[0][1] >= limit:
                    recent.popleft()

                model = load(item)
                origin = detector_origin(model)
                saturated = (model.dq & SATURATED) > 0
                mask = np.zeros(model.dq.shape, dtype=bool)
                used = 0
                for earlier_seconds, _, earlier_name, earlier_origin, earlier in recent:
                    if earlier_seconds == seconds:
                        # a detector takes one exposure at a time, so this is the same exposure given twice
                        log.warning(
                            f"{model.meta.filename} starts at the same time as {earlier_name} on {detector}, so it is "
                            "taken for the same exposure and not flagged from it"
                        )
                        continue
                    or_overlap(mask, origin, earlier, earlier_origin)
                    used += 1

                log.info(f"Pixels flagged: {model.meta.filename} {mask.sum()} from {used} earlier exposures")
                model.dq |= (mask * (DO_NOT_USE | PERSISTENCE)).astype(model.dq.dtype)
                self.set_cal_step(model, "COMPLETE")
                if self.save_mask:
                    mask_name = self.make_output_path(basepath=model.meta.filename, suffix="persistmask")
                    fits.PrimaryHDU(data=mask.astype(np.uint8)).writeto(mask_name, overwrite=True)
                    log.info(f"Writing out persistence mask {mask_name}")

                recent.append((seconds, end * SECONDS_PER_DAY, model.meta.filename, origin, saturated))
                done(model)


def packaged_time(detector):
    """The persistence time measured for a detector and packaged with snowblind.

    ``snowblind/data/persistence_<detector>.asdf`` holds ``time``: the delay,
    in seconds from the end of a saturated exposure to the middle of a later
    one, at which a pixel persisting ``margin`` (2) times more than the
    detector's typical saturated star falls to the file's ``threshold``
    (DN/s); ``typical_time``, the delay at which the typical star itself
    does; and the fit and the measurement behind them.  ``scripts/measure_persistence.py``
    measures the persistence in a series of exposures, and
    ``scripts/package_persistence_defaults.py`` fits it and writes the files.

    Parameters
    ----------
    detector : str
        Detector name as in ``meta.instrument.detector``; case does not matter.

    Returns
    -------
    float or None
        The time in seconds; None when the package has no file for the detector.
    """
    path = PACKAGED_DIR.joinpath(PACKAGED_TIME_PATTERN.format(detector=str(detector).lower()))
    if not path.is_file():
        return None
    with asdf.open(str(path), lazy_load=False, memmap=False) as handle:
        return float(handle.tree["time"])


def detector_origin(model):
    """The detector pixel of the first pixel of an image, from its subarray start.

    Parameters
    ----------
    model : DataModel
        Image model; ``meta.subarray.ystart`` and ``xstart`` (1-based, in the
        frame of the data) place it on the detector, and are taken as 1 when
        not set.

    Returns
    -------
    tuple of int
        (row, column) on the detector, counted from 0, of ``model.dq[0, 0]``.
    """
    return (model.meta.subarray.ystart or 1) - 1, (model.meta.subarray.xstart or 1) - 1


def or_overlap(mask, origin, other, other_origin):
    """Set in a mask the pixels set in the mask of another exposure of the detector, where the two overlap.

    Parameters
    ----------
    mask : ndarray of bool, shape (nrows, ncols)
        Mask to update in place.
    origin : tuple of int
        Detector (row, column) of ``mask[0, 0]``.
    other : ndarray of bool
        The other exposure's mask, of any shape.
    other_origin : tuple of int
        Detector (row, column) of ``other[0, 0]``.
    """
    low = np.maximum(origin, other_origin)
    high = np.minimum(np.add(origin, mask.shape), np.add(other_origin, other.shape))
    if np.any(high <= low):
        return
    here = tuple(slice(a - o, b - o) for a, b, o in zip(low, high, origin))
    there = tuple(slice(a - o, b - o) for a, b, o in zip(low, high, other_origin))
    mask[here] |= other[there]


def member_paths(input_data, input_path=lambda path: path):
    """The file names in an association file, file name or list of file names.

    Parameters
    ----------
    input_data : object
        Step input.
    input_path : callable
        Takes each file name of a list and returns its path; the step's
        ``make_input_path``, which puts a bare name in ``input_dir``.

    Returns
    -------
    list of str or None
        The members of the association's first product (relative to the
        association's directory) whatever their exposure type, as
        `~jwst.datamodels.ModelContainer` reads them; the file name; or the list
        of file names; None when the input holds datamodels.
    """
    if isinstance(input_data, (str, os.PathLike)):
        path = os.fspath(input_data)
        if not path.endswith(".json"):
            return [path]
        with open(path) as asn_file:
            asn = load_asn(asn_file)
        folder = os.path.dirname(path)
        return [os.path.join(folder, member["expname"]) for member in asn["products"][0]["members"]]
    if isinstance(input_data, (list, tuple)) and all(isinstance(i, (str, os.PathLike)) for i in input_data):
        return [input_path(os.fspath(i)) for i in input_data]
    return None

import logging
import os
from collections import defaultdict, deque

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


class PersistenceFlagStep(Step):
    """
    Given a series of exposures, for any pixel flagged as SATURATED in one exposure,
    flag the pixel as DO_NOT_USE | PERSISTENCE in the later exposures of the same
    detector that start less than ``time`` seconds after it, start time to start time.
    """

    spec = """
        time = float(default=2500.0)  # amount of time after exposure to flag [seconds]
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
            is handled on its own, in order of ``meta.exposure.start_time``; the
            ``meta.subarray`` start pixels place each exposure on the detector, so
            a subarray is flagged where a full-frame exposure saturated, and the
            other way round.  When file names are given and ``save_results`` is set,
            each exposure is opened, flagged, saved and closed in turn, so only the
            saturation masks of the last ``self.time`` seconds are held in memory;
            each is saved under its own file name with the step's suffix, whatever
            ``output_use_model`` and ``output_file`` say.

        Returns
        -------
        ModelContainer or list of str
            Copy of the input in which pixels saturated in an earlier exposure of the
            same detector, started less than ``self.time`` seconds before, have
            ``DO_NOT_USE | PERSISTENCE`` set in ``dq``; for file names with
            ``save_results`` set, the paths of the saved files instead.
        """
        paths = member_paths(input_data, self.make_input_path)
        if paths is not None and self.save_results:
            written = []
            exposures = []
            for path in paths:
                meta = read_metadata(path)
                exposures.append((meta["meta.instrument.detector"], meta["meta.exposure.start_time"], path))

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
        exposures = [(m.meta.instrument.detector, m.meta.exposure.start_time, m) for m in results]
        self.flag_in_time_order(exposures, lambda model: model, lambda model: None)
        return results

    def flag_in_time_order(self, exposures, load, done):
        """Flag each exposure from the saturated pixels of the earlier ones, one detector at a time.

        Parameters
        ----------
        exposures : list of (str, float, object)
            Detector name, start time (MJD) and an item that ``load`` turns into the
            image model, for each exposure.
        load : callable
            Takes an item and returns its datamodel.
        done : callable
            Takes the datamodel once its flags are set.
        """
        by_detector = defaultdict(list)
        for detector, start, item in exposures:
            by_detector[detector].append((start, item))

        for detector, sequence in by_detector.items():
            sequence.sort(key=lambda entry: entry[0])
            starts = np.array([start for start, _ in sequence])
            log.info(
                f"Time deltas [sec] between {detector} exposures are "
                f"{np.round(np.diff(starts, prepend=starts[0]) * SECONDS_PER_DAY, 1).tolist()}"
            )

            # start time [sec], file name, detector origin and saturation mask of the earlier exposures less than
            # self.time before
            recent = deque()
            for start, item in sequence:
                seconds = start * SECONDS_PER_DAY
                while recent and seconds - recent[0][0] >= self.time:
                    recent.popleft()

                model = load(item)
                origin = detector_origin(model)
                saturated = (model.dq & SATURATED) > 0
                mask = np.zeros(model.dq.shape, dtype=bool)
                used = 0
                for earlier_seconds, earlier_name, earlier_origin, earlier in recent:
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

                recent.append((seconds, model.meta.filename, origin, saturated))
                done(model)


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

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
            The exposures are grouped by ``meta.instrument.detector`` and each group
            is handled on its own, in order of ``meta.exposure.start_time``.  When
            file names are given and ``save_results`` is set, each exposure is
            opened, flagged, saved and closed in turn, so only the saturation masks
            of the last ``self.time`` seconds are held in memory.

        Returns
        -------
        ModelContainer or list of str
            Copy of the input in which pixels saturated in an earlier exposure of the
            same detector, started less than ``self.time`` seconds before, have
            ``DO_NOT_USE | PERSISTENCE`` set in ``dq``; for file names with
            ``save_results`` set, the paths of the saved files instead.
        """
        paths = member_paths(input_data)
        if paths is not None and self.save_results:
            written = []
            exposures = []
            for path in paths:
                meta = read_metadata(path)
                exposures.append((meta["meta.instrument.detector"], meta["meta.exposure.start_time"], path))

            def save(model):
                # stpipe finalizes and saves the result after process returns; these are saved one at a time here
                self.finalize_result(model, [])
                written.append(self.save_model(model))
                model.close()

            self.flag_in_time_order(exposures, datamodels.open, save)
            return written

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

            # start time [sec] and saturation mask of the earlier exposures less than self.time before
            recent = deque()
            for start, item in sequence:
                seconds = start * SECONDS_PER_DAY
                while recent and seconds - recent[0][0] >= self.time:
                    recent.popleft()

                model = load(item)
                saturated = (model.dq & SATURATED) > 0
                mask = np.zeros(model.dq.shape, dtype=bool)
                for _, earlier in recent:
                    mask |= earlier

                log.info(f"Pixels flagged: {model.meta.filename} {mask.sum()} from {len(recent)} earlier exposures")
                model.dq |= (mask * (DO_NOT_USE | PERSISTENCE)).astype(model.dq.dtype)
                self.set_cal_step(model, "COMPLETE")
                if self.save_mask:
                    mask_name = self.make_output_path(basepath=model.meta.filename, suffix="persistmask")
                    fits.PrimaryHDU(data=mask.astype(np.uint8)).writeto(mask_name, overwrite=True)
                    log.info(f"Writing out persistence mask {mask_name}")

                recent.append((seconds, saturated))
                done(model)


def member_paths(input_data):
    """The file names in an association file, file name or list of file names.

    Parameters
    ----------
    input_data : object
        Step input.

    Returns
    -------
    list of str or None
        The science members of an association (relative to the association's
        directory), the file name, or the list of file names; None when the input
        holds datamodels.
    """
    if isinstance(input_data, (str, os.PathLike)):
        path = os.fspath(input_data)
        if not path.endswith(".json"):
            return [path]
        with open(path) as asn_file:
            asn = load_asn(asn_file)
        folder = os.path.dirname(path)
        return [
            os.path.join(folder, member["expname"])
            for product in asn["products"]
            for member in product["members"]
            if member["exptype"] == "science"
        ]
    if isinstance(input_data, (list, tuple)) and all(isinstance(i, (str, os.PathLike)) for i in input_data):
        return [os.fspath(i) for i in input_data]
    return None

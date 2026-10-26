import logging
from pathlib import Path

from astropy.time import Time
from astropy.io import fits
import numpy as np
from jwst import datamodels
from .step import Step

log = logging.getLogger(__name__)


DO_NOT_USE = datamodels.dqflags.pixel["DO_NOT_USE"]
SATURATED = datamodels.dqflags.group["SATURATED"]
PERSISTENCE = datamodels.dqflags.pixel["PERSISTENCE"]


class PersistenceFlagStep(Step):
    """
    Given a series of exposures in an assocation, for any pixel flagged as saturated
    in one exposure, flag the pixel as DO_NOT_USE | PERSISTENCE in subsequent exposures
    up to some time in seconds, start time to start time.
    """

    spec = """
        time = float(default=2500.0)  # amount of time after exposure to flag [seconds]
        save_mask = boolean(default=False)  # write out persistence mask for each exposure
        output_use_model = boolean(default=True)
        output_use_index = boolean(default=False)
    """

    class_alias = "persist"
    fits_keyword = "S_PERSIS"
    fits_comment = "Persistence: snowblind.PersistenceFlagStep"

    def process(self, input_data):
        """Flag persistence from earlier saturated exposures in each image of an association.

        Parameters
        ----------
        input_data : str or ModelContainer
            Association file or container of image datamodels (``_rate`` or ``_cal``
            products) with ``dq`` arrays of shape (nrows, ncols).  The models are
            grouped by ``meta.instrument.detector`` and each group is handled on its
            own.  The matching ``_jump`` files are read from ``self.input_dir``.

        Returns
        -------
        ModelContainer
            Copy of the input in which pixels saturated in an earlier exposure of the
            same detector, started less than ``self.time`` seconds before, have
            ``DO_NOT_USE | PERSISTENCE`` set in ``dq``.
        """
        with datamodels.open(input_data) as images:
            # Find detector names in the association
            images_grouped_by_detector = {}
            detector_names = set([image.meta.instrument.detector for image in images])

            results = images.copy()

        # Sort exposures into a dict, one list per detector
        for detector in detector_names:
            det_list = [i for i in results if i.meta.instrument.detector == detector]
            images_grouped_by_detector.update({detector: det_list})

        # For each detector represented in the association
        for detector, models in images_grouped_by_detector.items():
            # Get DQ arrays sorted by observation start time, and difference in time between
            models_sorted, time_deltas = self.sort_by_start_times(models)

            log.info(f"Time deltas [sec] between {detector} exposures are {time_deltas}")

            # Get boolean cube of persistence pixels
            persist_bool = self.flag_saturated_in_subsequent(models_sorted, time_deltas)

            # Convert bool cube into PERSISTENCE flags for each image dq array
            for model, mask in zip(models_sorted, persist_bool):
                log.info(f"Pixels flagged: {model.meta.filename} {mask.sum()}")
                model.dq |= (mask * (DO_NOT_USE | PERSISTENCE)).astype(np.uint32)
                self.set_cal_step(model, "COMPLETE")

        return results

    def sort_by_start_times(self, images):
        """Returns sorted lists of datamodels and time_deltas between them [sec]

        Parameters
        ----------
        images : list of DataModel
            Exposures of one detector; sorted by ``meta.exposure.start_time`` (MJD).

        Returns
        -------
        models_sorted : list of DataModel
            The input models in order of increasing start time.
        time_deltas : list of float
            Seconds from the start of each exposure to the start of the one before
            it; the first entry is 0.
        """
        start_times = []
        for image in images:
            start_times.append((image.meta.exposure.start_time, image))

        # Sort all exposures by MJD
        start_times.sort()

        # Compute time deltas in seconds
        start_times_mjd = [Time(t, format="mjd") for t, _ in start_times]
        time_deltas = start_times_mjd - np.roll(start_times_mjd, shift=1)
        time_deltas_seconds = [tdelt.sec for tdelt in time_deltas]
        # Fix the first time delta due to np.roll() putting last item first
        time_deltas_seconds[0] = 0.

        return [m for _, m in start_times], time_deltas_seconds

    def flag_saturated_in_subsequent(self, models_sorted, time_deltas):
        """Flag as many subsequent SATURATED exposures as allowed by self.time

        Parameters
        ----------
        models_sorted : list of DataModel
            Exposures of one detector in start-time order, as from
            ``sort_by_start_times``.
        time_deltas : list of float
            Seconds between consecutive start times, first entry 0, as from
            ``sort_by_start_times``.

        Returns
        -------
        ndarray of bool, shape (nexposures, nrows, ncols)
            For each exposure, the pixels saturated in any earlier exposure whose
            start lies less than ``self.time`` seconds before this exposure's start.
        """
        # Find how man subsequent slices need to be flagged for each slice
        n_after = []
        for i, tdelt in enumerate(time_deltas):
            n_after.append(np.sum((np.cumsum(time_deltas[i:]) - tdelt) < self.time) - 1)

        # Make boolean array cube of saturated pixels
        satur_cube = self.get_saturation_masks(models_sorted)
        persist_cube = np.zeros_like(satur_cube, dtype=bool)

        for i, (sat_slice, n) in enumerate(zip(satur_cube, n_after)):
            for iter in range(n):
                try:
                    persist_cube[i + 1 + iter] |= sat_slice
                except IndexError:
                    pass

        return persist_cube

    def get_saturation_masks(self, models_sorted):
        """Get boolean SATURATION mask from output of JumpStep

        Parameters
        ----------
        models_sorted : list of DataModel
            Exposures whose ``meta.filename`` names the ``_jump.fits`` file to read
            from ``self.input_dir``.  When ``self.save_mask`` is set, each saturation
            cube is also written to a ``_satmask`` FITS file.

        Returns
        -------
        ndarray of bool, shape (nexposures, nrows, ncols)
            SATURATED pixels of the last group of the last integration of each
            ``_jump`` file.
        """
        # For the list of input files, convert them to the _jump.fits filenames
        def jumpify(filename):
            """Replace the product suffix of a standard JWST filename with ``_jump.fits``.

            Parameters
            ----------
            filename : str
                Exposure filename in the fixed-width JWST layout
                ``jwPPPPPOOOVVV_GGSAA_EEEEE_<detector>_<suffix>.fits``; the first 26
                characters and the rest of the detector name up to the next
                underscore are kept.

            Returns
            -------
            str
                The corresponding ``_jump.fits`` filename.
            """
            return filename[:26] + filename[26:26+filename[26:].find("_")] + \
                "_jump.fits"

        file_names = [jumpify(m.meta.filename) for m in models_sorted]

        masks = []
        for f in file_names:
            with datamodels.open(Path(self.input_dir) / f) as model:
                mask = (model.groupdq & SATURATED) == SATURATED
                filename = model.meta.filename

            # Get the last group of the last integration, as this will have the
            # cummulative, uncorrected saturated pixels flagged.
            masks.append(mask[-1, -1])

            if self.save_mask:
                sat_mask_name = filename.replace("_jump", "_satmask")
                fits.HDUList(
                    fits.PrimaryHDU(
                        data=mask.astype(np.uint8)
                    )
                ).writeto(sat_mask_name, overwrite=True)
                log.info(f"Writing out saturation mask {sat_mask_name}")

        return np.array(masks)

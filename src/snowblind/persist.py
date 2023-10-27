from pathlib import Path

from astropy.time import Time
from astropy.io import fits
import numpy as np
from jwst import datamodels
from jwst.stpipe import Step


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

    def process(self, input_data):
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

            self.log.info(f"Time deltas [sec] between {detector} exposures are {time_deltas}")

            # Get boolean cube of persistence pixels
            persist_bool = self.flag_saturated_in_subsequent(models_sorted, time_deltas)

            # Convert bool cube into PERSISTENCE flags for each image dq array
            for model, mask in zip(models_sorted, persist_bool):
                self.log.info(f"Pixels flagged: {model.meta.filename} {mask.sum()}")
                model.dq |= (mask * (DO_NOT_USE | PERSISTENCE)).astype(np.uint32)

        return results

    def sort_by_start_times(self, images):
        """Returns sorted lists of datamodels and time_deltas between them [sec]
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
        """
        # For the list of input files, convert them to the _jump.fits filenames
        def jumpify(filename):
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
                self.log.info(f"Writing out saturation mask {sat_mask_name}")

        return np.array(masks)

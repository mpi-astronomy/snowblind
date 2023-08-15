from glob import glob

from astropy.stats import sigma_clipped_stats
import numpy as np
from stdatamodels.jwst import datamodels
from jwst.stpipe import Step


RC = datamodels.dqflags.pixel["RC"]
DO_NOT_USE = datamodels.dqflags.pixel["DO_NOT_USE"]


class RcCrossStep(Step):
    """Removes cross-shaped defects caused by RC-type bad pixels in NIR detectors
    """
    spec = """
        glob_pattern = string(default=None) # glob pattern of files used to generate mask
        mask = is_string_or_datamodel(default=None) # boolean mask identifying RC pixels
        threshold = float(default=3.0) # threshold in sigma above stddev to flag
    """

    class_alias = "rc_cross"

    reference_file_types = ['mask']

    def process(self, input_data):
        with datamodels.open(input_data) as rate:

            # Make sure we're running this on 2D _rate files or equivalent
            if rate.data.ndim != 2:
                raise RuntimeError("Run this step on 2D images only")
            
            result = rate.copy()
        
        if self.glob_pattern is not None:
            image_stack = self._get_selfcal_stack()

            # Median collapse the stack of images
            stack_median = np.nanmedian(image_stack, axis=0)

            # Subtract background
            stack_median -= np.nanmedian(stack_median)

            # Clip to threshold
            _, median, std = sigma_clipped_stats(stack_median)
            hotpixel_mask = stack_median > median + self.threshold * std

            result.dq |= hotpixel_mask * (DO_NOT_USE + RC)

        return result

    def _get_selfcal_stack(self):
        """
        Get a stack of exposures taken with same detector as the data
        """
        rate_files = glob(self.glob_pattern)

        # Make sure the rate files are all the same detector
        detector_names = {filename.split(".")[0].split("_")[-2] for filename in rate_files}
        if len(detector_names) > 1:
            raise RuntimeError("Not all files are the same detector")

        stack = []
        for f in rate_files:
            with datamodels.open(f) as model:
                stack.append(model.data)

        return np.array(stack)
        
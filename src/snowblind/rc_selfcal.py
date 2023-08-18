from glob import glob

from astropy.stats import sigma_clipped_stats
import numpy as np
from stdatamodels.jwst import datamodels
from jwst.stpipe import Step
from astropy.io import fits


RC = datamodels.dqflags.pixel["RC"]
DO_NOT_USE = datamodels.dqflags.pixel["DO_NOT_USE"]


class RcSelfCalStep(Step):
    """Removes cross-shaped and other defects caused by RC-type bad pixels in NIR detectors

    Input is an assocation (or glob pattern of files) of all images in visit or program ID
    on which ones wishes to do a self-cal.  These are split into separate detector stacks,
    and then each image in the stack is operated on by the median of the images in that
    stack.  So input is N images, and output is N images, much like the first steps of
    stage 3 pipelines such as tweakreg, skymatch and outlier detection.

    Like outlier_detection, the input and output science images are the same, and only the
    data quality (DQ) array has new pixels flagged as DO_NOT_USE and RC.

    This should be run after flatfielding is finished in image2 pipeline.  It is fine to
    insert it anywhere in the level3 pipeline before resample.
    """
    spec = """
        threshold = float(default=3.0) # threshold in sigma to flag hot pixels above median
    """

    class_alias = "rc_selfcal"

    output_use_model = True

    def process(self, input_data):
        with datamodels.open(input_data) as images:

            # Sort into a dict of lists, grouped by detector
            images_grouped_by_detector = {}
            detector_names = set([image.meta.instrument.detector for image in images])
            for detector in detector_names:
                det_list = [i for i in images if i.meta.instrument.detector == detector]
                images_grouped_by_detector.update(dict(detector=det_list))

            results = images.copy()

        # For each detector represented in the association, compute a hot pixel mask and
        # np.bitwise_or() it with each input image for that detector
        for detector, dm in images_grouped_by_detector.items():
            image_stack = self.get_selfcal_stack(dm)
            mask = self.create_hotpixel_mask(image_stack)

            for result in results:
                if result.meta.instrument.detector == detector:
                    result.dq |= mask * (DO_NOT_USE + RC)

        return results

    def get_selfcal_stack(self, images):
        """
        Get a stack of exposures taken with same detector as the data
        """
        stack = []
        for model in images:
            stack.append(model.data)

        return np.array(stack)

    def create_hotpixel_mask(self, image_stack):
        # Median collapse the stack of images
        stack_median = np.nanmedian(image_stack, axis=0)

        # Subtract background
        stack_median -= np.nanmedian(stack_median)

        # Clip to threshold
        _, median, std = sigma_clipped_stats(stack_median)
        mask = stack_median > median + self.threshold * std
        mask |= stack_median < median + self.threshold * std

        return mask

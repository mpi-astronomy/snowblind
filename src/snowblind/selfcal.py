from os.path import commonprefix
import warnings

from astropy.stats import sigma_clipped_stats
import numpy as np
from jwst import datamodels
from jwst.stpipe import Step


OPEN = datamodels.dqflags.pixel["OPEN"]
ADJ_OPEN = datamodels.dqflags.pixel["ADJ_OPEN"]
DO_NOT_USE = datamodels.dqflags.pixel["DO_NOT_USE"]


class OpenPixelStep(Step):
    """Flags cross-shaped and hot pixel defects caused by open pixels in NIR detectors

    Input is an assocation (or glob pattern of files) of all images in visit or program ID
    on which ones wishes to do a self-cal.  These are split into separate detector stacks,
    and then each image in the stack is operated on by the median of the images in that
    stack.  So input is N images, and output is N images, much like the first steps of
    stage 3 pipelines such as tweakreg, skymatch and outlier detection.

    Like outlier_detection, the input and output science images are the same, and only the
    data quality (DQ) array has new pixels flagged as DO_NOT_USE and ADJ_OPEN.

    This should be run after flatfielding is finished in image2 pipeline.  It is fine to
    insert it anywhere in the level3 pipeline before resample.
    """
    spec = """
        threshold = float(default=3.0)  # threshold in sigma to flag hot pixels above local background
        save_mask = boolean(default=False)  # write out per-detector bad-pixel mask and median
        output_use_model = boolean(default=True)
        output_use_index = boolean(default=False)
        flag_low_signal_pix = boolean(default=False)
    """

    class_alias = "open_pixel"

    def process(self, input_data):
        with datamodels.open(input_data) as images:

            # Sort into a dict of lists, grouped by detector
            images_grouped_by_detector = {}
            detector_names = set([image.meta.instrument.detector for image in images])
            for detector in detector_names:
                det_list = [i for i in images if i.meta.instrument.detector == detector]
                images_grouped_by_detector.update({detector: det_list})

            results = images.copy()

        # For each detector represented in the association, compute a hot pixel mask and
        # np.bitwise_or() it with each input image for that detector
        for detector, models in images_grouped_by_detector.items():
            image_stack = self.get_selfcal_stack(models)
            self.log.info(f"Creating mask for detector {detector}")
            mask, median = self.create_hotpixel_mask(image_stack)
            self.log.info(f"Flagged {mask.sum()} pixels with {self.threshold} sigma")
            if self.save_mask:
                filename_prefix = f"{commonprefix([f.meta.filename for f in models])}_{detector.lower()}_{self.class_alias}"
                mask_model = datamodels.MaskModel(data=mask.astype(np.uint8))
                mask_model.meta.filename = f"{filename_prefix}.fits"
                self.save_model(mask_model, suffix="mask", force=True)
                median_model = datamodels.ImageModel(data=median)
                median_model.meta.filename = f"{filename_prefix}.fits"
                self.save_model(median_model, suffix="median", force=True)

            for result in results:
                if result.meta.instrument.detector == detector:
                    result.dq |= (mask * (DO_NOT_USE | ADJ_OPEN)).astype(np.uint32)

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
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
            median2d = np.nanmedian(image_stack, axis=0)

        # Clip to threshold
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore",
                                    message="Input data contains invalid values")
            _, med, std = sigma_clipped_stats(median2d, mask_value=np.nan)

        mask = median2d > med + self.threshold * std
        if self.flag_low_signal_pix:
            self.log.info(f"Flagging pixels {self.threshold}-sigma below median as well as those above.")
            mask |= median2d < med - self.threshold * std

        return mask, median2d

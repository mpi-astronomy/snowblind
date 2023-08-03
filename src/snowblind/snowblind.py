import skimage
import numpy as np
from stdatamodels.jwst import datamodels
from jwst.stpipe import Step


JUMP_DET = datamodels.dqflags.group["JUMP_DET"]


class SnowblindStep(Step):
    spec = """
        growth_factor = float(default=2.0)
    """

    class_alias = "snowblind"

    def process(self, input_image):
        with datamodels.open(input_image) as jump:
            outimage = jump.copy()
            bool_jump = (jump.groupdq & JUMP_DET) == JUMP_DET
            filename = jump.meta.filename

        # Loop over integrations and groups, ignoring first group of each integration
        # which will not have any jumps flagged.
        # Note, these are all boolean masks in this block
        for integ in range(bool_jump.shape[0]):
            for grp in range(bool_jump.shape[1]):
                jump_slice = bool_jump[integ, grp]

                # If there are no JUMP_DET in this group, skip it
                if not jump_slice.any():
                    continue

                # Fill holes in the JUMP_DET flagged areas (i.e. the saturated cores)
                cores_filled = skimage.morphology.remove_small_holes(jump_slice, area_threshold=200)

                # Get rid of the small-area jumps, leaving only large area CR events
                footprint = skimage.morphology.disk(radius=4)
                big_events = skimage.morphology.binary_opening(cores_filled, footprint=footprint)

                # Label and get properites of each snowball/shower
                event_labels, nlabels = skimage.measure.label(big_events, return_num=True)
                region_properties = skimage.measure.regionprops(event_labels)

                # For each snowball/shower, measure its size, and dilate by <growth_factor> * size
                jumps_expanded = np.zeros_like(jump_slice, dtype=bool)
                # zero-indexed loop, but labels are 1-indexed
                for label, region in zip(range(1, nlabels + 1), region_properties):
                    # make a boolean slice for each labelled event
                    segmentation_slice = event_labels == label
                    # Compute radius from equal-area circle
                    radius = np.ceil(np.sqrt(region.area / np.pi) * self.growth_factor)
                    # Warn if there are very large snowballs or showers detected
                    if region.area > 600:
                        self.log.warning(f"Large CR event with masked radius={radius} in {filename}[{integ},{grp}] at {region.centroid}")
                    segment_dilated = skimage.morphology.isotropic_dilation(segmentation_slice, radius=radius)
                    jumps_expanded |= segment_dilated

                # Add the new expanded halo JUMP_DET masks to the groupdq mask
                outimage.groupdq[integ, grp] = (jumps_expanded * JUMP_DET) | outimage.groupdq[integ, grp]

        return outimage

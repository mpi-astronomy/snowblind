import skimage
import numpy as np
from stdatamodels.jwst import datamodels
from stpipe import Step


JUMP_DET = datamodels.dqflags.group["JUMP_DET"]

class SnowblindStep(Step):
    spec = """
        growth_factor = float(default=2.5)
    """
    
    class_alias = "snowblind"

    def process(self, input_image):
        with datamodels.open(input_image) as jump:
            outimage = jump.copy()
            dq_jump = (jump.groupdq & JUMP_DET) == JUMP_DET

        # Loop over integrations and groups, ignoring first group of each integration
        # which will not have any jumps flagged.
        for integ in range(dq_jump.shape[0]):
            for grp in range(dq_jump.shape[1]):
                dq_slice = dq_jump[integ, grp]

                # If there are no JUMP_DET in this group, skip it
                if not dq_slice.any():
                    continue

                # Fill holes in the JUMP_DET flagged areas (i.e. the saturated cores)
                dq_no_holes = skimage.morphology.remove_small_holes(dq_slice, area_threshold=200)

                # Get rid of the small-area jumps, leaving only large snowballs/showers
                footprint = skimage.morphology.disk(radius=4)
                dq_open = skimage.morphology.binary_opening(dq_no_holes, footprint=footprint)

                # Label and get properites of each snowball/shower
                dq_labels, nlabels = skimage.measure.label(dq_open, return_num=True)
                region_properties = skimage.measure.regionprops(dq_labels)

                # For each snowball/shower, measure its size, and dilate by <growth_factor>
                # scaled by that size
                dq_iso_dilate = dq_open.copy()
                for label, region in zip(range(1, nlabels + 1), region_properties):
                    segmentation_slice = dq_labels == label
                    radius = np.ceil(np.sqrt(region.area / np.pi) * self.growth_factor)
                    # Warn if there are very large snowballs or showers detected
                    if region.area > 200:
                        self.log(f"Warning: snowball with radius={radius} in slice({integ},{grp}) with label {label} at {region.centroid}")
                    segment_dilated = skimage.morphology.isotropic_dilation(segmentation_slice, radius=radius)
                    dq_iso_dilate = dq_iso_dilate | segment_dilated

                # Add the new expanded halo JUMP_DET masks to the groupdq mask
                outimage.groupdq[integ,grp] = (dq_iso_dilate * JUMP_DET) | outimage.groupdq[integ,grp]
            
            return outimage

import skimage
import numpy as np
from jwst import datamodels
from jwst.stpipe import Step


JUMP_DET = datamodels.dqflags.group["JUMP_DET"]
SATURATED = datamodels.dqflags.group["SATURATED"]


class SnowblindStep(Step):
    spec = f"""
        min_radius = integer(default=4) # Minimum radius of connected pixels in CR
        growth_factor = float(default=2.0) # scale factor to dilate large CR events
        after_jumps = integer(default=2) # number of groups to flag around saturated cores after a jump
        ring_width = float(default=2.0) # number of pixels to dilate around saturated cores
        new_jump_flag = integer(default={JUMP_DET}) # DQ flag to set for dilated jumps
    """

    class_alias = "snowblind"

    def process(self, input_data):
        with datamodels.open(input_data) as jump:
            result = jump.copy()
            if hasattr(jump, 'groupdq'):
                self._has_groups = True
                bool_jump = (jump.groupdq & JUMP_DET) == JUMP_DET
                bool_sat = (jump.groupdq & SATURATED) == SATURATED
            else:
                self._has_groups = False
                bool_jump = (jump.dq & JUMP_DET) == JUMP_DET
                bool_sat = (jump.dq & SATURATED) == SATURATED

        # Expand jumps with large areas by self.growth_factor
        dilated_jumps = self.dilate_large_area_jumps(bool_jump)

        # bitwise OR together the dilated masks with the original GROUPDQ mask
        # We set the dilated saturated cores as jumps, as they are not saturated
        if self._has_groups:
            # Expand saturated cores within large event jumps by 2 pixels
            dilated_sats = self.dilate_saturated_cores(bool_sat, dilated_jumps)

            result.groupdq |= (dilated_jumps * self.new_jump_flag).astype(np.uint32)
            result.groupdq |= (dilated_sats * self.new_jump_flag).astype(np.uint32)
        else:
            result.dq |= (dilated_jumps * self.new_jump_flag).astype(np.uint32)

        # Update the metadata with the step completion status
        setattr(result.meta.cal_step, self.class_alias, "COMPLETE")

        return result

    def dilate_jump_slice(self, jump_slice, ig=None):
        """
        Dilate a boolean mask with contiguous large areas by a self.growth_factor

        Parameters
        ----------
        bool_jump : array-like, bool

        ig : (int, int) or None
            Integer indices of ``(integer, group)`` for logging

        Returns
        -------
        array-like, bool
        """
        # Create a mask to remove small CR events, used by binary_opening()
        disk = skimage.morphology.disk(radius=self.min_radius)

        # Fill holes in the flagged areas (i.e. the saturated cores)
        cores_filled = skimage.morphology.remove_small_holes(jump_slice, area_threshold=200)

        # Get rid of the small-area jumps, leaving only large area CR events
        big_events = skimage.morphology.binary_opening(cores_filled, footprint=disk)

        # Label and get properites of each large area event
        event_labels, nlabels = skimage.measure.label(big_events, return_num=True)
        region_properties = skimage.measure.regionprops(event_labels)

        # Break up the segmentation map <event_labels> into a slice for each labeled event
        # For each labeled event, measure its size, and dilate by <growth_factor> * size

        event_dilated = np.zeros_like(jump_slice)

        if self.growth_factor == 0:
            event_dilated |= big_events
            return event_dilated

        # zero-indexed loop, but labels are 1-indexed
        for label, region in zip(range(1, nlabels + 1), region_properties):
            # make a boolean slice for each labelled event
            segmentation_slice = event_labels == label
            # Compute radius from equal-area circle
            radius = np.sqrt(region.area / np.pi)
            dilate_radius = np.ceil(radius * self.growth_factor)
            # Warn if there are very large snowballs or showers detected
            if region.area > 900:
                y, x = region.centroid
                if ig is None:
                    msg = f"Large CR masked with radius={radius:.1f} at [{round(y)}, {round(x)}]"

                else:
                    msg = f"Large CR masked with radius={radius:.1f} at [{ig[0]}, {ig[1]}, {round(y)}, {round(x)}]"

                self.log.warning(msg)

            if dilate_radius > 0:
                event_dilated |= skimage.morphology.isotropic_dilation(segmentation_slice, radius=dilate_radius)
            else:
                event_dilated |= skimage.morphology.isotropic_erosion(segmentation_slice, radius=-dilate_radius)

        return event_dilated

    def dilate_large_area_jumps(self, bool_jump):
        """
        Dilate a boolean mask with contiguous large areas by a self.growth_factor

        Parameters
        ----------
        bool_jump : array-like, bool

        Returns
        -------
        array-like, bool
        """
        dilated_jumps = np.zeros_like(bool_jump, dtype=bool)

        # Loop over integrations and groups so we are dealing with one group slice at a time
        # Note, these are boolean masks in this block
        if self._has_groups:
            for i, integration in enumerate(bool_jump):
                for g, group in enumerate(integration):
                    jump_slice = group

                    # If there are no JUMP_DET in this group, skip it. True for the first group
                    # of an integration
                    if not jump_slice.any():
                        continue

                    dilated_jumps[i, g] |= self.dilate_jump_slice(jump_slice, ig=(i, g))
        else:
            if bool_jump.ndim == 3:
                # e.g., rateints
                for g in range(bool_jump.shape[0]):
                    dilated_jumps[g, :, :] |= self.dilate_jump_slice(bool_jump[g, :, :], ig=(0, g))
            else:
                # e.g., rate
                dilated_jumps |= self.dilate_jump_slice(bool_jump, ig=None)

        return dilated_jumps

    def dilate_saturated_cores(self, bool_sat, bool_jump):
        """
        Dilate the saturated cores of large CR events and propogate to subsequent groups
        """
        dilated_sats = np.zeros_like(bool_sat, dtype=bool)

        sat_from_jump = bool_sat & bool_jump

        # Propogate the saturated jump core flags to self.after_jumps subsequent group
        for integration in sat_from_jump:
            for i in range(self.after_jumps):
                shifted_flags = np.roll(integration, axis=0, shift=1)
                # Clean up the first group flags, which now have the last group
                shifted_flags[0] = False
                integration |= shifted_flags

        # Now that the boolean mask shows the saturated cores when the jump occurs
        # plus self.after_groups subsequent groups, dilate all of these by ring width
        for i, integ in enumerate(sat_from_jump):
            for g, grp in enumerate(integ):
                sat_slice = grp
                dilated_slice = skimage.morphology.isotropic_dilation(sat_slice, radius=self.ring_width)

                dilated_sats[i, g] = dilated_slice

        return dilated_sats

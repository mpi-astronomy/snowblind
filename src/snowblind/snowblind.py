import logging
import skimage
import numpy as np
from scipy import ndimage
from jwst import datamodels
from .step import Step

log = logging.getLogger(__name__)


JUMP_DET = datamodels.dqflags.group["JUMP_DET"]
SATURATED = datamodels.dqflags.group["SATURATED"]
REFERENCE_PIXEL = datamodels.dqflags.pixel["REFERENCE_PIXEL"]


class SnowblindStep(Step):
    """Mask snowball halos by growing each large jump-detected region around a saturated core.

    A snowball is a saturated core inside a halo of JUMP_DET pixels.  Each contiguous
    region of JUMP_DET pixels, its holes filled, that holds pixels saturating in its
    group or the next and contains a disk of ``min_radius`` is dilated by
    ``growth_factor`` times its equal-area radius, and its saturated core is dilated by
    ``ring_width`` pixels and carried into the ``after_jumps`` following groups.  In a ramp
    the core itself is also flagged in every group of the next integration, where it
    persists after the reset.  A rate image or rateints cube has no saturation per group,
    so there the holes in the region take the place of the core.

    Part of an event can lie on the reference pixels or past the array border, where the
    jump step cannot see it.  There the disk of ``min_radius`` may hang over the edge as
    long as its center is on a light-sensitive pixel, and the hidden part of the event is
    filled in before its radius is measured.  An event centered on the outermost
    light-sensitive row or column may have its core past the edge; it needs none if it
    holds a disk of ``min_radius`` that does not hang over the edge.

    The grown pixels are flagged with ``new_jump_flag`` (JUMP_DET by default) in the group
    data quality array of a ramp, or in the data quality array of a rate image or
    rateints cube.  Replaces ``expand_large_events`` in the standard JWST pipeline jump
    step and can be run on the ``_jump`` product or on rate products after the fact.

    Options from ``spec``: ``min_radius`` (minimum radius in pixels of a region to treat
    as a large event), ``growth_factor`` (dilation scale; 0 leaves the regions undilated,
    a negative value erodes them), ``after_jumps`` (number of groups after the jump that
    receive the saturated-core ring), ``ring_width`` (dilation radius in pixels around
    saturated cores) and ``new_jump_flag`` (DQ bit value set on the grown pixels).
    """
    spec = f"""
        min_radius = integer(default=2) # Minimum radius of connected pixels in CR
        growth_factor = float(default=2.0) # scale factor to dilate large CR events
        after_jumps = integer(default=2) # number of groups to flag around saturated cores after a jump
        ring_width = float(default=2.0) # number of pixels to dilate around saturated cores
        new_jump_flag = integer(default={JUMP_DET}) # DQ flag to set for dilated jumps
    """

    class_alias = "snowblind"
    fits_keyword = "S_SNOWBL"
    fits_comment = "Snowball flagging: snowblind.SnowblindStep"

    _n_events = 0  # large events found by dilate_jump_slice since process() started

    def process(self, input_data):
        """Flag the dilated large jump regions and saturated-core rings in a copy of the input.

        Parameters
        ----------
        input_data : str or RampModel or ImageModel or CubeModel
            Filename or datamodel after jump detection.  A model with a ``groupdq``
            array, shape (nints, ngroups, nrows, ncols), is treated as a ramp, whose
            cores are its newly saturated pixels and get dilated too; a model with only a
            ``dq`` array, shape (nrows, ncols) or (nints, nrows, ncols), is a rate image
            or rateints cube, whose cores are the holes in its jump regions.  Reference
            pixels are those with REFERENCE_PIXEL in ``pixeldq`` or ``dq``.

        Returns
        -------
        RampModel or ImageModel or CubeModel
            Copy of the input with ``new_jump_flag`` set in ``groupdq`` (ramps) or ``dq``
            (rate products) on the grown pixels and, in a ramp, on each core in every group
            of the next integration, and ``meta.cal_step.snowblind`` set to
            ``"COMPLETE"``.  For a ramp, ``meta.exposure.extended_emission_events`` (FITS
            keyword EXTNCRS) is set to the number of large events found per million
            pixels per second, replacing the value written by the jump step.
        """
        self._n_events = 0
        with datamodels.open(input_data) as jump:
            result = jump.copy()
            if hasattr(jump, 'groupdq'):
                self._has_groups = True
                bool_jump = (jump.groupdq & JUMP_DET) == JUMP_DET
                bool_sat = (jump.groupdq & SATURATED) == SATURATED
                refpix = (jump.pixeldq & REFERENCE_PIXEL) != 0
            else:
                self._has_groups = False
                bool_jump = (jump.dq & JUMP_DET) == JUMP_DET
                bool_sat = None
                refpix = np.any((jump.dq & REFERENCE_PIXEL) != 0, axis=tuple(range(jump.dq.ndim - 2)))

        # Expand jumps with large areas around saturated cores by self.growth_factor
        dilated_jumps, cores = self.dilate_large_area_jumps(bool_jump, bool_sat, refpix)
        log.info(f"Total snowballs = {self._n_events}")

        # bitwise OR together the dilated masks with the original GROUPDQ mask
        # We set the dilated saturated cores as jumps, as they are not saturated
        if self._has_groups:
            # Expand saturated cores within large event jumps by 2 pixels
            dilated_sats = self.dilate_saturated_cores(cores)
            persisting = self.cores_in_next_integration(cores)

            result.groupdq |= (dilated_jumps * self.new_jump_flag).astype(np.uint32)
            result.groupdq |= (dilated_sats * self.new_jump_flag).astype(np.uint32)
            result.groupdq |= (persisting * self.new_jump_flag).astype(np.uint32)
            self.set_event_rate(result, self._n_events)
        else:
            result.dq |= (dilated_jumps * self.new_jump_flag).astype(np.uint32)

        # Update the metadata with the step completion status
        self.set_cal_step(result, "COMPLETE")

        return result

    @staticmethod
    def set_event_rate(result, n_events):
        """
        Record the rate of large events in ``result.meta.exposure.extended_emission_events``

        Uses the same normalization as the jump step: events per million pixels per
        second of integration time, where the time is the group time times the number
        of groups, less the groups that are entirely DO_NOT_USE and the first group of
        each integration.  Leaves the keyword alone if the group time is unknown or no
        groups remain.

        Parameters
        ----------
        result : RampModel
            Model to update.
        n_events : int
            Number of large events found, summed over all groups and integrations.
        """
        nints, ngroups, nrows, ncols = result.groupdq.shape
        group_time = result.meta.exposure.group_time
        if group_time is None:
            return
        dnu = datamodels.dqflags.pixel["DO_NOT_USE"]
        all_dnu = np.all((result.groupdq & dnu) != 0, axis=(2, 3))
        total_groups = nints * ngroups - int(all_dnu.sum()) - nints
        if total_groups < 1:
            return
        result.meta.exposure.extended_emission_events = (
            1e6 * n_events / (group_time * total_groups * nrows * ncols)
        )

    def dilate_jump_slice(self, jump_slice, sat_slice=None, refpix=None, ig=None):
        """
        Dilate the large jump regions around saturated cores by a self.growth_factor

        Holes in the jump regions are filled when they hold a saturated core or are
        smaller than 200 pixels, regions without a core are dropped, regions smaller
        than a disk of ``self.min_radius`` are removed by a binary opening, and each
        remaining region is dilated by ``self.growth_factor`` times its equal-area radius
        (eroded when the factor is negative; left as is when it is 0).  Regions away
        from the edge go to ``_events_inside``, and regions next to the reference pixels
        or the array border to ``_events_at_edge``.

        Parameters
        ----------
        jump_slice : ndarray of bool, shape (nrows, ncols)
            Jump-detected pixels of one group (or of a rate image).
        sat_slice : ndarray of bool, shape (nrows, ncols), or None
            Pixels that saturate in this group or the next.  None for a rate image,
            where the holes in the jump regions are the cores.
        refpix : ndarray of bool, shape (nrows, ncols), or None
            Reference pixels.  None if there are none.
        ig : (int, int) or None
            Integer indices of ``(integer, group)`` for logging

        Returns
        -------
        dilated : ndarray of bool, shape (nrows, ncols)
            True on the grown large regions.
        cores : ndarray of bool, shape (nrows, ncols)
            The cores of those regions, before growing.
        """
        if refpix is None:
            refpix = np.zeros_like(jump_slice)

        # Fill holes in the flagged areas (i.e. the saturated cores)
        filled = self._fill_jump_holes(jump_slice, sat_slice, refpix)
        cores = filled & (~jump_slice if sat_slice is None else sat_slice)

        regions = skimage.measure.label(filled)
        with_core = np.unique(regions[cores])
        # Regions that touch the reference pixels or the array border may continue past them
        at_edge = set(np.unique(regions[self._next_to_edge(refpix) & filled]))

        dilated = np.zeros_like(jump_slice)
        event_cores = np.zeros_like(jump_slice)
        boxes = ndimage.find_objects(regions)
        for label in np.union1d(with_core, list(at_edge)).astype(int):
            # One pixel around the bounding box holds the empty pixels nearest the region
            window = tuple(
                slice(max(s.start - 1, 0), min(s.stop + 1, n)) for s, n in zip(boxes[label - 1], regions.shape)
            )
            region = regions[window] == label
            if label in at_edge:
                events = self._events_at_edge(region, window, cores, refpix)
            else:
                events = self._events_inside(region, window, cores)
            for event, completed in events:
                self._grow_event(dilated, completed, ig)
                event_cores[event] = cores[event]

        return dilated, event_cores

    def _events_inside(self, region, window, cores):
        """
        Find the large events of one filled jump region away from the edge

        A binary opening with a disk of ``self.min_radius`` removes the parts of the
        region too small to hold it; each part left that holds a core is an event.

        Parameters
        ----------
        region : ndarray of bool
            The region in ``window``.
        window : (slice, slice)
            Where the region sits in the slice.
        cores : ndarray of bool, shape (nrows, ncols)
            Core pixels.

        Returns
        -------
        list of ((ndarray, ndarray), (ndarray, ndarray))
            For each event, the row and column indices of its pixels, twice, as returned
            by ``_events_at_edge``.
        """
        # Get rid of the small-area jumps, leaving only large area CR events
        disk = skimage.morphology.disk(radius=self.min_radius)
        big = skimage.morphology.opening(region, footprint=disk)
        events = []
        for part in skimage.measure.regionprops(skimage.measure.label(big)):
            rows, cols = part.coords.T + np.array([[window[0].start], [window[1].start]])
            if cores[rows, cols].any():
                events.append(((rows, cols), (rows, cols)))
        return events

    @staticmethod
    def _fill_jump_holes(jump_slice, sat_slice, refpix):
        """
        Fill the holes of up to 200 pixels in the jump regions, and any that holds a saturated pixel

        Reference pixels count as jumps here, so that a core cut by the edge of the
        light-sensitive area is a hole too.  The largest hole, the background around the
        regions, stays open.  Returns the filled regions without the reference pixels.
        """
        blocked = jump_slice | refpix
        holes, nholes = ndimage.label(~blocked)
        size = np.bincount(holes.ravel(), minlength=nholes + 1)
        fill = size <= 200
        if sat_slice is not None and nholes > 0:
            holds_core = np.bincount(holes[sat_slice], minlength=nholes + 1) > 0
            holds_core[np.argmax(size[1:]) + 1] = False
            fill |= holds_core
        fill[0] = False
        return (jump_slice | fill[holes]) & ~refpix

    @staticmethod
    def _next_to_edge(refpix):
        """Pixels within one pixel of a reference pixel or of the array border."""
        padded = np.pad(refpix, 1, constant_values=True)
        return ndimage.binary_dilation(padded, structure=np.ones((3, 3)))[1:-1, 1:-1]

    @staticmethod
    def _past_edge(rows, cols, refpix):
        """True for the pixels ``(rows, cols)`` that are reference pixels or lie outside the array."""
        nrows, ncols = refpix.shape
        inside = (rows >= 0) & (cols >= 0) & (rows < nrows) & (cols < ncols)
        past = ~inside
        past[inside] = refpix[rows[inside], cols[inside]]
        return past

    def _events_at_edge(self, region, window, cores, refpix):
        """
        Find the large events of one filled jump region that touches the edge, and complete them past it

        An event may continue on the reference pixels or past the array border, so only
        the pixels next to the region are known to be empty.  The disk of
        ``self.min_radius`` that makes a large event may hang over the edge as long as
        its center is on the region.  Each event is completed past the edge by the
        largest disk around one of its pixels clear of the known empty pixels that comes
        closest to enclosing the event: for a round event whose center is on the
        detector, the event itself.  An event without a core is kept only when that disk
        is centered next to the edge, where the core may be past it, and the event holds
        a disk of ``self.min_radius`` that does not hang over the edge.

        Parameters
        ----------
        region : ndarray of bool
            The region in ``window``.
        window : (slice, slice)
            Where the region sits in the slice, with a margin of one pixel where the
            slice allows.
        cores : ndarray of bool, shape (nrows, ncols)
            Core pixels.
        refpix : ndarray of bool, shape (nrows, ncols)
            Reference pixels.

        Returns
        -------
        list of ((ndarray, ndarray), (ndarray, ndarray))
            For each event kept, the row and column indices of its pixels, and of its
            pixels with the completion past the edge added, which may lie outside the array.
        """
        top, left = window[0].start, window[1].start
        distance = ndimage.distance_transform_edt(region | refpix[window])
        centers = region & (distance > self.min_radius)
        big = region & ndimage.binary_dilation(centers, structure=skimage.morphology.disk(self.min_radius))
        centers_on_detector = ndimage.distance_transform_edt(np.pad(region, 1))[1:-1, 1:-1] > self.min_radius

        events = []
        event_labels, nevents = ndimage.label(big, structure=np.ones((3, 3)))
        for k in range(1, nevents + 1):
            event = event_labels == k
            rows, cols = np.nonzero(event)
            outline_rows, outline_cols = np.nonzero(event & ~ndimage.binary_erosion(event))
            farthest = np.hypot(rows[:, None] - outline_rows, cols[:, None] - outline_cols).max(axis=1)
            best = np.argmax(distance[rows, cols] - farthest)
            center = (rows[best] + top, cols[best] + left)
            disk_rows, disk_cols = skimage.draw.disk(center, distance[rows[best], cols[best]])
            hidden = self._past_edge(disk_rows, disk_cols, refpix)
            around_rows, around_cols = np.mgrid[-1:2, -1:2]
            centered_at_edge = self._past_edge(center[0] + around_rows, center[1] + around_cols, refpix).any()

            rows, cols = rows + top, cols + left
            if not cores[rows, cols].any():
                # No core in view: only a large event centered at the edge can have its core past it
                if not (centered_at_edge and (event & centers_on_detector).any()):
                    continue
            completed = (np.concatenate([rows, disk_rows[hidden]]), np.concatenate([cols, disk_cols[hidden]]))
            events.append(((rows, cols), completed))
        return events

    def _grow_event(self, dilated, event, ig):
        """
        Dilate one event by ``self.growth_factor`` times its equal-area radius into ``dilated``

        ``event`` holds the row and column indices of the event's pixels, which for an
        event at the edge include its completion past the edge, outside the array where
        it goes past the border.  The dilation is done on a canvas around the event and
        the part on the array is ORed into ``dilated``.
        """
        rows, cols = event
        self._n_events += 1
        # Compute radius from equal-area circle
        radius = np.sqrt(len(rows) / np.pi)
        dilate_radius = int(np.ceil(radius * self.growth_factor))
        # Warn if there are very large snowballs or showers detected
        if len(rows) > 1000:
            y, x = rows.mean(), cols.mean()
            if ig is None:
                msg = f"Large CR masked with radius={radius:.1f} at [{round(y)}, {round(x)}]"

            else:
                msg = f"Large CR masked with radius={radius:.1f} at [{ig[0]}, {ig[1]}, {round(y)}, {round(x)}]"

            log.warning(msg)

        pad = abs(dilate_radius) + 1
        top, left = rows.min() - pad, cols.min() - pad
        canvas = np.zeros((rows.max() - top + pad + 1, cols.max() - left + pad + 1), dtype=bool)
        canvas[rows - top, cols - left] = True
        if dilate_radius > 0:
            canvas = skimage.morphology.isotropic_dilation(canvas, radius=dilate_radius)
        elif dilate_radius < 0:
            canvas = skimage.morphology.isotropic_erosion(canvas, radius=-dilate_radius)

        nrows, ncols = dilated.shape
        y0, x0 = max(top, 0), max(left, 0)
        y1, x1 = min(top + canvas.shape[0], nrows), min(left + canvas.shape[1], ncols)
        dilated[y0:y1, x0:x1] |= canvas[y0 - top:y1 - top, x0 - left:x1 - left]

    def dilate_large_area_jumps(self, bool_jump, bool_sat=None, refpix=None):
        """
        Dilate a boolean mask with contiguous large areas by a self.growth_factor

        Parameters
        ----------
        bool_jump : ndarray of bool
            JUMP_DET pixels.  Shape (nints, ngroups, nrows, ncols) for a ramp, when
            ``self._has_groups`` is True; otherwise (nints, nrows, ncols) for a
            rateints cube or (nrows, ncols) for a rate image.  Each 2-d slice is
            passed to ``dilate_jump_slice``; slices with no jumps are skipped.
        bool_sat : ndarray of bool or None
            SATURATED pixels of a ramp, same shape as ``bool_jump``; None for rate
            products.
        refpix : ndarray of bool, shape (nrows, ncols), or None
            Reference pixels.

        Returns
        -------
        dilated_jumps : ndarray of bool
            Same shape as ``bool_jump``, True on the grown large regions.
        cores : ndarray of bool
            Same shape as ``bool_jump``, True on the cores of those regions.
        """
        dilated_jumps = np.zeros_like(bool_jump, dtype=bool)
        cores = np.zeros_like(bool_jump, dtype=bool)

        # Loop over integrations and groups so we are dealing with one group slice at a time
        # Note, these are boolean masks in this block
        if self._has_groups:
            for i, integration in enumerate(bool_jump):
                saturating = self._saturating_in_group_or_next(bool_sat[i])
                for g, group in enumerate(integration):
                    # If there are no JUMP_DET in this group, skip it. True for the first group
                    # of an integration
                    if not group.any():
                        continue

                    dilated_jumps[i, g], cores[i, g] = self.dilate_jump_slice(
                        group, saturating[g], refpix, ig=(i, g)
                    )
        else:
            if bool_jump.ndim == 3:
                # e.g., rateints
                for g in range(bool_jump.shape[0]):
                    dilated_jumps[g], cores[g] = self.dilate_jump_slice(bool_jump[g], refpix=refpix, ig=(0, g))
            else:
                # e.g., rate
                dilated_jumps, cores = self.dilate_jump_slice(bool_jump, refpix=refpix)

        return dilated_jumps, cores

    @staticmethod
    def _saturating_in_group_or_next(sat):
        """
        Pixels that saturate in each group or in the next one, for an integration of shape (ngroups, nrows, ncols)

        The core of a snowball that hits late in a group may saturate only in the next group.
        """
        newly = np.zeros_like(sat)
        newly[1:] = sat[1:] & ~sat[:-1]
        saturating = newly.copy()
        saturating[:-1] |= newly[1:]
        return saturating

    def dilate_saturated_cores(self, cores):
        """
        Dilate the saturated cores of large cosmic ray events and propagate them to subsequent groups

        Parameters
        ----------
        cores : ndarray of bool, shape (nints, ngroups, nrows, ncols)
            Cores of the large events found in each group, from
            ``dilate_large_area_jumps``.

        Returns
        -------
        ndarray of bool, shape (nints, ngroups, nrows, ncols)
            Cores carried into the ``self.after_jumps`` following groups of each
            integration and dilated by ``self.ring_width`` pixels.
        """
        dilated_sats = np.zeros_like(cores, dtype=bool)

        sat_from_jump = cores.copy()

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
                # isotropic_dilation of an empty image sets pixels in its corner
                if not grp.any():
                    continue
                sat_slice = grp
                dilated_slice = skimage.morphology.isotropic_dilation(sat_slice, radius=self.ring_width)

                dilated_sats[i, g] = dilated_slice

        return dilated_sats

    @staticmethod
    def cores_in_next_integration(cores):
        """
        Carry the saturated cores of each integration's large events into every group of the next integration

        The cores persist after the reset for the whole next integration; their neighbors do not.

        Parameters
        ----------
        cores : ndarray of bool, shape (nints, ngroups, nrows, ncols)
            Cores of the large events found in each group, from
            ``dilate_large_area_jumps``.

        Returns
        -------
        ndarray of bool, shape (nints, ngroups, nrows, ncols)
            True in every group of integration i + 1 on the cores of integration i.
        """
        persisting = np.zeros_like(cores, dtype=bool)
        persisting[1:] = cores[:-1].any(axis=1)[:, np.newaxis]
        return persisting

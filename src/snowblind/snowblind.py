import logging
import warnings
from importlib.resources import files

import asdf
import skimage
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError
from jwst import datamodels
from .step import Step

log = logging.getLogger(__name__)


JUMP_DET = datamodels.dqflags.group["JUMP_DET"]
SATURATED = datamodels.dqflags.group["SATURATED"]
REFERENCE_PIXEL = datamodels.dqflags.pixel["REFERENCE_PIXEL"]
SECONDS_PER_DAY = 86400.0
SQUARE = np.ones((3, 3), dtype=bool)
PACKAGED_TIME_PATTERN = "snowball_persistence_{detector}.asdf"
# the folder of the packaged times; tests point it elsewhere
PACKAGED_DIR = files("snowblind").joinpath("data")
AFTER_JUMPS_DEPRECATION = (
    "SnowblindStep's after_jumps is deprecated and will be removed in a future release. Left at None, the "
    "default, the ring around each snowball core is flagged to the end of its integration."
)


class SnowblindStep(Step):
    """Mask snowball halos by growing each large jump-detected region around a saturated core.

    A snowball is a saturated core inside a halo of JUMP_DET pixels.  Each contiguous
    region of JUMP_DET pixels, its holes filled, that holds a core and contains a disk of
    ``min_radius`` is dilated by ``growth_factor`` times its equal-area radius.  In a
    ramp the core is the pixels that saturate in the region's group or the next; in a
    rate image or rateints cube, which do not record when a pixel saturated, its
    SATURATED pixels.  In a ramp the core is also dilated by ``ring_width`` pixels in
    every group from the hit to the end of the integration (the ring, where charge from
    the core leaks into its neighbors), and it persists after the reset: in the later
    integrations of the exposure the core is flagged for ``core_persistence_time``
    seconds after the end of the integration of the hit, and the core grown by one pixel
    for ``neighbor_persistence_time`` seconds.  Both default to the times measured for
    the detector and packaged with snowblind, or to the longest packaged ones for a
    detector without its own.

    The cores are the SATURATED pixels of the event, so ``ring_width`` and the one-pixel
    growth for the neighbors are counted from the SATURATED flags.  With the jwst
    saturation step's default, ``n_pix_grow_sat=1``, those include a one-pixel ring
    around the pixels that reached the saturation threshold, and the ring reaches
    ``ring_width`` + 1 pixels from them; ``snowblind_detector1`` sets ``ring_width`` to
    reach 3 pixels from them whatever ``n_pix_grow_sat`` it runs with.

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
    a negative value erodes them), ``ring_width`` (dilation radius in pixels around the
    saturated cores), ``after_jumps`` (deprecated: when set, the ring is flagged in the
    group of the jump and this many groups after it instead of to the end of the
    integration), ``core_persistence_time`` and ``neighbor_persistence_time`` (seconds
    after the end of the integration of the hit during which the core, and the pixels
    next to it, are flagged in later integrations; None takes the detector's packaged
    times) and ``new_jump_flag`` (DQ bit value set on the grown pixels).
    """
    spec = f"""
        min_radius = integer(default=2) # Minimum radius of connected pixels in CR
        growth_factor = float(default=2.0) # scale factor to dilate large CR events
        ring_width = float(default=2.0) # pixels to dilate around saturated cores, counted from the SATURATED flags
        after_jumps = integer(default=None) # deprecated; None flags the ring to the end of the integration
        core_persistence_time = float(default=None) # s after the hit's integration to flag cores; None: packaged
        neighbor_persistence_time = float(default=None) # the same for the pixels next to the cores
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
            or rateints cube, whose cores are its SATURATED pixels.  Reference pixels are
            those with REFERENCE_PIXEL in ``pixeldq`` or ``dq``.

        Returns
        -------
        RampModel or ImageModel or CubeModel
            Copy of the input with ``new_jump_flag`` set in ``groupdq`` (ramps) or ``dq``
            (rate products) on the grown pixels and, in a ramp, on the rings around the
            cores and on the cores and their neighbors in later integrations (see
            `cores_in_later_integrations`), and ``meta.cal_step.snowblind`` set to
            ``"COMPLETE"``.  For a ramp, ``meta.exposure.extended_emission_events`` (FITS
            keyword EXTNCRS) is set to the number of large events found per million
            pixels per second, replacing the value written by the jump step.
        """
        if self.after_jumps is not None:
            warnings.warn(AFTER_JUMPS_DEPRECATION, DeprecationWarning, stacklevel=2)
            log.warning(AFTER_JUMPS_DEPRECATION)
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
                # the ramp fit sets SATURATED in dq (the same bit) on a pixel saturated in any group
                bool_sat = (jump.dq & SATURATED) == SATURATED
                refpix = np.any((jump.dq & REFERENCE_PIXEL) != 0, axis=tuple(range(jump.dq.ndim - 2)))

        # Expand jumps with large areas around saturated cores by self.growth_factor
        dilated_jumps, cores = self.dilate_large_area_jumps(bool_jump, bool_sat, refpix)
        log.info(f"Total snowballs = {self._n_events}")

        # bitwise OR together the dilated masks with the original GROUPDQ mask
        # We set the dilated saturated cores as jumps, as they are not saturated
        if self._has_groups:
            dilated_sats = self.dilate_saturated_cores(cores)
            result.groupdq |= (dilated_jumps * self.new_jump_flag).astype(np.uint32)
            result.groupdq |= (dilated_sats * self.new_jump_flag).astype(np.uint32)
            del dilated_sats
            if cores[:-1].any():
                persisting = self._cores_persisting(result, cores)
                if persisting is not None:
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

    def dilate_jump_slice(self, jump_slice, sat_slice, refpix=None, ig=None):
        """
        Dilate the large jump regions around saturated cores by a self.growth_factor

        Holes in the jump regions are filled when they are of up to 200 pixels or are
        made only of saturated pixels and their neighbors (a core of any size), the
        saturated pixels in a filled region are its core, regions without one are
        dropped, regions smaller
        than a disk of ``self.min_radius`` are removed by a binary opening, and each
        remaining region is dilated by ``self.growth_factor`` times its equal-area radius
        (eroded when the factor is negative; left as is when it is 0).  Regions away
        from the edge go to ``_events_inside``, and regions next to the reference pixels
        or the array border to ``_events_at_edge``.

        Parameters
        ----------
        jump_slice : ndarray of bool, shape (nrows, ncols)
            Jump-detected pixels of one group (or of a rate image).
        sat_slice : ndarray of bool, shape (nrows, ncols)
            Pixels that saturate in this group or the next; for a rate image, its
            SATURATED pixels.
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
        cores = filled & sat_slice

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
        Fill the holes of up to 200 pixels in the jump regions, and any made only of saturated pixels and their neighbors

        A snowball's core is its saturated pixels and the pixels next to them, which the
        saturation step flags so that no jump is found there.  A larger hole that also
        holds pixels farther from them is sky: around a saturated star in a rate image,
        or cut off by a jump region that crosses the array.  Reference pixels count as
        jumps here, so that a core cut by the edge of the light-sensitive area is a hole
        too.  Returns the filled regions without the reference pixels.
        """
        blocked = jump_slice | refpix
        holes, nholes = ndimage.label(~blocked)
        size = np.bincount(holes.ravel(), minlength=nholes + 1)
        near_saturated = ndimage.binary_dilation(sat_slice, structure=SQUARE)
        far_from_saturated = np.bincount(holes[~near_saturated], minlength=nholes + 1)
        fill = (size <= 200) | (far_from_saturated == 0)
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

    @staticmethod
    def _distance_to_empty(known):
        """Each pixel's distance to the nearest pixel not in ``known``, or to just past the array when every pixel is.

        scipy measures from the array's first corner when it has no empty pixel, as in a
        small subarray that an event covers together with the reference pixels.
        """
        if known.all():
            return ndimage.distance_transform_edt(np.pad(known, 1, constant_values=False))[1:-1, 1:-1]
        return ndimage.distance_transform_edt(known)

    @staticmethod
    def _distance_to_farthest_pixel(event, rows, cols):
        """For each pixel ``(rows, cols)`` of ``event``, the distance to the event's farthest pixel.

        The farthest pixel is always a corner of the event's convex hull, so only the
        corners are measured, one at a time, which keeps memory to the event's size.
        """
        outline = np.argwhere(event & ~ndimage.binary_erosion(event))
        try:
            corners = outline[ConvexHull(outline).vertices]
        except QhullError:  # fewer than three pixels, or all on one line
            corners = outline
        farthest = np.zeros(len(rows))
        for corner_row, corner_col in corners:
            np.maximum(farthest, np.hypot(rows - corner_row, cols - corner_col), out=farthest)
        return farthest

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
        distance = self._distance_to_empty(region | refpix[window])
        centers = region & (distance > self.min_radius)
        big = region & ndimage.binary_dilation(centers, structure=skimage.morphology.disk(self.min_radius))
        centers_on_detector = ndimage.distance_transform_edt(np.pad(region, 1))[1:-1, 1:-1] > self.min_radius

        events = []
        event_labels, nevents = ndimage.label(big, structure=np.ones((3, 3)))
        for k in range(1, nevents + 1):
            event = event_labels == k
            rows, cols = np.nonzero(event)
            best = np.argmax(distance[rows, cols] - self._distance_to_farthest_pixel(event, rows, cols))
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

    def dilate_large_area_jumps(self, bool_jump, bool_sat, refpix=None):
        """
        Dilate a boolean mask with contiguous large areas by a self.growth_factor

        Parameters
        ----------
        bool_jump : ndarray of bool
            JUMP_DET pixels.  Shape (nints, ngroups, nrows, ncols) for a ramp, when
            ``self._has_groups`` is True; otherwise (nints, nrows, ncols) for a
            rateints cube or (nrows, ncols) for a rate image.  Each 2-d slice is
            passed to ``dilate_jump_slice``; slices with no jumps are skipped.
        bool_sat : ndarray of bool
            SATURATED pixels, same shape as ``bool_jump``.
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
        elif bool_jump.ndim == 3:
            # e.g., rateints
            for g in range(bool_jump.shape[0]):
                dilated_jumps[g], cores[g] = self.dilate_jump_slice(bool_jump[g], bool_sat[g], refpix, ig=(0, g))
        else:
            # e.g., rate
            dilated_jumps, cores = self.dilate_jump_slice(bool_jump, bool_sat, refpix)

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
        Dilate the saturated cores of large cosmic ray events and carry them to the end of the integration

        Parameters
        ----------
        cores : ndarray of bool, shape (nints, ngroups, nrows, ncols)
            Cores of the large events found in each group, from
            ``dilate_large_area_jumps``.

        Returns
        -------
        ndarray of bool, shape (nints, ngroups, nrows, ncols)
            Each core dilated by ``self.ring_width`` pixels in its group and every later
            group of its integration; with ``self.after_jumps`` set (deprecated), in its
            group and the ``self.after_jumps`` following groups only.
        """
        if self.after_jumps is not None:
            return self._ring_for_after_jumps_groups(cores)
        dilated_sats = np.zeros_like(cores, dtype=bool)
        for i, integration in enumerate(cores):
            hit = np.zeros(cores.shape[-2:], dtype=bool)
            ring = None
            for g, group in enumerate(integration):
                # the ring changes only in the groups that bring new cores
                if (group & ~hit).any():
                    hit |= group
                    ring = skimage.morphology.isotropic_dilation(hit, radius=self.ring_width)
                if ring is not None:
                    dilated_sats[i, g] = ring
        return dilated_sats

    def _ring_for_after_jumps_groups(self, cores):
        """`dilate_saturated_cores` with ``self.after_jumps`` set: the ring in the group of the jump and the
        ``self.after_jumps`` following groups of the integration."""
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

    def _cores_persisting(self, model, cores):
        """`cores_in_later_integrations` for a ramp, with its group times and its persistence times.

        Returns None, after a warning, when the group times are unknown.
        """
        times = group_difference_times(model)
        if times is None:
            log.warning("The frame time is unknown, so the snowball cores are not flagged in later integrations")
            return None
        core_time, neighbor_time = self._persistence_times(model.meta.instrument.detector)
        log.info(f"Snowball cores flagged in later integrations for {core_time:.0f} s after the end of the "
                 f"integration of the hit, the pixels next to them for {neighbor_time:.0f} s")
        return self.cores_in_later_integrations(cores, *times, core_time, neighbor_time)

    def _persistence_times(self, detector):
        """``core_persistence_time`` and ``neighbor_persistence_time``, each taken from the detector's packaged
        file when None; when the package has no file for the detector, the longest of each over the packaged
        files, after a warning, and 0 s if there are none."""
        core_time, neighbor_time = self.core_persistence_time, self.neighbor_persistence_time
        if core_time is None or neighbor_time is None:
            packaged = packaged_snowball_times(detector)
            if packaged is None:
                packaged = longest_packaged_snowball_times()
                log.warning(
                    f"No snowball persistence times are packaged for detector {detector}: taking the longest "
                    f"packaged, {packaged[0]:.0f} s for the cores and {packaged[1]:.0f} s for the pixels next to them"
                )
            core_time = packaged[0] if core_time is None else core_time
            neighbor_time = packaged[1] if neighbor_time is None else neighbor_time
        return core_time, neighbor_time

    @staticmethod
    def cores_in_later_integrations(cores, difference_starts, integration_ends, core_time, neighbor_time):
        """
        Flag the cores of each integration's large events, and the pixels next to them, in the later integrations

        A core persists after the reset: the excess rate of its pixels falls with the time
        since the end of the integration of the hit, and that of the pixels next to it
        (one pixel away, the 3x3 dilation of the core) faster.  A group of a later
        integration is flagged when the difference it ends starts less than the given time
        after that end, group 0 included, so that a fit from the reset leaves out the
        reset difference.

        Parameters
        ----------
        cores : ndarray of bool, shape (nints, ngroups, nrows, ncols)
            Cores of the large events found in each group, from
            ``dilate_large_area_jumps``.
        difference_starts : ndarray, shape (nints, ngroups)
            Time, in seconds, at which the difference ending at each group starts: the
            start of the integration for group 0, the read of the group before for the others.
        integration_ends : ndarray, shape (nints,)
            Time, in seconds on the same scale, of the last read of each integration.
        core_time : float
            Seconds after the end of the integration of the hit during which the core is flagged.
        neighbor_time : float
            Seconds after the end of the integration of the hit during which the core
            grown by one pixel is flagged.

        Returns
        -------
        ndarray of bool, shape (nints, ngroups, nrows, ncols)
            True in the groups of later integrations where a core, or a pixel next to it,
            is still within its time.
        """
        persisting = np.zeros_like(cores, dtype=bool)
        for i in range(len(cores) - 1):
            core = cores[i].any(axis=0)
            if not core.any():
                continue
            grown = ndimage.binary_dilation(core, structure=SQUARE)
            for j in range(i + 1, len(cores)):
                since = difference_starts[j] - integration_ends[i]
                persisting[j, since < neighbor_time] |= grown
                persisting[j, since < core_time] |= core
        return persisting


def group_difference_times(model):
    """When each group's difference starts and each integration ends, in seconds from the first integration's start.

    The integration start and end times come from the ``int_times`` table when it holds
    every integration of the model with each ending after it starts; otherwise from the
    frame counts: an integration spans ``ngroups * (nframes + groupgap)`` frames from its
    start to its end, and ``nresets_between_ints`` (default 1) reset frames separate it
    from the next.  A group is read at the end of its last frame.

    Parameters
    ----------
    model : RampModel
        The ramp, with ``meta.exposure.frame_time``, ``nframes`` and ``groupgap``.

    Returns
    -------
    tuple of ndarray or None
        ``difference_starts`` of shape (nints, ngroups), the start of the integration for
        group 0 and the read of the group before for the others, and
        ``integration_ends`` of shape (nints,); None when the frame time is unknown.
    """
    nints, ngroups = model.groupdq.shape[:2]
    exposure = model.meta.exposure
    if not exposure.frame_time:
        return None
    frame_time = float(exposure.frame_time)
    nframes = exposure.nframes or 1
    frames_per_group = nframes + (exposure.groupgap or 0)
    starts, ends = _integration_times_from_table(model, nints)
    if starts is None:
        span = ngroups * frames_per_group * frame_time
        starts = np.arange(nints) * (span + (exposure.nresets_between_ints or 1) * frame_time)
        ends = starts + span
    reads = (np.arange(ngroups) * frames_per_group + nframes) * frame_time
    difference_starts = starts[:, np.newaxis] + np.concatenate([[0.0], reads[:-1]])
    return difference_starts, ends


def _integration_times_from_table(model, nints):
    """Start and end, in seconds from the first start, of each integration of the model from its ``int_times``
    table; (None, None) when the table lacks one of them or has one that does not end after it starts."""
    table = getattr(model, "int_times", None)
    if table is None or len(table) == 0:
        return None, None
    first = model.meta.exposure.integration_start or 1
    numbers = np.asarray(table["integration_number"])
    rows = [np.flatnonzero(numbers == first + k) for k in range(nints)]
    if any(len(row) == 0 for row in rows):
        return None, None
    rows = [row[0] for row in rows]
    starts = np.asarray(table["int_start_MJD_UTC"], dtype=float)[rows] * SECONDS_PER_DAY
    ends = np.asarray(table["int_end_MJD_UTC"], dtype=float)[rows] * SECONDS_PER_DAY
    if not (ends > starts).all():
        return None, None
    return starts - starts[0], ends - starts[0]


def packaged_snowball_times(detector):
    """The snowball persistence times measured for a detector and packaged with snowblind.

    ``snowblind/data/snowball_persistence_<detector>.asdf`` holds ``core_time``: the
    99th percentile over the detector's snowballs of the time, in seconds from the end of
    the integration of the hit, at which a core's saturated pixels fall to the file's
    ``threshold`` (DN/s); ``neighbor_time``, the same for the pixels next to them, both
    lengthened where later darks show more excess (``later_check``); and the
    measurement behind them.  ``scripts/measure_snowball_persistence.py`` measures
    the excess in dark exposures, ``scripts/fit_snowball_persistence.py`` fits it and
    ``scripts/package_snowball_persistence.py`` writes the files.

    Parameters
    ----------
    detector : str or None
        Detector name as in ``meta.instrument.detector``; case does not matter.

    Returns
    -------
    tuple of float or None
        ``(core_time, neighbor_time)`` in seconds; None when the package has no file for
        the detector.
    """
    if detector is None:
        return None
    path = PACKAGED_DIR.joinpath(PACKAGED_TIME_PATTERN.format(detector=str(detector).lower()))
    if not path.is_file():
        return None
    return _times_in_file(path)


def longest_packaged_snowball_times():
    """The longest core time and the longest neighbor time over every detector's packaged file.

    The default for a detector without a file of its own.

    Returns
    -------
    tuple of float
        ``(core_time, neighbor_time)`` in seconds, each the largest of its kind over the
        packaged files (not necessarily from the same detector); ``(0.0, 0.0)`` when there
        are none.
    """
    prefix, suffix = PACKAGED_TIME_PATTERN.split("{detector}")
    times = [
        _times_in_file(path)
        for path in PACKAGED_DIR.iterdir()
        if path.name.startswith(prefix) and path.name.endswith(suffix)
    ]
    if not times:
        return 0.0, 0.0
    return max(core for core, _ in times), max(neighbor for _, neighbor in times)


def _times_in_file(path):
    """``(core_time, neighbor_time)`` of one packaged snowball persistence file."""
    with asdf.open(str(path), lazy_load=False, memmap=False) as handle:
        return float(handle.tree["core_time"]), float(handle.tree["neighbor_time"])

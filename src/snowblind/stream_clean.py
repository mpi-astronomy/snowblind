"""Remove 1/f (flicker) noise from near-infrared group data.

The step works on group differences, as ``jwst.clean_flicker_noise`` does,
but estimates the noise per output channel along the detector readout
stream instead of per image row.  The scene mask, background model and
group-difference loop reuse the jwst step's machinery (see
``_jwst_imports.py`` for the pinned helpers).  The loop over integrations
and groups is adapted from ``do_correction`` in
``jwst/clean_flicker_noise/clean_flicker_noise.py`` at jwst commit e15d3e779.
"""
import logging

import numpy as np
from jwst import datamodels
from jwst.stpipe import Step
from stdatamodels.jwst.datamodels import dqflags

from . import readout
from ._jwst_imports import (
    background_level,
    clip_to_background,
    make_intermediate_model,
    make_processed_rate_image,
    make_scene_mask,
    mask_unusable,
    read_flat_file,
)
from .noise_spectrum import (
    LEAKAGE_BINS,
    channel_spectra,
    fit_power_law,
    power_law,
    smooth_spectrum,
    taper_high_frequencies,
    wiener_gain,
)
from .scene_mask import chisel_mask
from .stream_filters import FILTERS

__all__ = [
    "StreamCleanStep",
    "clean_ramp",
    "clean_difference",
    "prepare_difference",
    "estimate_noise",
    "even_odd_pattern",
    "column_offset_pattern",
    "estimate_channel_gains",
    "integration_backgrounds",
    "masked_mean_background",
    "slope_per_group",
    "exclude_reference_border",
    "read_scene_model",
    "dilate_sources",
    "first_group_time_ratio",
    "channel_gain",
    "zero_gain",
    "write_spectra",
    "read_psd_prior",
]

log = logging.getLogger(__name__)

SATURATED = dqflags.group["SATURATED"]
SUPPORTED_INSTRUMENTS = ("NIRCAM", "NIRISS")


class StreamCleanStep(Step):
    """Remove 1/f noise from NIR group data by filtering each channel's readout stream.

    Input is a `~stdatamodels.jwst.datamodels.RampModel` after the jump
    step, or any group-level product with ``groupdq`` populated.  Output is
    the same model type with cleaned ``data`` and
    ``meta.cal_step.stream_clean`` set to COMPLETE.  Reference pixels are
    expected to have been handled upstream by the refpix step; they take
    part in the clock grid but are not used to fit the noise.
    """

    class_alias = "stream_clean"

    @classmethod
    def get_stpipe_loggers(cls):
        """The jwst loggers plus snowblind's, so that this package's messages reach the step log."""
        return super().get_stpipe_loggers() + ("snowblind",)

    spec = """
        fit_method = option('wiener', 'boxcar', 'row_median', default='wiener')  # noise estimator per channel
        window_length = integer(default=4096)  # pixel times (10 microseconds each); boxcar only
        psd_prior = string(default=None)  # optional ASDF with a dark-derived PSD shape per channel; wiener only
        background_method = option('median', 'model', 'integration_model', 'exposure_model', None, default='integration_model')  # background removed before fitting
        background_box_size = int_list(min=2, max=2, default=None)  # box size for background_method='model'
        scene_mask = option('chisel', 'clip', default='chisel')  # how sources are found in the draft rate image
        n_sigma = float(default=2.0)  # clipping level for scene_mask='clip'
        mask_passes = integer(default=1)  # with 2 or more, re-derive the chisel mask from the cleaned ramp and clean again
        clip_sigma = float(default=0.0)  # per-difference clipping of the background-subtracted image; 0 keeps the cleaning linear
        mask_science_regions = boolean(default=False)  # mask known science regions
        apply_flat_field = boolean(default=False)  # divide by the flat before fitting
        user_mask = string(default=None)  # path to a user-supplied background mask
        mask_dilation = integer(default=0)  # pixels by which to grow the masked (source) regions, for source wings
        clean_first_group = boolean(default=True)  # also clean group 1, against zero or against the reset prior; needed for reset_rampfit, harmless for OLS
        reset_calib = string(default=None)  # snowblind-reset-calib ASDF; with it the first read is cleaned against its reset-level prior, as reset_rampfit will use it
        scene_model = string(default=None)  # FITS image of the scene in DN/s, detector frame, before flat-fielding (e.g. the blotted mosaic); replaces the background model
        iterations = integer(default=1)  # noise estimates per difference; re-clips with clip_sigma between them
        min_density = float(default=0.2)  # local background fraction below which the estimate tapers off
        first_direction = integer(default=1)  # readout direction of channel 0 in the detector frame, 1 or -1
        even_odd_columns = boolean(default=True)  # remove the per-channel even-odd column offset of each difference
        column_offsets = boolean(default=True)  # remove each column's masked mean from each difference (a per-column offset that drifts during the integration)
        anchor_mean = boolean(default=True)  # set each channel's mean removed noise to the masked mean of the residual, so a robust estimator cannot bias the sky
        max_lag = integer(default=65536)  # autocorrelation length in pixel times for the noise spectrum; wiener only
        protect_period = integer(default=None)  # pixel times; noise slower than this is left to the background model; wiener only
        min_period = integer(default=300)  # pixel times (3 ms); the wiener gain is zero for shorter periods (None keeps all)
        save_mask = boolean(default=False)  # save the background mask
        save_noise = boolean(default=False)  # save the removed noise
        save_spectra = boolean(default=False)  # save per-channel spectra and gains; wiener only
        skip = boolean(default=False)
    """  # noqa: E501

    reference_file_types = ["flat"]

    def process(self, input_data):
        """Clean the group data of a ramp.

        Parameters
        ----------
        input_data : str or `~stdatamodels.jwst.datamodels.RampModel`
            Group-level data with ``groupdq`` populated.

        Returns
        -------
        `~stdatamodels.jwst.datamodels.RampModel`
            Cleaned copy of the input.
        """
        with datamodels.open(input_data) as input_model:
            result = input_model.copy()

        if self.first_direction not in (1, -1):
            raise ValueError(f"first_direction must be 1 or -1, got {self.first_direction}")

        reason = _unsupported_reason(result)
        if reason is not None:
            log.warning(f"{reason}; skipping stream_clean")
            setattr(result.meta.cal_step, self.class_alias, "SKIPPED")
            return result

        flat = None
        if self.apply_flat_field:
            flat_filename = self.get_reference_file(result, "flat")
            log.info(f"Using FLAT reference file {flat_filename}")
            flat = read_flat_file(result, flat_filename)

        self._scene = None
        if self.scene_model is not None:
            self._scene = read_scene_model(self.scene_model, result.data.shape[-2:])
            log.info(
                f"Scene model {self.scene_model}: median {np.nanmedian(self._scene):.4g} DN/s, "
                f"{100 * np.mean(~np.isfinite(self._scene)):.2f} percent without coverage"
            )

        background_mask, mask_model = self._scene_mask(result, flat)
        background_mask = exclude_reference_border(background_mask)
        if self.mask_dilation > 0:
            background_mask = dilate_sources(background_mask, self.mask_dilation)
        if background_mask.shape[-2:] != result.data.shape[-2:]:
            log.warning(
                f"Mask shape {background_mask.shape} does not match data shape "
                f"{result.data.shape}; skipping stream_clean"
            )
            setattr(result.meta.cal_step, self.class_alias, "SKIPPED")
            return result

        log.info(f"Cleaning {result.meta.filename} with fit_method={self.fit_method}")
        prior_alphas = read_psd_prior(self.psd_prior) if self.psd_prior else None
        cleaned, spectra = self._clean_with_mask(result, background_mask, flat, prior_alphas)
        for extra_pass in range(1, max(self.mask_passes, 1)):
            if self.user_mask is not None or self.scene_mask != "chisel":
                break
            # a draft rate from the cleaned ramp, free of the banding, gives a better mask
            draft = np.nanmean([slope_per_group(cleaned[i], result.groupdq[i]) for i in range(cleaned.shape[0])], axis=0)
            draft = draft / (result.meta.exposure.group_time or 1.0)
            draft[(result.pixeldq & dqflags.pixel["DO_NOT_USE"]) > 0] = np.nan
            background_mask = exclude_reference_border(chisel_mask(draft))
            if self.mask_dilation > 0:
                background_mask = dilate_sources(background_mask, self.mask_dilation)
            log.info(f"Scene mask pass {extra_pass + 1}: {100 * (~background_mask).mean():.2f} percent of pixels masked")
            cleaned, spectra = self._clean_with_mask(result, background_mask, flat, prior_alphas)

        if self.save_spectra and spectra is not None:
            spectra_path = self.make_output_path(basepath=result.meta.filename, suffix="spectra", ext="asdf")
            write_spectra(spectra_path, spectra, result.meta.instrument.detector, self.first_direction, self.max_lag)
            log.info(f"Saved spectra to {spectra_path}")

        if self.save_noise:
            noise_model = make_intermediate_model(result, cleaned - result.data)
            self.save_model(noise_model, suffix="noise", force=True)
            noise_model.close()
        if mask_model is not None:
            if self.mask_passes > 1:
                mask_model.data = background_mask.astype(mask_model.data.dtype)
            self.save_model(mask_model, suffix="mask", force=True)
            mask_model.close()

        result.data = cleaned
        if self._cleaned_zeroframe is not None:
            result.zeroframe = self._cleaned_zeroframe
        setattr(result.meta.cal_step, self.class_alias, "COMPLETE")
        return result

    def _clean_with_mask(self, result, background_mask, flat, prior_alphas):
        """Run `clean_ramp` with the step's parameters and a given background mask."""
        reference = None
        if self.reset_calib is not None and self.clean_first_group:
            from .reset_rampfit import reset_prior_map

            reference = reset_prior_map(
                self.reset_calib,
                result.data.shape[-2:],
                result.meta.subarray.fastaxis,
                result.meta.subarray.slowaxis,
                science_superbias=result.meta.ref_file.superbias.name,
            )
            log.info(f"First read cleaned against the reset prior of {self.reset_calib}")
        scene_background = None
        if self._scene is not None:
            # the scene in DN per group; pixels without coverage get zero and are outside the mask
            scene_background = (np.where(np.isfinite(self._scene), self._scene, 0.0) * result.meta.exposure.group_time).astype(
                np.float32
            )
        self._cleaned_zeroframe = None
        zeroframe = None
        if self.clean_first_group and getattr(result, "zeroframe", None) is not None and np.any(result.zeroframe != 0):
            # frame-averaged readouts carry the first frame separately; the reset fit uses it, so clean it too
            zeroframe = np.array(result.zeroframe, dtype=np.float32, copy=True)
            self._cleaned_zeroframe = zeroframe
            log.info("Cleaning the zeroframe as well")
        frame_time = result.meta.exposure.frame_time or 0.0
        group_time = result.meta.exposure.group_time or 1.0
        return clean_ramp(
            result.data,
            result.groupdq,
            background_mask,
            result.meta.subarray.fastaxis,
            result.meta.subarray.slowaxis,
            first_group_reference=reference,
            scene_background=scene_background,
            zeroframe=zeroframe,
            zeroframe_scale=frame_time / group_time,
            fit_method=self.fit_method,
            background_method=self.background_method,
            background_box_size=self.background_box_size,
            clip_sigma=self.clip_sigma,
            clean_first_group=self.clean_first_group,
            iterations=self.iterations,
            flat=flat,
            first_direction=self.first_direction,
            window_length=self.window_length,
            min_density=self.min_density,
            max_lag=self.max_lag,
            protect_period=self.protect_period,
            min_period=self.min_period,
            prior_alphas=prior_alphas,
            return_spectra=True,
            first_group_scale=first_group_time_ratio(result),
            even_odd_columns=self.even_odd_columns,
            column_offsets=self.column_offsets,
            anchor_mean=self.anchor_mean,
        )

    def _scene_mask(self, model, flat):
        """Build the background mask with the jwst helpers.

        With ``user_mask`` set the mask is read from that file.  Otherwise a
        draft rate image is made by running the ramp fit step and the mask
        is derived from it.
        """
        if self.user_mask is None:
            image_model = make_processed_rate_image(
                model,
                True,
                self.input_dir,
                model.meta.exposure.type,
                self.mask_science_regions,
                flat,
                self.background_method,
            )
        else:
            image_model = model

        if self.user_mask is None and self.scene_mask == "chisel":
            rate = np.array(image_model.data, dtype=np.float64)
            if hasattr(image_model, "dq") and image_model.dq is not None:
                rate[(image_model.dq & dqflags.pixel["DO_NOT_USE"]) > 0] = np.nan
            if self._scene is not None:
                # with a scene model the mask only has to cover where the model is wrong
                rate = rate - self._scene
            background_mask = chisel_mask(rate)
            if self._scene is not None:
                background_mask &= np.isfinite(self._scene)
            log.info(f"Scene mask (chisel): {100 * (~background_mask).mean():.2f} percent of pixels masked")
            mask_model = make_intermediate_model(image_model, background_mask) if self.save_mask else None
        else:
            background_mask, mask_model = make_scene_mask(
                self.user_mask,
                image_model,
                self.mask_science_regions,
                self.n_sigma,
                False,
                True,
                self.save_mask,
            )
        if image_model is not model:
            image_model.close()
        return background_mask, mask_model


def read_scene_model(path, shape):
    """Read a scene model (DN/s, detector frame, before flat-fielding) from a FITS file.

    The image is taken from the SCI extension when present, else from the
    primary HDU.  Non-finite pixels mean no coverage.
    """
    from astropy.io import fits

    with fits.open(path) as handle:
        data = handle["SCI"].data if "SCI" in handle else handle[0].data
        scene = np.asarray(data, dtype=np.float32)
    if scene.shape != tuple(shape):
        raise ValueError(f"Scene model shape {scene.shape} does not match data {tuple(shape)}")
    return scene


def exclude_reference_border(background_mask, width=readout.NREFPIX):
    """Return the mask with the reference-pixel border set to False.

    Reference pixels take part in the clock grid but carry no sky and are
    not light sensitive; the refpix step has already used them.  They must
    not enter the noise fit or the background model.
    """
    mask = np.array(background_mask, dtype=bool, copy=True)
    mask[..., :width, :] = False
    mask[..., -width:, :] = False
    mask[..., :, :width] = False
    mask[..., :, -width:] = False
    return mask


def dilate_sources(background_mask, pixels):
    """Grow the masked (False) regions by ``pixels`` in every direction.

    The scene mask from a sigma clip of the draft rate stops where a source
    drops below the clipping level; its wings extend further and would
    otherwise be filtered as noise.
    """
    from scipy import ndimage

    sources = ~np.asarray(background_mask, dtype=bool)
    structure = ndimage.generate_binary_structure(2, 1)
    if sources.ndim == 3:
        grown = np.array([ndimage.binary_dilation(plane, structure, iterations=pixels) for plane in sources])
    else:
        grown = ndimage.binary_dilation(sources, structure, iterations=pixels)
    return ~grown


def first_group_time_ratio(model):
    """Mean time since reset of the first group's reads, divided by the group time.

    Follows the stcal convention that the first read comes one frame time
    after the reset, so the mean time of the first group is
    ``frame_time * (nframes + 1) / 2``.
    """
    nframes = model.meta.exposure.nframes or 1
    frame_time = model.meta.exposure.frame_time
    group_time = model.meta.exposure.group_time
    if not frame_time or not group_time:
        return 1.0
    return float(frame_time * (nframes + 1) / 2.0 / group_time)


def _unsupported_reason(model):
    """Return why the model cannot be cleaned, or None if it can."""
    instrument = str(model.meta.instrument.name).upper()
    if instrument not in SUPPORTED_INSTRUMENTS:
        return f"Instrument {instrument} is not supported"
    if model.data is None or model.data.ndim != 4:
        return "Input is not group-level (4D) data"
    if str(model.meta.subarray.name).upper() != "FULL":
        return f"Subarray {model.meta.subarray.name} is not supported; only FULL frame"
    if model.data.shape[-1] % readout.NOUTPUTS != 0:
        return f"{model.data.shape[-1]} columns cannot be split into {readout.NOUTPUTS} channels"
    return None


def iterate_differences(data, groupdq, background_mask, clean_first_group, first_group_reference=None):
    """Yield ``(i, j, image, mask)`` for every group difference to clean.

    ``image`` is group ``j`` minus group ``j - 1`` of integration ``i`` (or
    group 0 minus ``first_group_reference``, or group 0 itself, when ``j ==
    0`` and ``clean_first_group`` is set) as float32 in the science frame;
    ``mask`` is a fresh copy of the scene mask with DO_NOT_USE, JUMP_DET and
    SATURATED pixels of group ``j`` removed.
    """
    nints, ngroups = data.shape[:2]
    first = 0 if clean_first_group else 1
    for i in range(nints):
        for j in range(first, ngroups):
            if background_mask.ndim == 3:
                mask = background_mask[i].copy()
            else:
                mask = background_mask.copy()
            if j == 0:
                image = data[i, 0].astype(np.float32)
                if first_group_reference is not None:
                    image = image - first_group_reference
            else:
                image = data[i, j].astype(np.float32) - data[i, j - 1].astype(np.float32)
            dq = groupdq[i, j]
            mask_unusable(mask, dq)
            mask[(dq & SATURATED) > 0] = False
            yield i, j, image, mask


def clean_ramp(
    data,
    groupdq,
    background_mask,
    fastaxis,
    slowaxis,
    fit_method="row_median",
    background_method="median",
    background_box_size=None,
    clip_sigma=0.0,
    clean_first_group=False,
    iterations=1,
    flat=None,
    first_direction=1,
    max_lag=65536,
    prior_alphas=None,
    return_spectra=False,
    first_group_scale=1.0,
    min_period=None,
    even_odd_columns=False,
    column_offsets=False,
    anchor_mean=False,
    first_group_reference=None,
    scene_background=None,
    zeroframe=None,
    zeroframe_scale=None,
    **filter_kwargs,
):
    """Clean every group difference of every integration and rebuild the groups.

    Parameters
    ----------
    data : ndarray, shape (nints, ngroups, nrows, ncols)
        Group data in the science frame.
    groupdq : ndarray of int, same shape as ``data``
        Group DQ flags.  Pixels flagged DO_NOT_USE, JUMP_DET or SATURATED in
        the later group of a difference are excluded from the noise fit.
    background_mask : ndarray of bool, shape (nrows, ncols) or (nints, nrows, ncols)
        Scene mask, True for background pixels.
    fastaxis, slowaxis : int
        Detector readout axis keywords.
    fit_method : str
        Key into `snowblind.stream_filters.FILTERS`.
    background_method, background_box_size, clip_sigma, iterations, flat
        See `clean_difference`.
    clean_first_group : bool
        If True, group 1 is cleaned as a difference against a plane of zeros.
    first_direction : int
        Readout direction of channel 0 in the detector frame.
    max_lag : int
        Autocorrelation length in clocks for the ``wiener`` spectrum.
    prior_alphas : sequence of float or None
        Per-channel power-law slopes to hold fixed in the ``wiener`` fit.
    return_spectra : bool
        If True, also return the per-integration spectra (``wiener`` only;
        None otherwise).
    first_group_scale : float
        Ratio of the first group's mean time since reset to the group time;
        scales the ``integration_model`` background for group 1 cleaned
        against zero.
    min_period : int or None
        ``wiener`` only: the gain is cut to zero for periods shorter than
        this many clocks, so that compact sources missed by the scene mask
        are not filtered.
    even_odd_columns, column_offsets, anchor_mean : bool
        See `estimate_noise`.
    first_group_reference : ndarray or None
        With ``clean_first_group``, the image group 0 is cleaned against
        (the reset-level prior); None cleans it against zero.
    scene_background : ndarray or None
        The scene in DN per group (for example a blotted mosaic), used as
        the fixed background of every difference in place of the fitted
        model; ``background_method`` is then ignored.
    zeroframe : ndarray of shape (nints, nrows, ncols) or None
        The ZEROFRAME of frame-averaged readouts, the first frame of the
        first group, which the likelihood fit uses as a resultant of its
        own.  With ``clean_first_group`` it is cleaned in place, like
        group 0, as a difference against ``first_group_reference`` (or
        zero); pixels at exactly zero, the pipeline's marker for an
        unusable zeroframe pixel, stay zero.
    zeroframe_scale : float or None
        The frame time over the group time: scales a fixed background to
        the signal of one frame.
    **filter_kwargs
        Passed to the noise estimator.

    Returns
    -------
    cleaned : ndarray of float32
        Cleaned group data.  Cleaned group j+1 is cleaned group j plus the
        cleaned difference, so the noise removed from one difference is not
        reintroduced by the next.
    spectra : list or None
        Only when ``return_spectra`` is True: one entry per integration,
        each a list of per-channel dicts from `channel_gain`.
    """
    nints = data.shape[0]
    cleaned = np.array(data, dtype=np.float32, copy=True)

    fixed = None
    if scene_background is not None:
        background_method = "integration_model"
        fixed = [np.asarray(scene_background, dtype=np.float32)] * nints
    elif background_method in ("integration_model", "exposure_model"):
        fixed = integration_backgrounds(
            data, groupdq, background_mask, background_box_size, flat, per_exposure=background_method == "exposure_model"
        )

    gains = None
    spectra = None
    if fit_method == "wiener":
        spectra = estimate_channel_gains(
            data,
            groupdq,
            background_mask,
            fastaxis,
            slowaxis,
            background_method=background_method,
            background_box_size=background_box_size,
            clip_sigma=clip_sigma,
            clean_first_group=clean_first_group,
            flat=flat,
            first_direction=first_direction,
            max_lag=max_lag,
            prior_alphas=prior_alphas,
            fixed_backgrounds=fixed,
            first_group_reference=first_group_reference,
            first_group_scale=first_group_scale,
            min_period=min_period,
        )
        gains = [[{"gain": (ch["freq"], ch["gain"])} for ch in integration] for integration in spectra]
        for i in range(nints):
            summary = ", ".join(
                f"ch{k}: alpha={ch['alpha']:.2f} W={ch['white']:.3g} corner={ch['corner_period']:.0f} clocks"
                for k, ch in enumerate(spectra[i])
            )
            log.info(f"Integration {i + 1} noise spectra: {summary}")

    for i, j, image, mask in iterate_differences(data, groupdq, background_mask, clean_first_group, first_group_reference):
        if j == 0:
            previous = np.zeros_like(image) if first_group_reference is None else first_group_reference
        else:
            previous = cleaned[i, j - 1]
        cleaned_image, _background = clean_difference(
            image,
            mask,
            fastaxis,
            slowaxis,
            fit_method=fit_method,
            background_method=background_method,
            background_box_size=background_box_size,
            clip_sigma=clip_sigma,
            iterations=iterations,
            flat=flat,
            first_direction=first_direction,
            channel_kwargs=None if gains is None else gains[i],
            fixed_background=_fixed_for(fixed, i, j, first_group_scale),
            even_odd_columns=even_odd_columns,
            column_offsets=column_offsets,
            anchor_mean=anchor_mean,
            **filter_kwargs,
        )
        if cleaned_image is None:
            log.warning(f"No usable data in integration {i + 1}, group {j + 1}; difference left as is")
            cleaned_image = image
        cleaned[i, j] = previous + cleaned_image

    if zeroframe is not None and clean_first_group:
        reference = 0.0 if first_group_reference is None else first_group_reference
        for i in range(nints):
            frame = np.asarray(zeroframe[i], dtype=np.float32)
            unusable = frame == 0.0
            mask = (background_mask[i] if background_mask.ndim == 3 else background_mask).copy()
            mask_unusable(mask, groupdq[i, 0])
            mask &= ~unusable
            image = frame - reference
            scale = first_group_scale if zeroframe_scale is None else zeroframe_scale
            cleaned_image, _background = clean_difference(
                image,
                mask,
                fastaxis,
                slowaxis,
                fit_method=fit_method,
                background_method=background_method,
                background_box_size=background_box_size,
                clip_sigma=clip_sigma,
                iterations=iterations,
                flat=flat,
                first_direction=first_direction,
                channel_kwargs=None if gains is None else gains[i],
                fixed_background=None if fixed is None else fixed[i] * scale,
                even_odd_columns=even_odd_columns,
                column_offsets=column_offsets,
                anchor_mean=anchor_mean,
                **filter_kwargs,
            )
            if cleaned_image is None:
                log.warning(f"No usable data in the zeroframe of integration {i + 1}; left as is")
                continue
            new = reference + cleaned_image
            new[unusable] = 0.0
            zeroframe[i] = np.where(np.isfinite(new), new, 0.0)

    if return_spectra:
        return cleaned, spectra
    return cleaned


def _fixed_for(fixed, integration, group, first_group_scale):
    """The fixed background for one difference, or None when none is in use."""
    if fixed is None:
        return None
    if group == 0:
        return fixed[integration] * first_group_scale
    return fixed[integration]


def prepare_difference(
    image, mask, background_method="median", background_box_size=None, clip_sigma=0.0, flat=None, fixed_background=None
):
    """Mask NaNs, remove the background and clip signal, in the raw DN of the difference.

    This is the part of ``_clean_one_image`` in the jwst step that comes
    before the noise fit.  ``mask`` is updated in place.  With
    ``fixed_background`` given (the ``integration_model`` method) that array
    is subtracted instead of fitting a background to this image.  Clipping
    at ``clip_sigma`` happens only when a background was removed and only if
    ``clip_sigma`` is positive.

    With ``flat`` the background is measured on the flat-fielded image, as
    the sky is smooth there and the detector response is not, but what is
    subtracted is that background times the flat, so that the residual
    handed to the noise fit stays in raw DN, where the 1/f noise is
    additive and does not pass through the quantum efficiency.  (The jwst
    step divides the whole image by the flat, fits in that space and
    multiplies back, which modulates the noise estimate by the flat.)  A
    ``fixed_background`` is in flat-fielded units when a flat is given.

    Returns
    -------
    bkg_sub : ndarray of float32 or None
        Background-subtracted image in raw DN, or None if no pixel is usable.
    background : float or ndarray
        The background that was removed, in raw DN (times the flat when one
        was given).
    nan_pix : ndarray of bool
        Where the input was not finite.
    """
    image = image.astype(np.float32, copy=True)
    flat_ok = None
    if flat is not None:
        flat = np.asarray(flat, dtype=np.float32)
        flat_ok = np.isfinite(flat) & (flat > 0)
        image[~flat_ok] = np.nan

    nan_pix = ~np.isfinite(image)
    image[nan_pix] = 0.0
    mask[nan_pix] = False
    if not np.any(mask):
        return None, None, nan_pix

    no_background = background_method is None or str(background_method).lower() == "none"
    if background_method == "integration_model" and fixed_background is None:
        raise ValueError("background_method='integration_model' needs the per-integration background (fixed_background)")
    if fixed_background is not None:
        background = fixed_background
    elif no_background:
        background = 0.0
    else:
        # measured on the flat-fielded image, where the sky is smooth
        fielded = image if flat is None else np.where(flat_ok, image / np.where(flat_ok, flat, 1.0), 0.0)
        background = background_level(
            fielded.astype(np.float32), mask, background_method=background_method, background_box_size=background_box_size
        )
    if flat is not None and not no_background:
        background = np.asarray(background, dtype=np.float32) * np.where(flat_ok, flat, 0.0)
    bkg_sub = image - background
    # As in the jwst step, signal is clipped only after a background was removed.
    if clip_sigma > 0 and (fixed_background is not None or not no_background):
        clip_to_background(bkg_sub, mask, sigma_lower=clip_sigma, sigma_upper=clip_sigma, lower_half_only=True)
    return bkg_sub, background, nan_pix


def slope_per_group(data_int, groupdq_int):
    """Least-squares slope of one integration's ramp per pixel, in DN per group.

    Pixels with any flagged group fall back to the mean of their usable
    differences, which is what a slope through two points reduces to.
    """
    ngroups = data_int.shape[0]
    cube = data_int.astype(np.float64)
    t = np.arange(ngroups, dtype=np.float64)
    weights = (t - t.mean()) / np.sum((t - t.mean()) ** 2)
    slope = np.tensordot(weights, cube, axes=(0, 0))
    flagged = np.any(groupdq_int != 0, axis=0)
    if flagged.any():
        diffs = cube[1:] - cube[:-1]
        usable = (groupdq_int[1:] == 0) & (groupdq_int[:-1] == 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            fallback = np.where(usable, diffs, 0.0).sum(axis=0) / usable.sum(axis=0)
        slope = np.where(flagged, fallback, slope)
    return slope.astype(np.float32)


def masked_mean_background(image, mask, box_size=None, filter_size=3):
    """Smooth background as the masked mean per box, interpolated between boxes.

    ``photutils.Background2D`` with a mean estimator, no sigma clipping and
    every box that has any usable pixel, so that the model is unbiased with
    respect to the masked pixels at scales above the box.  Boxes without a
    usable pixel are filled from their neighbours.
    """
    from photutils.background import Background2D, MeanBackground

    box = (32, 32) if box_size is None else tuple(box_size)
    good = mask & np.isfinite(image)
    if not np.any(good):
        return np.zeros(image.shape, dtype=np.float32)
    filled = np.where(np.isfinite(image), image, 0.0).astype(np.float32)
    background = Background2D(
        filled,
        box,
        mask=~good,
        filter_size=(filter_size, filter_size),
        sigma_clip=None,
        bkg_estimator=MeanBackground(),
        exclude_percentile=100.0,
    )
    return np.asarray(background.background, dtype=np.float32)


def integration_backgrounds(data, groupdq, background_mask, background_box_size=None, flat=None, per_exposure=False):
    """One background model per integration, fitted to the least-squares slope of the ramp.

    The jwst step fits a background to every difference separately.  That
    model is nonlinear and noisy, and because cleaned groups are rebuilt by
    accumulating cleaned differences, its per-difference noise random-walks
    along the ramp and ends up in the fitted rate.  Fitting the model once
    per integration keeps the cleaning a fixed linear operation on every
    difference while still protecting the sky from the noise filter.

    The model is fitted to the ramp's least-squares slope expressed per
    group, not to the mean group difference: at scales above the box the
    cleaned rate reproduces whatever structure the model holds, and the
    mean difference (last group minus first) is a far noisier estimate of
    that structure than the slope.  With ``per_exposure`` the slopes of all
    integrations are averaged and the same model serves every integration.

    The model is the plain masked mean per box (`masked_mean_background`),
    not a sigma-clipped median.  The filter removes the mean of (data -
    model) per channel, so any bias of the model at scales above the box
    is a bias of the cleaned sky: on a crowded field a clipped median sat
    15 to 20 percent below the masked mean and took that much sky with it.
    Source rejection is the scene mask's job, not the model's.

    Returns
    -------
    list of ndarray
        One background image per integration, in the units of a group
        difference.  For group 1 cleaned against zero the caller scales it
        by the ratio of the first group's time to the group time.
    """
    nints = data.shape[0]
    slopes = []
    for i in range(nints):
        slope = slope_per_group(data[i], groupdq[i])
        if flat is not None:
            slope = slope / flat
        slopes.append(slope)

    def model(image, mask):
        return masked_mean_background(image, mask, background_box_size)

    if per_exposure:
        mask = np.all(background_mask, axis=0) if background_mask.ndim == 3 else background_mask
        shared = model(np.nanmean(slopes, axis=0), mask)
        return [shared] * nints
    return [model(slopes[i], background_mask[i] if background_mask.ndim == 3 else background_mask) for i in range(nints)]


def estimate_channel_gains(
    data,
    groupdq,
    background_mask,
    fastaxis,
    slowaxis,
    background_method="median",
    background_box_size=None,
    clip_sigma=0.0,
    clean_first_group=False,
    flat=None,
    first_direction=1,
    max_lag=65536,
    prior_alphas=None,
    fixed_backgrounds=None,
    first_group_reference=None,
    first_group_scale=1.0,
    min_period=None,
):
    """First pass of the ``wiener`` method: the noise spectrum and gain of every channel.

    For every group difference the background-subtracted, clipped image is
    laid out on the clock grid channel by channel and its masked
    autocorrelation turned into a spectrum.  The spectra are median-combined
    over the differences of an integration (the noise is stationary, the
    scene residual is not) and passed to `channel_gain`.

    Returns
    -------
    list
        One entry per integration, each a list of four dicts from
        `channel_gain`.
    """
    nints = data.shape[0]
    det_shape = readout.to_detector_frame(data[0, 0], fastaxis, slowaxis).shape
    slices = readout.channel_slices(det_shape[-1])
    lag_default = min(max_lag, readout.stream_length(det_shape[0], readout.channel_width(det_shape[-1])) // 4)
    per_integration = [[[] for _ in slices] for _ in range(nints)]
    freq = None

    for i, j, image, mask in iterate_differences(
        data, groupdq, background_mask, clean_first_group, first_group_reference
    ):
        bkg_sub, _background, _nan_pix = prepare_difference(
            image, mask, background_method, background_box_size, clip_sigma, flat,
            _fixed_for(fixed_backgrounds, i, j, first_group_scale),
        )
        if bkg_sub is None:
            continue
        image_det = readout.to_detector_frame(bkg_sub, fastaxis, slowaxis)
        mask_det = readout.to_detector_frame(mask, fastaxis, slowaxis)
        freq, spectra = channel_spectra(image_det, mask_det, first_direction, max_lag)
        for k, spectrum in enumerate(spectra):
            if spectrum is not None:
                per_integration[i][k].append(spectrum)

    results = []
    for i in range(nints):
        channels = []
        for k in range(len(slices)):
            if not per_integration[i][k] or freq is None:
                log.warning(
                    f"No usable difference to estimate the spectrum of channel {k} in integration {i + 1}; "
                    "nothing will be removed from that channel"
                )
                channels.append(zero_gain(lag_default))
                continue
            median = np.median(per_integration[i][k], axis=0)
            fixed = None if prior_alphas is None else prior_alphas[k]
            channels.append(channel_gain(freq, median, fixed_alpha=fixed, min_period=min_period))
        results.append(channels)
    return results


def zero_gain(max_lag):
    """A channel entry that removes nothing, for channels with no usable data."""
    freq = np.fft.rfftfreq(2 * max_lag, d=1.0)
    zeros = np.zeros(freq.size, dtype=np.float32)
    return {
        "freq": freq,
        "spectrum": zeros,
        "smooth": zeros,
        "gain": zeros.copy(),
        "amplitude": 0.0,
        "alpha": float("nan"),
        "white": 0.0,
        "corner_period": float("nan"),
    }


def channel_gain(freq, spectrum, alpha_bounds=(0.5, 1.5), fixed_alpha=None, min_period=None):
    """Turn a channel's median spectrum into a Wiener gain.

    A power law plus white floor is fitted to the log-binned spectrum.  The
    gain uses the smoothed nonparametric spectrum where it is well measured
    and the fit at the lowest frequencies (leakage) and the highest five
    percent (where the smoothing has few bins).  The gain is forced to one
    at zero frequency so that the normalized convolution is well defined.

    Returns
    -------
    dict
        ``freq``, ``spectrum`` (median input), ``smooth``, ``gain``,
        ``amplitude``, ``alpha``, ``white`` and ``corner_period`` (clocks
        at which the colored power equals the white floor).
    """
    if not np.any(np.isfinite(spectrum) & (spectrum > 0)):
        log.warning("Channel spectrum has no positive power; nothing will be removed from that channel")
        return zero_gain(freq.size - 1)
    try:
        amplitude, alpha, white = fit_power_law(freq, spectrum, alpha_bounds=alpha_bounds, fixed_alpha=fixed_alpha)
    except ValueError as err:
        log.warning(f"Channel spectrum could not be fitted ({err}); nothing will be removed from that channel")
        return zero_gain(freq.size - 1)
    smooth = smooth_spectrum(freq, spectrum)
    model = power_law(freq, amplitude, alpha, white)
    use_model = (freq < LEAKAGE_BINS * freq[1]) | (freq > 0.95 * freq[-1])
    combined = np.where(use_model, model, smooth)
    gain = wiener_gain(combined, white)
    if min_period is not None:
        gain = taper_high_frequencies(freq, gain, min_period)
    gain[0] = 1.0
    with np.errstate(divide="ignore"):
        corner_period = float((white / amplitude) ** (1.0 / alpha)) if amplitude > 0 else float("inf")
    return {
        "freq": freq,
        "spectrum": spectrum.astype(np.float32),
        "smooth": smooth.astype(np.float32),
        "gain": gain.astype(np.float32),
        "amplitude": float(amplitude),
        "alpha": float(alpha),
        "white": float(white),
        "corner_period": corner_period,
    }


def write_spectra(path, spectra, detector, first_direction, max_lag):
    """Write the per-integration, per-channel spectra and gains to an ASDF file."""
    import asdf

    tree = {
        "detector": detector,
        "first_direction": int(first_direction),
        "max_lag": int(max_lag),
        "integrations": [
            {
                "channels": [
                    {
                        "freq": ch["freq"].astype(np.float32),
                        "spectrum": ch["spectrum"],
                        "smooth": ch["smooth"],
                        "gain": ch["gain"],
                        "amplitude": ch["amplitude"],
                        "alpha": ch["alpha"],
                        "white": ch["white"],
                        "corner_period": ch["corner_period"],
                    }
                    for ch in integration
                ]
            }
            for integration in spectra
        ],
    }
    with asdf.AsdfFile(tree) as handle:
        handle.write_to(path)


def read_psd_prior(path):
    """Return the per-channel power-law slopes of the first integration in a spectra ASDF file (None where missing)."""
    import asdf

    with asdf.open(path, lazy_load=False, memmap=False) as handle:
        channels = handle.tree["integrations"][0]["channels"]
        alphas = [float(ch["alpha"]) for ch in channels]
    for k, alpha in enumerate(alphas):
        if not np.isfinite(alpha):
            log.warning(f"psd_prior has no slope for channel {k}; that channel's slope will be fitted")
            alphas[k] = None
    return alphas


def clean_difference(
    image,
    mask,
    fastaxis,
    slowaxis,
    fit_method="row_median",
    background_method="median",
    background_box_size=None,
    clip_sigma=0.0,
    iterations=1,
    flat=None,
    first_direction=1,
    channel_kwargs=None,
    fixed_background=None,
    even_odd_columns=False,
    column_offsets=False,
    anchor_mean=False,
    **filter_kwargs,
):
    """Clean one group-difference image.

    Follows ``_clean_one_image`` in the jwst step: optionally measure the
    background on the flat-fielded image, mask NaNs, fit and remove a background level, clip signal with a
    sigma from the lower half of the distribution, estimate the noise,
    re-clip on the cleaned image and re-estimate ``iterations`` times in
    total, then restore the background, flat and NaNs.

    Parameters
    ----------
    image : ndarray, shape (nrows, ncols)
        Difference image in the science frame.
    mask : ndarray of bool
        Background mask, True for usable pixels.  Updated in place.
    fastaxis, slowaxis : int
        Detector readout axis keywords.
    fit_method : str
        Key into `snowblind.stream_filters.FILTERS`.
    background_method : {'median', 'model', None}
        Passed to the jwst ``background_level`` helper.
    background_box_size : tuple of int or None
        Box size for ``background_method='model'``.
    clip_sigma : float
        Per-difference clipping level of the background-subtracted image;
        0 disables clipping and keeps the cleaning linear.
    iterations : int
        Number of noise estimates; the mask is re-clipped between them when
        ``clip_sigma`` is positive.
    flat : ndarray or None
        Flat field matching ``image``.
    first_direction : int
        Readout direction of channel 0 in the detector frame.
    channel_kwargs : list of dict or None
        Extra keyword arguments for the estimator, one dict per channel.
    fixed_background : ndarray or None
        Background to subtract instead of fitting one to this image.
    even_odd_columns, column_offsets, anchor_mean : bool
        See `estimate_noise`.

    Returns
    -------
    cleaned : ndarray of float32 or None
        Cleaned image, or None if the mask has no usable pixel.
    background : float or ndarray
        The background that was removed and restored.
    """
    bkg_sub, background, nan_pix = prepare_difference(
        image, mask, background_method, background_box_size, clip_sigma, flat, fixed_background
    )
    if bkg_sub is None:
        return None, None

    offsets = {"even_odd_columns": even_odd_columns, "column_offsets": column_offsets, "anchor_mean": anchor_mean}
    noise = estimate_noise(
        bkg_sub, mask, fastaxis, slowaxis, fit_method, first_direction, channel_kwargs, **offsets, **filter_kwargs
    )
    for _ in range(1, iterations):
        if clip_sigma > 0:
            clip_to_background(
                bkg_sub - noise, mask, sigma_lower=clip_sigma, sigma_upper=clip_sigma, lower_half_only=True
            )
        noise = estimate_noise(
            bkg_sub, mask, fastaxis, slowaxis, fit_method, first_direction, channel_kwargs, **offsets, **filter_kwargs
        )

    cleaned = bkg_sub - noise + background
    cleaned[nan_pix] = np.nan
    return cleaned.astype(np.float32), background


def even_odd_pattern(channel_image, channel_mask):
    """The alternating-column offset of one channel: +d on even columns, -d on odd, from the masked means."""
    masked = np.where(channel_mask, channel_image, np.nan)
    with np.errstate(all="ignore"):
        even = np.nanmean(masked[:, ::2])
        odd = np.nanmean(masked[:, 1::2])
    if not (np.isfinite(even) and np.isfinite(odd)):
        return np.zeros(channel_image.shape, dtype=np.float32)
    half = 0.5 * (even - odd)
    pattern = np.full(channel_image.shape, -half, dtype=np.float32)
    pattern[:, ::2] = half
    return pattern


def column_offset_pattern(channel_image, channel_mask, min_pixels=16):
    """The per-column offset of one channel: each column's masked mean, repeated down the column.

    Columns with fewer than ``min_pixels`` usable pixels get an offset of
    zero.  The mean is a fixed linear operation on the difference, so the
    per-difference estimates add up to the column mean of the rate image
    and nothing accumulates along the ramp.
    """
    masked = np.where(channel_mask, channel_image, 0.0)
    count = channel_mask.sum(axis=0)
    with np.errstate(all="ignore"):
        offset = masked.sum(axis=0) / count
    offset[count < min_pixels] = 0.0
    return np.broadcast_to(offset.astype(np.float32), channel_image.shape)


def estimate_noise(
    image,
    mask,
    fastaxis,
    slowaxis,
    fit_method,
    first_direction=1,
    channel_kwargs=None,
    even_odd_columns=False,
    column_offsets=False,
    anchor_mean=False,
    **filter_kwargs,
):
    """Estimate the 1/f noise of a background-subtracted image, channel by channel.

    Parameters
    ----------
    image : ndarray, shape (nrows, ncols)
        Background-subtracted difference image in the science frame.
    mask : ndarray of bool
        Background mask, True for usable pixels.
    fastaxis, slowaxis : int
        Detector readout axis keywords.
    fit_method : str
        Key into `snowblind.stream_filters.FILTERS`.
    first_direction : int
        Readout direction of channel 0 in the detector frame.
    channel_kwargs : list of dict or None
        Extra keyword arguments for the estimator, one dict per channel
        (the ``wiener`` method's per-channel gain).
    even_odd_columns : bool
        Also remove each channel's even-odd column offset, measured from
        the masked pixels of this difference.  A fixed linear operation,
        so it does not accumulate along the ramp.
    column_offsets : bool
        Also remove each column's masked mean (`column_offset_pattern`),
        after the even-odd offset and before the stream filter.  This is
        the per-column offset that drifts during an integration; it is not
        a readout-stream process and the stream filter leaves it alone.
    anchor_mean : bool
        Shift each channel's noise estimate so that its mean over the
        masked pixels equals the masked mean of the channel's residual.
        The Wiener filter already has that property (gain 1 at zero
        frequency); a median-based estimator does not, and on a crowded
        field, where the background distribution is skewed, a median sits
        below the mean and would raise the sky.

    Returns
    -------
    ndarray of float32
        Noise estimate for every pixel, in the science frame.
    """
    try:
        estimator = FILTERS[fit_method]
    except KeyError:
        raise ValueError(f"Unknown fit_method {fit_method!r}; choose from {sorted(FILTERS)}") from None

    image_det = readout.to_detector_frame(image, fastaxis, slowaxis)
    mask_det = readout.to_detector_frame(mask, fastaxis, slowaxis)
    noise_det = np.zeros(image_det.shape, dtype=np.float32)
    for channel, columns in enumerate(readout.channel_slices(image_det.shape[-1])):
        direction = readout.channel_direction(channel, first_direction)
        extra = {} if channel_kwargs is None else channel_kwargs[channel]
        block = np.ascontiguousarray(image_det[:, columns])
        block_mask = np.ascontiguousarray(mask_det[:, columns])
        alternating = even_odd_pattern(block, block_mask) if even_odd_columns else 0.0
        per_column = column_offset_pattern(block - alternating, block_mask) if column_offsets else 0.0
        fixed = alternating + per_column
        estimate = estimator(block - fixed, block_mask, direction=direction, **extra, **filter_kwargs)
        if anchor_mean and np.any(block_mask):
            estimate = estimate + (np.mean((block - fixed)[block_mask]) - np.mean(estimate[block_mask]))
        noise_det[:, columns] = fixed + estimate
    return np.ascontiguousarray(readout.to_science_frame(noise_det, fastaxis, slowaxis))

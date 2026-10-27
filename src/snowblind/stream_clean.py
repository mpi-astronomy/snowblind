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
from crds.core.exceptions import CrdsError
from scipy import ndimage
from jwst import datamodels
from jwst.lib.reffile_utils import get_subarray_model
from .step import Step
from stdatamodels.jwst.datamodels import dqflags

from . import readout
from ._jwst_imports import (
    background_level,
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
    notch_period as notch_gain_period,
    taper_high_frequencies,
    wiener_gain,
)
from .scene_mask import chisel_mask, spike_mask, tiled_sky
from .stream_filters import FILTERS, row_median_noise, wiener_noise

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

    The main options (see ``spec`` for the full list and defaults):
    ``fit_method`` picks the per-channel noise estimator (``wiener`` or
    ``row_median``); ``background_method`` and ``scene_model`` choose the
    background removed before the fit (``crds_background``, the default
    on WFSS exposures, scales the CRDS background reference once to the
    ramp); ``user_mask``, ``mask_dilation``
    and ``mask_spikes`` control which pixels count as background; ``clean_first_group`` and
    ``reset_calib`` extend the cleaning to group 1 (and the zeroframe)
    against zero or the reset-level prior; ``even_odd_columns``,
    ``column_offsets`` and ``anchor_mean`` remove per-channel offsets
    around the stream filter; ``save_mask``, ``save_noise`` and
    ``save_spectra`` write intermediate products.
    """

    class_alias = "stream_clean"
    fits_keyword = "S_STRCLN"
    fits_comment = "1/f cleaning: snowblind.StreamCleanStep"

    spec = """
        fit_method = option('wiener', 'row_median', default='wiener')  # noise estimator per channel
        background_method = option('median', 'integration_model', 'exposure_model', 'crds_background', None, default='median')  # background removed before fitting: a scalar median suits sky-limited fields, integration_model (a box model) only extended structure above its box, None darks; crds_background (the scaled CRDS background reference) is the default on WFSS exposures
        background_box_size = int_list(min=2, max=2, default=None)  # box size of the box-spline background (integration_model, exposure_model); default 32 by 32
        apply_flat_field = boolean(default=True)  # measure the background on the flat-fielded difference (the sky times the response); without a flat reference the raw difference is used
        reference_units = option('auto', 'raw', 'flat_fielded', default='auto')  # units of the CRDS background reference: raw (it carries the detector's pixel response, as the NIRISS files do), flat_fielded (as the NIRCam files are), or auto (decided from its pixel pattern against the flat)
        user_mask = string(default=None)  # path to a user-supplied background mask
        mask_dilation = integer(default=0)  # pixels by which to grow the masked (source) regions, for source wings
        mask_spikes = boolean(default=True)  # mask the column-aligned spikes of saturated stars along their extent (chisel mask only)
        allow_dispersion_along_stream = boolean(default=False)  # clean a grism exposure whose dispersion runs along the readout stream (NIRISS GR150R, NIRCam GRISMR), where faint trace shapes are read as noise; skipped otherwise
        clean_first_group = boolean(default=True)  # also clean group 1, against zero or against the reset prior; needed for reset_rampfit, harmless for OLS
        reset_calib = string(default=None)  # snowblind-reset-calib ASDF; with it the first read is cleaned against its reset-level prior, as reset_rampfit will use it
        scene_model = string(default=None)  # FITS image of the scene in DN/s, detector frame, before flat-fielding (e.g. the blotted mosaic); replaces the background model
        min_density = float(default=0.2)  # local background fraction below which the estimate tapers off
        first_direction = integer(default=1)  # readout direction of channel 0 in the detector frame, 1 or -1
        even_odd_columns = boolean(default=True)  # remove the per-channel even-odd column offset of each difference
        column_offsets = boolean(default=True)  # remove each column's masked mean from each difference (a per-column offset that drifts during the integration); turned off on WFSS exposures, where the traces run along the columns and would bias the means
        anchor_mean = boolean(default=True)  # set each channel's mean removed noise to the masked mean of the residual, so a robust estimator cannot bias the sky
        max_lag = integer(default=65536)  # autocorrelation length in pixel times for the noise spectrum; wiener only
        min_period = integer(default=300)  # pixel times (3 ms); the wiener gain is zero for shorter periods (None keeps all)
        save_mask = boolean(default=False)  # save the background mask
        save_noise = boolean(default=False)  # save the removed noise
        save_spectra = boolean(default=False)  # save per-channel spectra and gains; wiener only
        skip = boolean(default=False)
    """  # noqa: E501

    reference_file_types = ["flat", "bkg"]
    # fetched in process, where an exposure type without a flat, such as a dark, is handled.  Same for WFSS
    # exposures that need bkg.
    prefetch_references = False

    def process(self, input_data):
        """Clean the group data of a ramp.

        The input is copied, the scene mask is built, every group difference
        is cleaned with `clean_ramp` and the optional
        mask, noise and spectra products are saved.  Unsupported input (see
        `_unsupported_reason`) or a mask whose shape does not match the data
        leaves the copy unchanged with the step marked SKIPPED.

        Parameters
        ----------
        input_data : str or `~stdatamodels.jwst.datamodels.RampModel`
            Group-level data with ``groupdq`` populated; a filename is
            opened with `jwst.datamodels.open`.

        Returns
        -------
        `~stdatamodels.jwst.datamodels.RampModel`
            Cleaned copy of the input, with ``data`` (and ``zeroframe`` when
            it was cleaned) replaced and ``meta.cal_step.stream_clean`` set
            to COMPLETE, or the unchanged copy marked SKIPPED.
        """
        with datamodels.open(input_data) as input_model:
            result = input_model.copy()

        if self.first_direction not in (1, -1):
            raise ValueError(f"first_direction must be 1 or -1, got {self.first_direction}")

        reason = _unsupported_reason(result)
        if reason is not None:
            log.warning(f"{reason}; skipping stream_clean")
            self.set_cal_step(result, "SKIPPED")
            return result
        grism = dispersion_along_stream(result)
        if grism is not None and not self.allow_dispersion_along_stream:
            log.warning(
                f"{grism} disperses along the readout stream: a trace is a feature of a few hundred (NIRISS) to over a "
                "thousand (NIRCam) clocks there, with its continuum shape and emission lines at the scales the Wiener "
                "gain passes, so the spectral shape of "
                "faint traces and trace wings under the scene mask's threshold would be read as 1/f noise and removed; "
                "skipping stream_clean (allow_dispersion_along_stream=True overrides, at that risk)"
            )
            self.set_cal_step(result, "SKIPPED")
            return result
        if str(result.meta.exposure.type).upper() in WFSS_EXPOSURE_TYPES and self.background_method == "median":
            # the dispersed background is a smooth function of position that no scalar background can model
            self.background_method = "crds_background"
            log.debug("WFSS exposure: background_method set to 'crds_background'")
        if str(result.meta.exposure.type).upper() in WFSS_EXPOSURE_TYPES and self.column_offsets:
            # the traces run along the axis the offsets are measured on, so each column's masked mean carries
            # the outskirts of its detected traces and the whole of its undetected ones, a bias that would be
            # removed from every trace in the column
            self.column_offsets = False
            log.debug("WFSS exposure: column_offsets turned off")
        self._stripe_axes = (1, 0)
        self._notch_period = None
        if str(result.meta.exposure.type).upper() in WFSS_EXPOSURE_TYPES:
            # the scene mask destripes the draft rate with row and column medians before detecting; a trace along
            # more than half of a column (or row) would set that median and vanish, so only the stream's axis is used
            self._stripe_axes = (1,) if abs(int(result.meta.subarray.fastaxis or 1)) == 1 else (0,)
            # the unmasked wings and faint traces along the columns sit at the same clock in every readout row, a
            # pattern at the row frequency that the Wiener gain would pass as noise; a narrow notch there removes it
            det_shape = readout.to_detector_frame(
                np.zeros(result.data.shape[-2:], dtype=np.uint8), result.meta.subarray.fastaxis, result.meta.subarray.slowaxis
            ).shape
            self._notch_period = float(readout.channel_width(det_shape[1]) + readout.LINE_OVERHEAD)
            log.debug(f"WFSS exposure: the Wiener gain is notched at the readout row period of {self._notch_period:.0f} clocks")
            log.info(
                "A dispersed exposure: the background is the scaled CRDS background reference, and the traces along the "
                "columns are kept out of the noise estimate (no per-column offsets, the readout row period notched)"
            )

        flat = None
        flat_unusable = None
        if self.apply_flat_field:
            try:
                flat_filename = self.get_reference_file(result, "flat")
            except CrdsError as err:
                log.info(f"CRDS has no flat for this exposure ({err})")
                flat_filename = None
            if flat_filename in (None, "", "N/A"):
                log.info("No flat reference for this exposure; the background is measured on the raw difference")
            else:
                log.info(f"Using FLAT reference file {flat_filename}")
                flat = read_flat_file(result, flat_filename)
                flat_unusable = unusable_in_flat(result, flat_filename)
                log.debug(
                    f"The flat marks {100 * flat_unusable.mean():.2f} percent of pixels unusable; "
                    "they are kept out of the background"
                )

        self._scene = None
        self._scene_flat = None
        if self.scene_model is not None:
            self._scene = read_scene_model(self.scene_model, result.data.shape[-2:])
            log.info(
                f"Scene model {self.scene_model}: median {np.nanmedian(self._scene):.4g} DN/s, "
                f"{100 * np.mean(~np.isfinite(self._scene)):.2f} percent without coverage"
            )
            # the scene is in raw DN/s; where a flat is in use the backgrounds are in flat-fielded units
            self._scene_flat = self._scene
            if flat is not None:
                with np.errstate(invalid="ignore", divide="ignore"):
                    self._scene_flat = np.where(np.isfinite(flat) & (flat > 0), self._scene / flat, np.nan)
        self._reference = None
        self._reference_scale = None
        if self.background_method == "crds_background":
            if self.scene_model is not None:
                raise ValueError("background_method='crds_background' and scene_model are both scene models; give one of them")
            try:
                bkg_filename = self.get_reference_file(result, "bkg")
            except CrdsError as err:
                raise ValueError(
                    f"background_method='crds_background' needs the CRDS background reference, and CRDS has none: {err}"
                ) from err
            if bkg_filename in (None, "", "N/A"):
                raise ValueError(
                    "background_method='crds_background' needs the CRDS background reference; CRDS has none for this exposure"
                )
            log.info(f"Using BKG reference file {bkg_filename}")
            with datamodels.open(bkg_filename) as bkg_model:
                self._reference = np.array(bkg_model.data, dtype=np.float64)

        background_mask, mask_model = self._scene_mask(result, flat)
        if background_mask.shape[-2:] != result.data.shape[-2:]:
            log.warning(
                f"Mask shape {background_mask.shape} does not match data shape "
                f"{result.data.shape}; skipping stream_clean"
            )
            self.set_cal_step(result, "SKIPPED")
            return result
        background_mask = exclude_reference_border(background_mask)
        if flat_unusable is not None:
            background_mask = background_mask & ~flat_unusable
        if self.mask_dilation > 0:
            background_mask = dilate_sources(background_mask, self.mask_dilation)

        log.info(
            f"Cleaning {result.meta.filename} with "
            + ("the Wiener filter along the readout stream" if self.fit_method == "wiener" else "row medians")
        )
        cleaned, spectra = self._clean_with_mask(result, background_mask, flat)

        if self.save_spectra and spectra is not None:
            spectra_path = self.make_output_path(basepath=result.meta.filename, suffix="spectra", ext="asdf")
            write_spectra(spectra_path, spectra, result.meta.instrument.detector, self.first_direction, self.max_lag)
            log.info(f"Saved spectra to {spectra_path}")

        if self.save_noise:
            noise_model = make_intermediate_model(result, cleaned - result.data)
            self.save_model(noise_model, suffix="noise", force=True)
            noise_model.close()
        if mask_model is not None:
            # the mask as used: border excluded, flat's unusable pixels out, sources dilated
            mask_model.data = background_mask.astype(mask_model.data.dtype)
            self.save_model(mask_model, suffix="mask", force=True)
            mask_model.close()

        result.data = cleaned
        if self._cleaned_zeroframe is not None:
            result.zeroframe = self._cleaned_zeroframe
        self.set_cal_step(result, "COMPLETE")
        return result

    def _clean_with_mask(self, result, background_mask, flat):
        """Run `clean_ramp` with the step's parameters and a given background mask.

        Also builds the reset-level prior (with ``reset_calib``), the fixed
        scene background in DN per group (with ``scene_model``) and takes a
        copy of the zeroframe to clean alongside group 1; the cleaned
        zeroframe is kept on ``self._cleaned_zeroframe`` (None when none).

        Parameters
        ----------
        result : `~stdatamodels.jwst.datamodels.RampModel`
            The model being cleaned; supplies ``data``, ``groupdq``,
            ``zeroframe`` and the readout and exposure metadata.
        background_mask : ndarray of bool, shape (nrows, ncols) or (nints, nrows, ncols)
            Scene mask, True for background pixels.
        flat : ndarray or None, shape (nrows, ncols)
            Flat field from `read_flat_file`, or None when
            ``apply_flat_field`` is off.
        Returns
        -------
        cleaned : ndarray of float32, shape (nints, ngroups, nrows, ncols)
            Cleaned group data in DN.
        spectra : list or None
            Per-integration lists of per-channel dicts from `channel_gain`
            (``wiener`` only; None otherwise).
        """
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
            log.info(f"The first read is cleaned against the reset level of {self.reset_calib}")
        scene_background = None
        if self._scene_flat is not None:
            # the scene in DN per group, in the units the backgrounds are in (flat-fielded when a flat is in use);
            # pixels without coverage get zero and are outside the mask
            scene_background = (
                np.where(np.isfinite(self._scene_flat), self._scene_flat, 0.0) * result.meta.exposure.group_time
            ).astype(np.float32)
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
            clean_first_group=self.clean_first_group,
            flat=flat,
            first_direction=self.first_direction,
            min_density=self.min_density,
            max_lag=self.max_lag,
            min_period=self.min_period,
            notch_period=self._notch_period,
            return_spectra=True,
            first_group_scale=first_group_time_ratio(result),
            even_odd_columns=self.even_odd_columns,
            column_offsets=self.column_offsets,
            anchor_mean=self.anchor_mean,
        )

    def _fit_reference(self, model, background_mask, flat):
        """Scale the background reference once to this exposure and make it the scene model.

        The scale is the median ratio of the ramp's own least-squares slope
        image (raw DN/s, so no flat is needed) to the reference over the
        background mask, through `reference_scale`.  One number for the
        whole exposure: each group difference then gets the same scaled
        reference, scaled only by its time, so nothing is fitted per
        difference and nothing random-walks along the ramp.

        Parameters
        ----------
        model : `~stdatamodels.jwst.datamodels.RampModel`
            The ramp; ``data``, ``groupdq`` and ``meta.exposure.group_time``
            are read.
        background_mask : ndarray of bool, shape (nrows, ncols)
            True for background pixels.
        flat : ndarray or None, shape (nrows, ncols)
            Flat field in use, or None.  A flat-fielded reference (the
            NIRCam ones) is multiplied by it first, through
            `reference_in_raw_units`, and the scaled reference is divided by
            it for the flat-fielded paths.
        """
        data, groupdq = np.asarray(model.data), np.asarray(model.groupdq)
        draft = np.nanmean([slope_per_group(data[i], groupdq[i]) for i in range(data.shape[0])], axis=0)
        draft = draft / (model.meta.exposure.group_time or 1.0)
        mask = np.asarray(background_mask, dtype=bool)
        if mask.ndim == 3:
            mask = mask.all(axis=0)
        reference, carries_flat = reference_in_raw_units(self._reference, flat, self.reference_units)
        if flat is not None:
            if self.reference_units == "auto":
                correlation = reference_flat_correlation(self._reference, flat)
                log.debug(f"Pixel-scale structure of the background reference against the flat's: correlation {correlation:.2f}")
                log.info(
                    "The background reference carries the detector's pixel response and is used as it is" if carries_flat
                    else "The background reference is flat-fielded; it is multiplied by the flat before scaling"
                )
                if not np.isfinite(correlation) or 0.3 <= correlation <= 0.7:
                    log.warning(
                        "That correlation does not settle whether the background reference is flat-fielded (the NIRISS "
                        "files give 0.8 to 0.97 against their own flats, the NIRCam files 0); set reference_units to "
                        "'raw' or 'flat_fielded' to decide"
                    )
            else:
                log.info(f"The background reference is taken as {self.reference_units.replace('_', '-')} (reference_units)")
        scale = reference_scale(draft, reference, mask)
        self._reference_scale = scale
        self._scene = (scale * reference).astype(np.float64)
        self._scene_flat = self._scene
        if flat is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                self._scene_flat = np.where(np.isfinite(flat) & (flat > 0), self._scene / flat, np.nan)
        log.info(f"Background reference scaled to this exposure by {scale:.4g} (one scale for every group difference)")

    def _chisel_scene_mask(self, rate, scene=None):
        """The chisel scene mask of a rate image, with the scene model and the spike mask applied.

        Parameters
        ----------
        rate : ndarray, shape (nrows, ncols)
            Draft rate image in DN/s, NaN where unusable.  With a scene
            model the mask is built on the residual against it, since the
            mask then only has to cover where the model is wrong, and
            pixels the model does not cover are masked.
        scene : ndarray or None, shape (nrows, ncols)
            The scene model in the units of ``rate`` (flat-fielded when
            ``rate`` is), or None for no scene model.

        Returns
        -------
        ndarray of bool, shape (nrows, ncols)
            True for background pixels.
        """
        rate = np.array(rate, dtype=np.float64)
        if scene is not None:
            rate = rate - scene
        background_mask = chisel_mask(rate, stripe_axes=getattr(self, "_stripe_axes", (1, 0)))
        if scene is not None:
            background_mask &= np.isfinite(scene)
        log.debug(f"Scene mask (chisel): {100 * (~background_mask).mean():.2f} percent of pixels masked")
        if self.mask_spikes:
            background_mask, nstars = spike_mask(rate, background_mask)
            log.debug(f"Spike mask: {nstars} bright stars, {100 * (~background_mask).mean():.2f} percent of pixels masked")
            self._spike_stars = nstars
        return background_mask

    def _scene_mask(self, model, flat):
        """Build the background mask with the jwst helpers.

        With ``user_mask`` set the mask is read from that file.  Otherwise a
        draft rate image is made by running the ramp fit step and the mask
        is derived from it.

        Parameters
        ----------
        model : `~stdatamodels.jwst.datamodels.RampModel`
            The group-level model; the draft rate image is made from it, or
            it stands in for the image model when ``user_mask`` is given.
        flat : ndarray or None, shape (nrows, ncols)
            Flat field handed to the jwst rate-image helper, or None.

        Returns
        -------
        background_mask : ndarray of bool, shape (nrows, ncols)
            True for background pixels.
        mask_model : `~stdatamodels.jwst.datamodels.JwstDataModel` or None
            Intermediate model holding the mask when ``save_mask`` is set,
            else None.
        """
        if self.user_mask is None:
            image_model = make_processed_rate_image(
                model,
                True,
                self.input_dir,
                model.meta.exposure.type,
                False,
                flat,
                self.background_method,
            )
        else:
            image_model = model

        if self.user_mask is None:
            rate = np.array(image_model.data, dtype=np.float64)
            if hasattr(image_model, "dq") and image_model.dq is not None:
                rate[(image_model.dq & dqflags.pixel["DO_NOT_USE"]) > 0] = np.nan
            background_mask = self._chisel_scene_mask(rate, self._scene_flat)
            if self._reference is not None:
                self._fit_reference(model, background_mask, flat)
                background_mask = self._chisel_scene_mask(rate, self._scene_flat)
            if self.background_method == "median" and self._scene is None:
                low, high = background_structure(rate, background_mask)
                if high - low > 0.5:
                    # a galaxy across the frame, a dispersed background, or heavy crowding; flat sky fields sit within 0.15
                    log.warning(
                        f"The unmasked background has large-scale structure: its 32-pixel tiled sky runs from {low:.2f} to "
                        f"{high:.2f} times the median.  A scalar median background hands that structure to the noise "
                        "estimate, which removes it; for extended emission use background_method='integration_model', "
                        "for a dispersed exposure 'crds_background'"
                    )
            stars = getattr(self, "_spike_stars", 0)
            log.info(
                f"Background mask: {100 * (~background_mask).mean():.1f} percent of pixels are sources"
                + (f", the diffraction spikes of {stars} bright star{'s' if stars != 1 else ''} among them" if stars else "")
            )
            mask_model = make_intermediate_model(image_model, background_mask) if self.save_mask else None
        else:
            # the jwst helper reads the user's mask file; the sigma-clip arguments do not apply to a given mask
            background_mask, mask_model = make_scene_mask(self.user_mask, image_model, False, 2.0, False, True, self.save_mask)
            if self._reference is not None:
                self._fit_reference(model, background_mask, flat)
        if image_model is not model:
            image_model.close()
        return background_mask, mask_model


def unusable_in_flat(model, flat_filename):
    """Pixels the flat reference marks DO_NOT_USE or leaves at zero or NaN.

    The jwst flat reader used for the flat itself replaces such pixels by
    a smoothed local value, so that dividing by the flat is harmless there;
    but the data under them are not sky (the NIRISS occulting spots are
    flagged only here, at a tenth of the sky, and no other DQ marks them),
    and to a linear noise estimate a dark patch is slow 1/f noise as much
    as a bright source.  The step keeps these pixels out of the background.

    Parameters
    ----------
    model : `~stdatamodels.jwst.datamodels.RampModel`
        The data; the flat is cut to its subarray when their shapes differ.
    flat_filename : str
        Path of the flat reference file.

    Returns
    -------
    ndarray of bool, shape (nrows, ncols)
        True where the flat is unusable.
    """
    with datamodels.FlatModel(flat_filename) as flat_model:
        if flat_model.data.shape != model.data.shape[-2:]:
            flat_model = get_subarray_model(model, flat_model)
        data = np.asarray(flat_model.data, dtype=np.float64)
        dq = np.asarray(flat_model.dq)
    return ((dq & dqflags.pixel["DO_NOT_USE"]) > 0) | ~np.isfinite(data) | (data <= 0)


def reference_flat_correlation(reference, flat, size=15):
    """Correlation of the pixel-scale structure of a background reference image with the flat field's.

    The CRDS WFSS background references are not all in the same units:
    the NIRISS ones are stacks of rate images and carry the detector's
    pixel response, the NIRCam ones are flat-fielded and smooth, and
    neither the files nor their documentation say which.  The pixel-scale
    structure of each image (the image over its local mean on ``size``
    pixels, minus one) tells them apart: its correlation with the flat's
    is 0.8 to 0.97 for the NIRISS files against the flats CRDS selects for
    them and within 0.03 of zero for the NIRCam files.

    Parameters
    ----------
    reference : ndarray, shape (nrows, ncols)
        Background reference image.
    flat : ndarray, shape (nrows, ncols)
        Flat field for the same detector and filter.
    size : int
        Side of the local mean, in pixels.

    Returns
    -------
    float
        Pearson correlation of the two structure images over the usable
        pixels, or NaN when fewer than 10000 pixels are usable or either
        image has no structure.
    """
    reference = np.asarray(reference, dtype=np.float64)
    flat = np.asarray(flat, dtype=np.float64)
    usable = np.isfinite(reference) & np.isfinite(flat) & (flat > 0.5)
    usable &= reference > 0.2 * np.nanmedian(reference[usable]) if usable.any() else False
    if usable.sum() < 10000:
        return float("nan")
    filled_ref = np.where(usable, reference, np.nanmedian(reference[usable]))
    filled_flat = np.where(usable, flat, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ref_structure = filled_ref / ndimage.uniform_filter(filled_ref, size) - 1.0
        flat_structure = filled_flat / ndimage.uniform_filter(filled_flat, size) - 1.0
    x, y = ref_structure[usable], flat_structure[usable]
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def reference_carries_flat(reference, flat, size=15, threshold=0.5):
    """Whether a background reference image carries the flat field's pixel-to-pixel pattern.

    Parameters
    ----------
    reference : ndarray, shape (nrows, ncols)
        Background reference image.
    flat : ndarray, shape (nrows, ncols)
        Flat field for the same detector and filter.
    size : int
        Side of the local mean of `reference_flat_correlation`, in pixels.
    threshold : float
        Correlation above which the reference is taken to carry the flat.

    Returns
    -------
    bool
        True when `reference_flat_correlation` exceeds ``threshold``.
    """
    correlation = reference_flat_correlation(reference, flat, size)
    return bool(np.isfinite(correlation) and correlation > threshold)


def reference_in_raw_units(reference, flat, units="auto"):
    """A background reference in the raw detector units of a group difference.

    A reference that already carries the flat's pixel pattern is returned
    as it is; a flat-fielded one is multiplied by the flat, so that the
    sky it models has the response the raw differences have (where the
    flat is not finite or not positive the reference is left as it is).

    Parameters
    ----------
    reference : ndarray, shape (nrows, ncols)
        Background reference image.
    flat : ndarray or None, shape (nrows, ncols)
        Flat field in use, or None, in which case the reference is
        returned unchanged.
    units : {'auto', 'raw', 'flat_fielded'}
        What the reference is in: ``raw`` leaves it alone, ``flat_fielded``
        multiplies it by the flat, ``auto`` decides with
        `reference_carries_flat`.

    Returns
    -------
    raw : ndarray of float64, shape (nrows, ncols)
        The reference in raw units.
    carries_flat : bool
        Whether the reference carried the flat already (True also when no
        flat is given, since nothing was done).
    """
    reference = np.asarray(reference, dtype=np.float64)
    if flat is None or units == "raw":
        return reference, True
    if units == "auto" and reference_carries_flat(reference, flat):
        return reference, True
    flat = np.asarray(flat, dtype=np.float64)
    gain = np.where(np.isfinite(flat) & (flat > 0), flat, 1.0)
    return reference * gain, False


def reference_scale(rate, reference, background_mask, floor=0.2, min_pixels=1000):
    """The factor that scales a background reference image to a rate image, from the masked background pixels.

    The median of the ratio ``rate / reference`` over the background pixels
    where the reference is above ``floor`` times its median, which keeps
    the ratio away from the reference's dark regions.  The jwst WFSS
    background step uses an error-weighted mean with percentile clipping
    for the same purpose; the median needs no error array and is as
    robust.

    Parameters
    ----------
    rate : ndarray, shape (nrows, ncols)
        Rate image in the units the scale is wanted in (raw DN/s here).
    reference : ndarray, shape (nrows, ncols)
        Background reference image, normalised or not.
    background_mask : ndarray of bool, shape (nrows, ncols)
        True for background pixels.
    floor : float
        Fraction of the reference's median below which a pixel is left out.
    min_pixels : int
        Fewest usable pixels for a scale.

    Returns
    -------
    float
        The scale.

    Raises
    ------
    ValueError
        If fewer than ``min_pixels`` pixels are usable.
    """
    reference = np.asarray(reference, dtype=np.float64)
    usable = np.asarray(background_mask, dtype=bool) & np.isfinite(rate) & np.isfinite(reference)
    level = np.nanmedian(reference[usable]) if usable.any() else np.nan
    usable &= reference > floor * level
    if usable.sum() < min_pixels:
        raise ValueError(f"Only {int(usable.sum())} background pixels to scale the background reference on; {min_pixels} needed")
    return float(np.median(rate[usable] / reference[usable]))


def background_structure(rate, background_mask, box=32, percentiles=(1.0, 99.0)):
    """How much large-scale structure the unmasked background holds, as a range relative to its median.

    A scalar median background assumes the background is flat.  Extended
    emission that the scene mask does not cover, such as a galaxy filling
    half the frame, is a gradient of tens of percent of the sky, and under
    a scalar background the stream filter removes it as slow noise.  The
    tiled sky of the masked rate on ``box``-pixel tiles, away from the
    reference border, measures that structure.

    Parameters
    ----------
    rate : ndarray, shape (nrows, ncols)
        Draft rate image in DN/s, NaN where unusable.
    background_mask : ndarray of bool, shape (nrows, ncols)
        True for background pixels.
    box : int
        Tile size in pixels of the sky model.
    percentiles : tuple of float
        The low and high percentiles of the tiled sky that bound the range.

    Returns
    -------
    low : float
        The low percentile of the tiled sky divided by its median.
    high : float
        The high percentile of the tiled sky divided by its median.
    """
    sky = tiled_sky(rate, background_mask, box)
    inner = sky[readout.NREFPIX:-readout.NREFPIX, readout.NREFPIX:-readout.NREFPIX]
    inner = inner[np.isfinite(inner)]
    if inner.size == 0:
        return 1.0, 1.0
    level = float(np.median(inner))
    if not np.isfinite(level) or level == 0.0:
        return 1.0, 1.0
    low, high = np.percentile(inner, percentiles)
    return float(low / level), float(high / level)


def read_scene_model(path, shape):
    """Read a scene model (DN/s, detector frame, before flat-fielding) from a FITS file.

    The image is taken from the SCI extension when present, else from the
    primary HDU.  Non-finite pixels mean no coverage.

    Parameters
    ----------
    path : str
        FITS file holding the scene image.
    shape : tuple of int
        Expected image shape ``(nrows, ncols)``; a mismatch raises
        ``ValueError``.

    Returns
    -------
    ndarray of float32, shape (nrows, ncols)
        The scene in DN/s.
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

    Parameters
    ----------
    background_mask : ndarray of bool, shape (..., nrows, ncols)
        Scene mask, True for background pixels; not modified.
    width : int
        Width of the border in pixels on each of the four sides; default
        ``readout.NREFPIX``.  Zero leaves the mask unchanged.

    Returns
    -------
    ndarray of bool, same shape as ``background_mask``
        Copy of the mask with the border set to False.
    """
    mask = np.array(background_mask, dtype=bool, copy=True)
    if width > 0:
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

    Parameters
    ----------
    background_mask : ndarray of bool, shape (nrows, ncols) or (nints, nrows, ncols)
        Scene mask, True for background pixels; a 3-D mask is dilated
        plane by plane.
    pixels : int
        Number of binary-dilation iterations with a 4-connected structuring
        element, i.e. the number of pixels the masked regions grow by.

    Returns
    -------
    ndarray of bool, same shape as ``background_mask``
        The mask with the grown source regions set to False.
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

    Parameters
    ----------
    model : `~stdatamodels.jwst.datamodels.RampModel`
        Supplies ``meta.exposure.nframes``, ``frame_time`` and
        ``group_time``.

    Returns
    -------
    float
        The ratio, or 1.0 when the frame time or group time is missing or
        zero.
    """
    nframes = model.meta.exposure.nframes or 1
    frame_time = model.meta.exposure.frame_time
    group_time = model.meta.exposure.group_time
    if not frame_time or not group_time:
        return 1.0
    return float(frame_time * (nframes + 1) / 2.0 / group_time)


WFSS_EXPOSURE_TYPES = ("NIS_WFSS", "NRC_WFSS")


def dispersion_along_stream(model):
    """The grism of a NIRISS or NIRCam WFSS exposure whose dispersion runs along the readout stream, or None.

    Each instrument has one grism of each orientation.  NIRISS reads along
    science columns and its GR150R disperses along the detector rows,
    which is that axis; NIRCam reads along rows and its GRISMR disperses
    along rows.  On those exposures a trace lies along the stream, where
    the Wiener filter reads structure on 30 to 300 pixel scales as noise.
    NIRISS GR150C and NIRCam GRISMC traces cross the stream in a few
    clocks and are safe.  Decided from the filter (NIRISS) and pupil
    (NIRCam) keywords, so it works at the Detector1 stage, before any WCS.

    Parameters
    ----------
    model : `~stdatamodels.jwst.datamodels.RampModel`
        The input; ``meta.instrument.name``, ``filter`` and ``pupil`` are read.

    Returns
    -------
    str or None
        The grism name when the dispersion runs along the stream, else None.
    """
    instrument = str(model.meta.instrument.name).upper()
    filter_name = str(model.meta.instrument.filter or "").upper()
    pupil = str(model.meta.instrument.pupil or "").upper()
    if instrument == "NIRISS" and filter_name.startswith("GR150R"):
        return f"NIRISS {filter_name}"
    if instrument == "NIRCAM" and pupil.startswith("GRISMR"):
        return f"NIRCam {pupil}"
    return None


def _unsupported_reason(model):
    """Return why the model cannot be cleaned, or None if it can.

    Supported input is NIRCam or NIRISS, group-level (4-D) data of the FULL
    subarray whose column count splits into ``readout.NOUTPUTS`` channels.

    Parameters
    ----------
    model : `~stdatamodels.jwst.datamodels.RampModel`
        The input model; its instrument name, data shape and subarray name
        are checked.

    Returns
    -------
    str or None
        A message naming the first failed check, or None when the model can
        be cleaned.
    """
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

    Parameters
    ----------
    data : ndarray, shape (nints, ngroups, nrows, ncols)
        Group data in the science frame, in DN.
    groupdq : ndarray of int, same shape as ``data``
        Group DQ flags.
    background_mask : ndarray of bool, shape (nrows, ncols) or (nints, nrows, ncols)
        Scene mask, True for background pixels; a 3-D mask supplies one
        plane per integration.
    clean_first_group : bool
        If True, group 0 is yielded as well (as itself or as the difference
        against ``first_group_reference``); otherwise the loop starts at
        group 1.
    first_group_reference : ndarray or None, shape (nrows, ncols)
        Image subtracted from group 0 when ``clean_first_group`` is set, in
        DN; None yields group 0 as it is.

    Yields
    ------
    i : int
        Integration index.
    j : int
        Group index of the later group of the difference.
    image : ndarray of float32, shape (nrows, ncols)
        The difference image in DN.
    mask : ndarray of bool, shape (nrows, ncols)
        Usable background pixels of this difference.
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
    clean_first_group=False,
    flat=None,
    first_direction=1,
    max_lag=65536,
    return_spectra=False,
    first_group_scale=1.0,
    min_period=None,
    notch_period=None,
    even_odd_columns=False,
    column_offsets=False,
    anchor_mean=False,
    first_group_reference=None,
    scene_background=None,
    zeroframe=None,
    zeroframe_scale=None,
    min_density=0.2,
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
    fastaxis : int
        Detector readout axis keyword ``meta.subarray.fastaxis``: the
        science-frame axis and direction of the fast readout, used to
        rotate the image into the detector frame.
    slowaxis : int
        Detector readout axis keyword ``meta.subarray.slowaxis``: the
        science-frame axis and direction of the slow readout.
    fit_method : str
        Key into `snowblind.stream_filters.FILTERS`.
    background_method : {'median', 'integration_model', 'exposure_model', 'crds_background', None}
        Background removed from each difference before the noise fit (see
        `clean_difference`).  ``integration_model`` and ``exposure_model``
        fit one model per integration or per exposure with
        `integration_backgrounds`; ``scene_background`` overrides this.
    background_box_size : tuple of int or None
        Box size for the ``model`` and per-integration backgrounds.
    flat : ndarray or None, shape (nrows, ncols)
        Flat field matching the images (see `clean_difference`).
    clean_first_group : bool
        If True, group 1 is cleaned as a difference against a plane of zeros.
    first_direction : int
        Readout direction of channel 0 in the detector frame.
    max_lag : int
        Autocorrelation length in clocks for the ``wiener`` spectrum.
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
    notch_period : float or None
        ``wiener`` only: period in clocks at which the gain is notched, the
        readout row length on a dispersed exposure, where unmasked traces
        along the columns repeat in every row; None for no notch.
    even_odd_columns : bool
        Remove each channel's even-odd column offset per difference (see
        `estimate_noise`).
    column_offsets : bool
        Remove each column's masked mean per difference (see
        `estimate_noise`).
    anchor_mean : bool
        Anchor each channel's noise estimate to the masked mean of the
        residual (see `estimate_noise`).
    first_group_reference : ndarray or None, shape (nrows, ncols)
        With ``clean_first_group``, the image group 0 is cleaned against
        (the reset-level prior), in DN; None cleans it against zero.
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
    min_density : float
        Local background fraction below which the ``wiener`` estimate blends
        toward the row median.

    Returns
    -------
    cleaned : ndarray of float32, shape (nints, ngroups, nrows, ncols)
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
            clean_first_group=clean_first_group,
            flat=flat,
            first_direction=first_direction,
            max_lag=max_lag,
            fixed_backgrounds=fixed,
            first_group_reference=first_group_reference,
            first_group_scale=first_group_scale,
            min_period=min_period,
            notch_period=notch_period,
        )
        gains = [[(ch["freq"], ch["gain"]) for ch in integration] for integration in spectra]
        for i in range(nints):
            summary = ", ".join(
                f"ch{k}: alpha={ch['alpha']:.2f} W={ch['white']:.3g} corner={ch['corner_period']:.0f} clocks"
                for k, ch in enumerate(spectra[i])
            )
            log.debug(f"Integration {i + 1} noise spectra: {summary}")

    removed_power = [0.0] * nints
    removed_count = [0] * nints
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
            flat=flat,
            first_direction=first_direction,
            channel_gains=None if gains is None else gains[i],
            fixed_background=_fixed_for(fixed, i, j, first_group_scale),
            even_odd_columns=even_odd_columns,
            column_offsets=column_offsets,
            anchor_mean=anchor_mean,
            min_density=min_density,
        )
        if cleaned_image is None:
            log.warning(f"No usable data in integration {i + 1}, group {j + 1}; difference left as is")
            cleaned_image = image
        else:
            removed = (image - cleaned_image)[mask & np.isfinite(image) & np.isfinite(cleaned_image)]
            removed_power[i] += float(np.sum(removed.astype(np.float64) ** 2))
            removed_count[i] += removed.size
        cleaned[i, j] = previous + cleaned_image
    for i in range(nints):
        if removed_count[i]:
            log.info(
                f"Integration {i + 1}: removed 1/f noise with an rms of "
                f"{np.sqrt(removed_power[i] / removed_count[i]):.2f} DN per group difference over the background pixels"
            )

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
                flat=flat,
                first_direction=first_direction,
                channel_gains=None if gains is None else gains[i],
                fixed_background=None if fixed is None else fixed[i] * scale,
                even_odd_columns=even_odd_columns,
                column_offsets=column_offsets,
                anchor_mean=anchor_mean,
                min_density=min_density,
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
    """The fixed background for one difference, or None when none is in use.

    Parameters
    ----------
    fixed : list of ndarray or None
        One background image per integration, in DN per group difference,
        from `integration_backgrounds` or the scene model; None when no
        fixed background is in use.
    integration : int
        Integration index into ``fixed``.
    group : int
        Group index of the difference; group 0 is the first group cleaned
        against zero or the reset prior.
    first_group_scale : float
        Factor applied to the background of group 0: the first group's mean
        time since reset over the group time.

    Returns
    -------
    ndarray or None
        ``fixed[integration]``, scaled by ``first_group_scale`` for group 0,
        or None.
    """
    if fixed is None:
        return None
    if group == 0:
        return fixed[integration] * first_group_scale
    return fixed[integration]


def prepare_difference(
    image, mask, background_method="median", background_box_size=None, flat=None, fixed_background=None
):
    """Mask NaNs and remove the background, in the raw DN of the difference.

    This is the part of ``_clean_one_image`` in the jwst step that comes
    before the noise fit.  ``mask`` is updated in place.  With
    ``fixed_background`` given (the ``integration_model`` method) that array
    is subtracted instead of fitting a background to this image.

    With ``flat`` the background is measured on the flat-fielded image, as
    the sky is smooth there and the detector response is not, but what is
    subtracted is that background times the flat, so that the residual
    handed to the noise fit stays in raw DN, where the 1/f noise is
    additive and does not pass through the quantum efficiency.  (The jwst
    step divides the whole image by the flat, fits in that space and
    multiplies back, which modulates the noise estimate by the flat.)  A
    ``fixed_background`` is in flat-fielded units when a flat is given.

    Parameters
    ----------
    image : ndarray, shape (nrows, ncols)
        Group-difference image in raw DN; copied, not modified.
    mask : ndarray of bool, shape (nrows, ncols)
        Background mask, True for usable pixels.  Updated in place: NaN
        pixels, pixels without a valid flat and clipped pixels are set to
        False.
    background_method : {'median', 'integration_model', 'crds_background', None}
        How the background is found: ``median`` or ``model`` are passed to
        the jwst ``background_level`` helper, ``integration_model``
        requires ``fixed_background``, and None (or ``'none'``) removes no
        background.
    background_box_size : tuple of int or None
        Box size of the box-spline background of ``integration_model`` and
        ``exposure_model``; None uses 32 by 32.
    flat : ndarray or None, shape (nrows, ncols)
        Flat field; pixels where it is not finite or not positive are
        treated as NaN.
    fixed_background : ndarray or float or None
        Background to subtract instead of fitting one, in DN per group
        difference (flat-fielded units when ``flat`` is given).

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
    if background_method in ("integration_model", "crds_background") and fixed_background is None:
        raise ValueError(f"background_method={background_method!r} needs its background as fixed_background")
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
    return bkg_sub, background, nan_pix


def slope_per_group(data_int, groupdq_int):
    """Least-squares slope of one integration's ramp per pixel, in DN per group.

    Pixels with any flagged group fall back to the mean of their usable
    differences, which is what a slope through two points reduces to.

    Parameters
    ----------
    data_int : ndarray, shape (ngroups, nrows, ncols)
        Group data of one integration, in DN.
    groupdq_int : ndarray of int, shape (ngroups, nrows, ncols)
        Group DQ flags of the same integration; any nonzero flag marks a
        group as unusable.

    Returns
    -------
    ndarray of float32, shape (nrows, ncols)
        Slope per pixel in DN per group; NaN where a flagged pixel has no
        usable difference.
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

    Parameters
    ----------
    image : ndarray, shape (nrows, ncols)
        Image to model; non-finite pixels are excluded.
    mask : ndarray of bool, shape (nrows, ncols)
        True for usable (background) pixels.
    box_size : tuple of int or None
        Box size ``(ny, nx)`` in pixels; None uses ``(32, 32)``.
    filter_size : int
        Size of the median filter applied to the grid of box values, in
        boxes.

    Returns
    -------
    ndarray of float32, shape (nrows, ncols)
        The interpolated background, in the units of ``image``; all zeros
        when no pixel is usable.
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

    Parameters
    ----------
    data : ndarray, shape (nints, ngroups, nrows, ncols)
        Group data in the science frame, in DN.
    groupdq : ndarray of int, same shape as ``data``
        Group DQ flags, used to exclude flagged groups from the slope.
    background_mask : ndarray of bool, shape (nrows, ncols) or (nints, nrows, ncols)
        Scene mask, True for background pixels; with ``per_exposure`` a 3-D
        mask is combined with a logical AND over integrations.
    background_box_size : tuple of int or None
        Box size for `masked_mean_background`; None uses its default.
    flat : ndarray or None, shape (nrows, ncols)
        Flat field; when given, the slope is divided by it before the model
        is fitted, so the model is in flat-fielded units.
    per_exposure : bool
        If True, fit one model to the mean slope of all integrations and
        return it for every integration.

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
        """Fit the masked-mean box model to one slope image.

        Parameters
        ----------
        image : ndarray, shape (nrows, ncols)
            Slope image in DN per group (flat-fielded when a flat is given).
        mask : ndarray of bool, shape (nrows, ncols)
            True for background pixels.

        Returns
        -------
        ndarray of float32, shape (nrows, ncols)
            `masked_mean_background` of ``image`` with the enclosing
            ``background_box_size``.
        """
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
    clean_first_group=False,
    flat=None,
    first_direction=1,
    max_lag=65536,
    fixed_backgrounds=None,
    first_group_reference=None,
    first_group_scale=1.0,
    min_period=None,
    notch_period=None,
):
    """First pass of the ``wiener`` method: the noise spectrum and gain of every channel.

    For every group difference the background-subtracted, clipped image is
    laid out on the clock grid channel by channel and its masked
    autocorrelation turned into a spectrum.  The spectra are median-combined
    over the differences of an integration (the noise is stationary, the
    scene residual is not) and passed to `channel_gain`.

    Parameters
    ----------
    data : ndarray, shape (nints, ngroups, nrows, ncols)
        Group data in the science frame, in DN.
    groupdq : ndarray of int, same shape as ``data``
        Group DQ flags.
    background_mask : ndarray of bool, shape (nrows, ncols) or (nints, nrows, ncols)
        Scene mask, True for background pixels.
    fastaxis : int
        Detector readout axis keyword ``meta.subarray.fastaxis``: the
        science-frame axis and direction of the fast readout, used to
        rotate the image into the detector frame.
    slowaxis : int
        Detector readout axis keyword ``meta.subarray.slowaxis``: the
        science-frame axis and direction of the slow readout.
    background_method : {'median', 'integration_model', 'crds_background', None}
        Background removed from each difference (see `prepare_difference`).
    background_box_size : tuple of int or None
        Box size of the box-spline background of ``integration_model`` and
        ``exposure_model``; None uses 32 by 32.
    clean_first_group : bool
        If True, group 0 (against zero or ``first_group_reference``) is
        included in the spectra.
    flat : ndarray or None, shape (nrows, ncols)
        Flat field matching the images.
    first_direction : int
        Readout direction of channel 0 in the detector frame, 1 or -1.
    max_lag : int
        Requested autocorrelation length in clocks; capped at a quarter of
        the stream length.
    fixed_backgrounds : list of ndarray or None
        One background per integration in DN per group difference, from
        `integration_backgrounds` or the scene model.
    first_group_reference : ndarray or None, shape (nrows, ncols)
        Image group 0 is differenced against when ``clean_first_group`` is
        set, in DN.
    first_group_scale : float
        Scale of the fixed background for group 0 (see `_fixed_for`).
    min_period : int or None
        Period in clocks below which the gain is tapered to zero (see
        `channel_gain`).
    notch_period : float or None
        Period in clocks at which the gain is notched (see `channel_gain`).

    Returns
    -------
    list
        One entry per integration, each a list of four dicts from
        `channel_gain` (`zero_gain` for a channel with no usable
        difference).
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
            image, mask, background_method, background_box_size, flat,
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
            channels.append(channel_gain(freq, median, min_period=min_period, notch_period=notch_period))
        results.append(channels)
    return results


def zero_gain(max_lag):
    """A channel entry that removes nothing, for channels with no usable data.

    Parameters
    ----------
    max_lag : int
        Autocorrelation length in clocks; the frequency grid is that of a
        ``2 * max_lag``-point real FFT.

    Returns
    -------
    dict
        Same keys as `channel_gain`: ``freq``, with ``spectrum``,
        ``smooth`` and ``gain`` all zero, ``amplitude`` and ``white`` 0.0
        and ``alpha`` and ``corner_period`` NaN.
    """
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


def channel_gain(freq, spectrum, alpha_bounds=(0.5, 1.5), fixed_alpha=None, min_period=None, notch_period=None):
    """Turn a channel's median spectrum into a Wiener gain.

    A power law plus white floor is fitted to the log-binned spectrum.  The
    gain uses the smoothed nonparametric spectrum where it is well measured
    and the fit at the lowest frequencies (leakage) and the highest five
    percent (where the smoothing has few bins).  The gain is forced to one
    at zero frequency so that the normalized convolution is well defined.

    Parameters
    ----------
    freq : ndarray, shape (nfreq,)
        Frequencies in cycles per clock, from zero to the Nyquist frequency.
    spectrum : ndarray, shape (nfreq,)
        The channel's median noise power spectrum on ``freq``.
    alpha_bounds : tuple of float
        Allowed range of the power-law slope in the fit.
    fixed_alpha : float or None
        If given, the slope is held at this value and only the amplitude
        and white floor are fitted.
    min_period : int or None
        Period in clocks below which the gain is ramped to zero
        (`taper_high_frequencies`); None keeps all frequencies.
    notch_period : float or None
        Period in clocks at which the gain is notched
        (`snowblind.noise_spectrum.notch_period`), the readout row length
        on a dispersed exposure; None for no notch.

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
    if notch_period is not None:
        gain = notch_gain_period(freq, gain, notch_period)
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
    """Write the per-integration, per-channel spectra and gains to an ASDF file.

    Parameters
    ----------
    path : str
        Output file path.
    spectra : list
        One entry per integration, each a list of per-channel dicts from
        `channel_gain`.
    detector : str
        Detector name stored in the tree.
    first_direction : int
        Readout direction of channel 0 stored in the tree.
    max_lag : int
        Autocorrelation length in clocks stored in the tree.
    """
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


def clean_difference(
    image,
    mask,
    fastaxis,
    slowaxis,
    fit_method="row_median",
    background_method="median",
    background_box_size=None,
    flat=None,
    first_direction=1,
    channel_gains=None,
    fixed_background=None,
    even_odd_columns=False,
    column_offsets=False,
    anchor_mean=False,
    min_density=0.2,
):
    """Clean one group-difference image.

    Follows ``_clean_one_image`` in the jwst step: optionally measure the
    background on the flat-fielded image, mask NaNs, fit and remove a
    background level, estimate the noise, then restore the background,
    flat and NaNs.  Nothing here is nonlinear in the data, so the same
    operation applies to every difference and nothing random-walks along
    the ramp.

    Parameters
    ----------
    image : ndarray, shape (nrows, ncols)
        Difference image in the science frame.
    mask : ndarray of bool, shape (nrows, ncols)
        Background mask, True for usable pixels.  Updated in place.
    fastaxis : int
        Detector readout axis keyword ``meta.subarray.fastaxis``: the
        science-frame axis and direction of the fast readout, used to
        rotate the image into the detector frame.
    slowaxis : int
        Detector readout axis keyword ``meta.subarray.slowaxis``: the
        science-frame axis and direction of the slow readout.
    fit_method : str
        Key into `snowblind.stream_filters.FILTERS`.
    background_method : {'median', 'integration_model', 'crds_background', None}
        Passed to `prepare_difference`; ``median`` and ``model`` go to the
        jwst ``background_level`` helper.
    background_box_size : tuple of int or None
        Box size of the box-spline background of ``integration_model`` and
        ``exposure_model``; None uses 32 by 32.
    flat : ndarray or None
        Flat field matching ``image``.
    first_direction : int
        Readout direction of channel 0 in the detector frame.
    channel_gains : list of tuple or None
        For the ``wiener`` method, the ``(freq, gain)`` of each channel from
        `channel_gain`; None for the other methods.
    fixed_background : ndarray or None
        Background to subtract instead of fitting one to this image.
    even_odd_columns : bool
        Remove each channel's even-odd column offset (see `estimate_noise`).
    column_offsets : bool
        Remove each column's masked mean (see `estimate_noise`).
    anchor_mean : bool
        Anchor each channel's noise estimate to the masked mean of the
        residual (see `estimate_noise`).
    min_density : float
        Local background fraction below which the ``wiener`` estimate blends
        toward the row median.

    Returns
    -------
    cleaned : ndarray of float32 or None
        Cleaned image, or None if the mask has no usable pixel.
    background : float or ndarray
        The background that was removed and restored.
    """
    bkg_sub, background, nan_pix = prepare_difference(
        image, mask, background_method, background_box_size, flat, fixed_background
    )
    if bkg_sub is None:
        return None, None

    noise = estimate_noise(
        bkg_sub, mask, fastaxis, slowaxis, fit_method, first_direction, channel_gains,
        even_odd_columns=even_odd_columns, column_offsets=column_offsets, anchor_mean=anchor_mean,
        min_density=min_density,
    )

    cleaned = bkg_sub - noise + background
    cleaned[nan_pix] = np.nan
    return cleaned.astype(np.float32), background


def even_odd_pattern(channel_image, channel_mask):
    """The alternating-column offset of one channel: +d on even columns, -d on odd, from the masked means.

    Parameters
    ----------
    channel_image : ndarray, shape (nrows, width)
        One channel's block of a background-subtracted difference, in DN.
    channel_mask : ndarray of bool, shape (nrows, width)
        True for usable pixels.

    Returns
    -------
    ndarray of float32, shape (nrows, width)
        Half the difference of the even and odd masked means, positive on
        even columns and negative on odd; all zeros when either mean is not
        finite.
    """
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

    Parameters
    ----------
    channel_image : ndarray, shape (nrows, width)
        One channel's block of a background-subtracted difference, in DN.
    channel_mask : ndarray of bool, shape (nrows, width)
        True for usable pixels.
    min_pixels : int
        Minimum number of usable pixels for a column to get a nonzero
        offset.

    Returns
    -------
    ndarray of float32, shape (nrows, width)
        Each column's masked mean broadcast down the column (a read-only
        view).
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
    channel_gains=None,
    even_odd_columns=False,
    column_offsets=False,
    anchor_mean=False,
    min_density=0.2,
):
    """Estimate the 1/f noise of a background-subtracted image, channel by channel.

    Parameters
    ----------
    image : ndarray, shape (nrows, ncols)
        Background-subtracted difference image in the science frame.
    mask : ndarray of bool, shape (nrows, ncols)
        Background mask, True for usable pixels.
    fastaxis : int
        Detector readout axis keyword ``meta.subarray.fastaxis``: the
        science-frame axis and direction of the fast readout, used to
        rotate the image into the detector frame.
    slowaxis : int
        Detector readout axis keyword ``meta.subarray.slowaxis``: the
        science-frame axis and direction of the slow readout.
    fit_method : str
        Key into `snowblind.stream_filters.FILTERS`.
    first_direction : int
        Readout direction of channel 0 in the detector frame.
    channel_gains : list of tuple or None
        For the ``wiener`` method, the ``(freq, gain)`` of each channel from
        `channel_gain`; None for the other methods.
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
    min_density : float
        Local background fraction below which the ``wiener`` estimate blends
        toward the row median.

    Returns
    -------
    ndarray of float32, shape (nrows, ncols)
        Noise estimate for every pixel in DN, in the science frame.
    """
    if fit_method not in FILTERS:
        raise ValueError(f"Unknown fit_method {fit_method!r}; choose from {sorted(FILTERS)}")

    image_det = readout.to_detector_frame(image, fastaxis, slowaxis)
    mask_det = readout.to_detector_frame(mask, fastaxis, slowaxis)
    noise_det = np.zeros(image_det.shape, dtype=np.float32)
    for channel, columns in enumerate(readout.channel_slices(image_det.shape[-1])):
        direction = readout.channel_direction(channel, first_direction)
        block = np.ascontiguousarray(image_det[:, columns])
        block_mask = np.ascontiguousarray(mask_det[:, columns])
        alternating = even_odd_pattern(block, block_mask) if even_odd_columns else 0.0
        per_column = column_offset_pattern(block - alternating, block_mask) if column_offsets else 0.0
        fixed = alternating + per_column
        residual = block - fixed
        if fit_method == "row_median":
            estimate = row_median_noise(residual, block_mask, direction)
        else:
            gain = None if channel_gains is None else channel_gains[channel]
            estimate = wiener_noise(residual, block_mask, direction, min_density=min_density, gain=gain)
        if anchor_mean and np.any(block_mask):
            estimate = estimate + (np.mean((block - fixed)[block_mask]) - np.mean(estimate[block_mask]))
        noise_det[:, columns] = fixed + estimate
    return np.ascontiguousarray(readout.to_science_frame(noise_det, fastaxis, slowaxis))

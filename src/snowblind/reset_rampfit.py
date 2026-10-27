"""Ramp fitting with the reset level as an extra resultant.

`ResetRampFitStep` builds the stcal ``RampData`` exactly as the jwst
``ramp_fit`` step does, then calls `snowblind.likely_fit_reset.likely_ramp_fit_reset`
instead of ``stcal.ramp_fitting.ramp_fit`` and assembles the rate and
rateints products with the jwst step's own helpers.  The two diagnostic
planes of the fit, ``used_reset`` and ``d0_resid``, are written to a
separate ``_resetfit.asdf`` product when ``save_opt`` is set; they are not
attached to the cube model.
"""
import logging

import asdf
import numpy as np
from jwst import datamodels
from jwst.ramp_fitting.ramp_fit_step import (
    create_image_model,
    create_integration_model,
    get_reference_file_subarrays,
)
from .step import Step, carry_cal_step
from stcal.ramp_fitting.ramp_fit import create_ramp_fit_class
from stdatamodels.jwst.datamodels import dqflags

from . import readout
from .likely_fit_reset import MIN_NGROUPS, likely_ramp_fit_reset, likely_ramp_fit_reset_blocks
from .reset_calib import (  # noqa: F401  superbias_difference is re-exported
    add_history,
    default_calibration,
    read_calibration,
    superbias_basename,
    superbias_difference,
)

__all__ = ["ResetRampFitStep", "write_resetfit"]

log = logging.getLogger(__name__)


def _span(values, fmt):
    """The range of ``values`` as "low to high" in the format ``fmt``, or the one value when they all agree.

    Parameters
    ----------
    values : sequence of float
        Per-channel values.
    fmt : str
        Format spec for one value, such as ``'+.2f'``.

    Returns
    -------
    str
        ``f"{low:{fmt}} to {high:{fmt}}"`` or ``f"{value:{fmt}}"``.
    """
    low, high = min(values), max(values)
    if f"{low:{fmt}}" == f"{high:{fmt}}":
        return f"{low:{fmt}}"
    return f"{low:{fmt}} to {high:{fmt}}"


class ResetRampFitStep(Step):
    """Fit ramps by maximum likelihood with a Gaussian prior on the reset level.

    Input is a `~stdatamodels.jwst.datamodels.RampModel` after the jump step
    (and after `StreamCleanStep` with ``clean_first_group`` when both are
    used).  Output is an `~stdatamodels.jwst.datamodels.ImageModel` and a
    `~stdatamodels.jwst.datamodels.CubeModel` as from the jwst ramp_fit step.
    ``meta.cal_step.ramp_fit`` is set to COMPLETE as well so that downstream
    steps treat the result as a rate product.

    The prior on the reset level comes from a calibration ASDF file written
    by ``snowblind-reset-calib`` (``reset_calib``), or from a scalar
    ``sigma_b``.  Main options: ``use_pedestal_mean`` takes the mean reset
    level map from the file; ``t1_scale`` and ``per_channel_timing`` set the
    first-read timing; ``anchor_offset_to_sky`` and
    ``anchor_timing_to_stars`` measure the reset offset and first-read scale
    per output channel from the exposure itself; ``save_opt`` writes the
    ``used_reset`` and ``d0_resid`` diagnostics.  Fewer than ``MIN_NGROUPS``
    groups gives no rate products, as in the jwst step.
    """

    class_alias = "reset_rampfit"
    fits_keyword = "S_RSTRMP"
    fits_comment = "Reset ramp fit: snowblind.ResetRampFitStep"

    spec = """
        reset_calib = string(default=None)  # ASDF from snowblind-reset-calib; without it and without sigma_b, the calibration packaged for the detector is used
        sigma_b = float(default=None)  # DN; scalar reset scatter, overrides the calibration file
        t1_scale = float(default=None)  # fraction of the nominal first interval the first read integrates, for this exposure; overrides the file and its rescaling
        per_channel_timing = boolean(default=True)  # use the file's per-channel t1_scale and reset_offset when present
        use_pedestal_mean = boolean(default=False)  # prior mean of the reset level from the calibration file's pedestal_mean map
        prior_highpass = boolean(default=True)  # remove the pedestal map's row, even-odd and column structure, as stream_clean removes it from the first read
        reset_offset = float(default=0.0)  # DN added to the prior mean of the reset level, on top of the file's or the anchored value
        anchor_offset_to_sky = boolean(default=True)  # measure the reset offset per channel from this exposure's own faint pixels instead of the file's
        anchor_max_signal = float(default=50.0)  # DN; pixels whose first-resultant signal is below this count as sky for the anchor
        anchor_min_pixels = integer(default=1000)  # fewer sky pixels than this in a channel keeps the file's offset there
        anchor_timing_to_stars = boolean(default=False)  # also fit the first-read scale per channel from this exposure's bright unsaturated pixels (needs many bright pixels)
        anchor_min_bright = integer(default=500)  # fewer bright pixels than this in a channel keeps the file's first-read scale there
        use_zeroframe = boolean(default=True)  # use the zeroframe when present
        rejection_threshold = float(default=None)  # sigma threshold of the likelihood jump search; None for stcal default
        save_opt = boolean(default=False)  # save used_reset and d0_resid as a _resetfit ASDF product
        # accepted so that the step fits the ramp_fit slot of calwebb_detector1, whose CRDS parameters set them; not used
        algorithm = string(default='LIKELY')  # the fit is always the reset-anchored likelihood fit
        int_name = string(default='')
        opt_name = string(default='')
        suppress_one_group = boolean(default=False)  # one-group pixels are what the reset point is for; never suppressed
        firstgroup = integer(default=None)  # not implemented; a value given is reported and ignored
        lastgroup = integer(default=None)  # not implemented; a value given is reported and ignored
        maximum_cores = string(default='1')  # the fit runs in one process
        as_hook = boolean(default=False)  # return the rate image alone, as a post-hook of a skipped ramp_fit must, and write the rateints product
        skip = boolean(default=False)
    """  # noqa: E501

    reference_file_types = ["readnoise", "gain", "superbias"]

    def process(self, step_input):
        """Fit the ramps of a group-level product.

        Parameters
        ----------
        step_input : str or `~stdatamodels.jwst.datamodels.RampModel`
            A ``_jump`` ramp or its path; data shape
            (nints, ngroups, nrows, ncols) in DN.

        Returns
        -------
        out_model : `~stdatamodels.jwst.datamodels.ImageModel` or None
            The rate product in DN/s, averaged over integrations.  With
            ``save_results`` the two products are written with the
            ``_rate`` and ``_rateints`` suffixes.  With ``as_hook`` only
            this model is returned and the ``_rateints`` product is written.
        int_model : `~stdatamodels.jwst.datamodels.CubeModel` or None
            The per-integration rate product in DN/s.  Both are None when
            the ramp has fewer than ``MIN_NGROUPS`` groups.
        """
        result = self.prepare_output(step_input, open_as_type=datamodels.RampModel)

        ngroups = result.data.shape[1]
        if ngroups < MIN_NGROUPS:
            # Like the jwst ramp_fit step when no fit is possible: no rate products.
            log.warning(f"ngroups = {ngroups} is below the minimum of {MIN_NGROUPS}; no fit done")
            return None, None

        subtracted = superbias_basename(result.meta.ref_file.superbias.name)
        try:
            best = superbias_basename(self.get_reference_file(result, "superbias"))
        except Exception as error:  # no CRDS match (synthetic data, no date): the check is informational only
            log.debug(f"No best superbias reference for this data: {error}")
            best = None
        if subtracted is not None and best is not None and subtracted != best:
            log.warning(
                f"The data had superbias {subtracted} subtracted; the current best reference is {best}. "
                "The prior is corrected for the file that was subtracted."
            )
        sigma_b, t1_scale, reset_level, channel_timing = self._reset_prior(result)

        readnoise_filename = self.get_reference_file(result, "readnoise")
        gain_filename = self.get_reference_file(result, "gain")
        log.info(f"Using READNOISE reference file: {readnoise_filename}")
        log.info(f"Using GAIN reference file: {gain_filename}")
        with (
            datamodels.ReadnoiseModel(readnoise_filename) as readnoise_model,
            datamodels.GainModel(gain_filename) as gain_model,
        ):
            if gain_model.meta.exposure.gain_factor is not None:
                result.meta.exposure.gain_factor = gain_model.meta.exposure.gain_factor
            readnoise_2d, gain_2d = get_reference_file_subarrays(result, readnoise_model, gain_model)

        ramp_data = create_ramp_fit_class(result, "LIKELY", dqflags.pixel, suppress_one_group=False)
        ramp_data.rejection_threshold = self.rejection_threshold
        if not self.use_zeroframe:
            ramp_data.zeroframe = None

        if self.firstgroup is not None or self.lastgroup is not None:
            log.warning("firstgroup and lastgroup are accepted for the ramp_fit slot but not implemented; all groups are fitted")
        log.info(
            f"Reset-anchored likelihood fit: the reset level enters as an extra resultant with a scatter of "
            f"{float(np.nanmedian(sigma_b)):.3g} DN, and the first read weighs {t1_scale:.3f} of its nominal time"
        )
        log.debug(
            f"sigma_b {'per pixel, median ' if np.ndim(sigma_b) == 2 else ''}{float(np.nanmedian(sigma_b)):.3g} DN, "
            f"t1_scale {t1_scale:.4f}, reset level median {float(np.nanmedian(reset_level)):.2f} DN"
            + (f", per channel t1_scale {channel_timing[0]} offsets {channel_timing[1]}" if channel_timing else "")
        )
        blocks = readout.channel_blocks(result.data.shape[-2:], result.meta.subarray.fastaxis, result.meta.subarray.slowaxis)
        if self.anchor_offset_to_sky and result.data.shape[1] >= 3:
            t1_scales = channel_timing[0] if channel_timing else [t1_scale] * len(blocks)
            # the prior without any scalar offset: the map (or zero); the step's own offset goes on top of the anchored value
            base = np.asarray(reset_level, dtype=np.float32) - np.float32(self._scalar_offset)
            # the file's absolute offsets per channel (channel_timing holds them relative to the file's scalar offset)
            file_scalar = self._scalar_offset - self.reset_offset
            file_offsets = [o + file_scalar for o in channel_timing[1]] if channel_timing else [file_scalar] * len(blocks)
            resultant = _first_resultant(result, self.use_zeroframe)
            if self.anchor_timing_to_stars:
                fitted = star_anchored_scales(
                    result, base, blocks, self.use_zeroframe, self.anchor_min_bright,
                    fallback=list(t1_scales), resultant=resultant,
                )
                log.info(
                    f"First-read timing anchored to this exposure's bright pixels: {_span(fitted, '.3f')} of the nominal "
                    f"time across the channels (calibration {_span(t1_scales, '.3f')})"
                )
                log.debug(
                    "First-read scales per channel: " + ", ".join(f"{k:.4f}" for k in fitted)
                    + " (calibration file, carried to this pattern: " + ", ".join(f"{k:.4f}" for k in t1_scales) + ")"
                )
                t1_scales = [float(k) for k in fitted]
            anchored = sky_anchored_offsets(
                result, base, t1_scales, blocks, self.use_zeroframe, self.anchor_max_signal, self.anchor_min_pixels,
                fallback=file_offsets, resultant=resultant,
            )
            log.info(
                f"Reset level anchored to this exposure's sky pixels: offsets of {_span(anchored, '+.2f')} DN across the "
                f"channels, against {_span(file_offsets, '+.2f')} in the calibration"
            )
            log.debug(
                "Reset offsets per channel: " + ", ".join(f"{b:+.2f}" for b in anchored)
                + " DN (calibration file: " + ", ".join(f"{b:+.2f}" for b in file_offsets) + ")"
            )
            channel_timing = (t1_scales, [float(b) + self.reset_offset for b in anchored])
            reset_level = base
        if channel_timing:
            t1_scales, offsets = channel_timing
            levels = [np.asarray(reset_level, dtype=np.float32) + np.float32(offset) for offset in offsets]
            image_info, integ_info, _ = likely_ramp_fit_reset_blocks(
                ramp_data, readnoise_2d, gain_2d, sigma_b, blocks, t1_scales, levels, use_zeroframe=self.use_zeroframe
            )
        else:
            image_info, integ_info, _ = likely_ramp_fit_reset(
                ramp_data,
                readnoise_2d,
                gain_2d,
                sigma_b,
                t1_scale=t1_scale,
                use_zeroframe=self.use_zeroframe,
                reset_level=reset_level,
            )

        if self.save_opt:
            path = self.make_output_path(basepath=result.meta.filename, suffix="resetfit", ext="asdf")
            write_resetfit(
                path, integ_info, sigma_b, t1_scale, result.meta.instrument.detector, reset_level,
                t1_scales=channel_timing[0] if channel_timing else None,
                reset_offsets=channel_timing[1] if channel_timing else None,
            )
            log.info(f"Saved reset diagnostics to {path}")

        out_model = create_image_model(result, image_info)
        out_model.meta.bunit_data = "DN/s"
        out_model.meta.bunit_err = "DN/s"
        out_model.meta.cal_step.ramp_fit = "COMPLETE"
        self.set_cal_step(out_model, "COMPLETE")

        int_times_stripe = getattr(result, "int_times_stripe", None)
        if int_times_stripe is not None and len(int_times_stripe) == 0:
            int_times_stripe = None
        int_model = create_integration_model(result, integ_info, result.int_times, int_times_stripe)
        int_model.meta.bunit_data = "DN/s"
        int_model.meta.bunit_err = "DN/s"
        int_model.meta.cal_step.ramp_fit = "COMPLETE"
        self.set_cal_step(int_model, "COMPLETE")
        # the jwst model copy keeps only the step records its schema knows; carry the others (stream_clean,
        # snowblind, jump_plus) from the ramp into the rate products
        for product in (out_model, int_model):
            carry_cal_step(result, product)

        # the pipeline's product names, from the exposure's file name (inside a hook stpipe has no other name
        # to go by); left to stpipe, a two-model result would get indexed names
        basename = result.meta.filename
        if self.as_hook:
            # a hook's return value becomes the pipeline's data, which must be one model; the pipeline writes
            # the rate, so the rateints product is written here or not at all
            self.save_model(int_model, suffix="rateints", output_file=basename, force=True)
            self.save_results = False
        elif self.save_results:
            self.save_model(out_model, suffix="rate", output_file=basename, force=True)
            self.save_model(int_model, suffix="rateints", output_file=basename, force=True)
            self.save_results = False

        if result is not step_input:
            del result
        if self.as_hook:
            return out_model
        return out_model, int_model

    def _reset_prior(self, model):
        """Return ``(sigma_b, t1_scale, reset_level, channel_timing)`` from the parameters and the calibration file.

        ``channel_timing`` is ``(t1_scales, reset_offsets)`` per channel when
        the file carries them and ``per_channel_timing`` is set and no
        ``t1_scale`` override was given; otherwise None.  The step's
        ``reset_offset`` is added on top of the file's offsets.

        Parameters
        ----------
        model : `~stdatamodels.jwst.datamodels.RampModel`
            The ramp being fitted; supplies the detector name, image shape,
            readout axes, subtracted superbias and first-group time.

        Returns
        -------
        sigma_b : float or ndarray of float64, shape (nrows, ncols)
            Reset-level scatter in DN.
        t1_scale : float
            First-read scale, carried to this exposure's first-group time
            when the file records ``t1_seconds``.
        reset_level : float32 or ndarray of float32, shape (nrows, ncols)
            Prior mean of the reset level in DN.
        channel_timing : tuple or None
            ``(t1_scales, reset_offsets)`` lists per channel, or None.
        """
        t1_scale = self.t1_scale
        sigma_b = self.sigma_b
        reset_level = np.float32(self.reset_offset)
        channel_timing = None
        if self.use_pedestal_mean and self.reset_calib is None:
            raise ValueError("use_pedestal_mean needs reset_calib")
        tree = None
        if self.reset_calib is not None:
            tree = read_calibration(self.reset_calib)
        elif sigma_b is None:
            tree = default_calibration(model.meta.instrument.detector)
            provenance = tree.get("provenance") or {}
            files = provenance.get("input_files", [])
            log.info(
                f"Using the reset calibration packaged for {tree['detector']}, measured "
                f"{str(provenance.get('date', ''))[:10]} from {len(files)} dark exposure{'s' if len(files) != 1 else ''}"
            )
            log.debug(
                f"Packaged calibration: sigma_b {tree['sigma_b_median']:.2f} DN, t1_scale {tree['t1_scale']:.4f}, "
                f"from {', '.join(files)}"
            )
        if tree is not None:
            file_offset = float(tree.get("reset_offset", 0.0) or 0.0)
            reset_level = np.float32(self.reset_offset + file_offset)
            scales = tree.get("t1_scale_channels")
            offsets = tree.get("reset_offset_channels")
            if self.per_channel_timing and (scales is not None or offsets is not None):
                nchannels = len(scales) if scales is not None else len(offsets)
                offsets = offsets if offsets is not None else [file_offset] * nchannels
                if self.t1_scale is None and scales is not None:
                    channel_scales = [float(v) for v in scales]
                else:
                    # a t1_scale override (or a file without per-channel scales) applies to every channel
                    one_scale = self.t1_scale if self.t1_scale is not None else tree.get("t1_scale", 1.0)
                    channel_scales = [float(one_scale)] * nchannels
                # the per-channel offsets replace the file's scalar offset; the step's offset stays on top
                channel_timing = (channel_scales, [float(v) - file_offset for v in offsets])
            if self.use_pedestal_mean:
                mean = calibrated_pedestal_map(
                    tree,
                    model.data.shape[-2:],
                    model.meta.subarray.fastaxis,
                    model.meta.subarray.slowaxis,
                    highpass=self.prior_highpass,
                    science_superbias=model.meta.ref_file.superbias.name,
                )
                reset_level = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32) + reset_level
            detector = str(tree.get("detector", "")).upper()
            if detector and detector != str(model.meta.instrument.detector).upper():
                raise ValueError(
                    f"Calibration file is for {detector}, data are {model.meta.instrument.detector}"
                )
            scale_from_file = t1_scale is None
            if t1_scale is None:
                t1_scale = float(tree.get("t1_scale", 1.0))
            t1_data = first_group_time(model)
            t1_cal = tree.get("t1_seconds")
            if t1_cal is not None and t1_data > 0 and scale_from_file:
                # a t1_scale given to the step is for this exposure's first-group time and is used as it is
                # the shortfall is a delay in seconds; carry it to this exposure's first-group time
                t1_scale = rescale_t1(t1_scale, float(t1_cal), t1_data)
                if channel_timing is not None:
                    channel_timing = ([rescale_t1(k, float(t1_cal), t1_data) for k in channel_timing[0]], channel_timing[1])
                log.info(
                    f"First-read timing carried from the calibration's first-group time of {float(t1_cal):.2f} s to this "
                    f"exposure's {t1_data:.2f} s"
                )
                log.debug(f"t1_scale after the carry: {t1_scale:.4f}")
            elif t1_cal is None and tree.get("t1_scale") is not None and abs(float(tree.get("t1_scale", 1.0)) - 1.0) > 1e-6:
                log.warning(
                    "The calibration's t1_scale has no first-group time recorded; it is used as it is, which is only right "
                    "for exposures with the readout pattern it was measured on"
                )
            if sigma_b is None:
                sigma_b = float(tree["sigma_b_median"])
        if sigma_b is None:
            raise ValueError("reset_rampfit needs either reset_calib or sigma_b")
        if np.ndim(sigma_b) == 0 and not (np.isfinite(sigma_b) and sigma_b > 0):
            raise ValueError(f"sigma_b must be positive and finite, got {sigma_b}")
        if t1_scale is None:
            t1_scale = 1.0
        self._scalar_offset = float(reset_level) if np.ndim(reset_level) == 0 else float(self.reset_offset + file_offset)
        return sigma_b, float(t1_scale), reset_level, channel_timing


def calibrated_pedestal_map(tree, shape, fastaxis, slowaxis, highpass=True, science_superbias=None):
    """The calibration's pedestal map for a science exposure, high-passed by default.

    The map is the mean reset level of the darks after their superbias was
    subtracted.  A superbias reference tracks the reset level of its epoch
    (two NRCALONG references from 2022 and 2025 differ by 11700 DN), so the
    residual relative to each epoch's own superbias is the quantity that
    transfers between epochs, and the map is used as it is for an exposure
    processed with a different superbias.  The step says so, because the
    per-pixel part of the residual has not been checked across epochs.

    Parameters
    ----------
    tree : dict
        Calibration tree from `~snowblind.reset_calib.read_calibration`;
        ``pedestal_mean`` and ``provenance["superbias"]`` are used.
    shape : tuple of int
        ``(nrows, ncols)`` of the data; the map must match it.
    fastaxis : int
        ``meta.subarray.fastaxis`` of the data, for the high-pass.
    slowaxis : int
        ``meta.subarray.slowaxis`` of the data.
    highpass : bool
        Apply `highpass_prior` to the map.
    science_superbias : str or None
        Name of the superbias subtracted from the data, compared with the
        calibration's to decide whether to warn.

    Returns
    -------
    ndarray of float32, shape (nrows, ncols)
        The pedestal map in DN, NaN where the calibration has no value.
    """
    mean = np.array(tree["pedestal_mean"], dtype=np.float32)
    if mean.shape != tuple(shape):
        raise ValueError(f"pedestal_mean shape {mean.shape} does not match data {tuple(shape)}")
    cal = superbias_basename(tree.get("provenance", {}).get("superbias"))
    sci = superbias_basename(science_superbias)
    if cal is None:
        log.warning("The calibration file does not record its superbias; the prior cannot be checked against the data's")
    elif sci is not None and cal != sci:
        log.warning(
            f"The data had superbias {sci} subtracted, the calibration darks {cal}: the pedestal map is used as "
            "the residual against the data's own epoch; its per-pixel part is untested across epochs"
        )
    if highpass:
        mean = highpass_prior(mean, fastaxis, slowaxis)
    return mean


def reset_prior_map(
    calib_path, shape, fastaxis, slowaxis, highpass=True, per_channel_timing=True, reset_offset=0.0, science_superbias=None
):
    """The per-pixel prior mean of the reset level, DN: pedestal map (high-passed by default) plus the offsets.

    This is the reference that ``stream_clean`` cleans the first read
    against and that ``reset_rampfit`` anchors to, built in one place so
    the two agree.  ``science_superbias`` is the name of the superbias
    subtracted from the data; when it differs from the calibration's the
    map is corrected by the difference of the two reference images.

    Parameters
    ----------
    calib_path : str
        Calibration ASDF file from ``snowblind-reset-calib``.
    shape : tuple of int
        ``(nrows, ncols)`` of the data.
    fastaxis : int
        ``meta.subarray.fastaxis`` of the data.
    slowaxis : int
        ``meta.subarray.slowaxis`` of the data.
    highpass : bool
        Apply `highpass_prior` to the pedestal map.
    per_channel_timing : bool
        Add the file's ``reset_offset_channels`` per output channel when it
        has them; otherwise the scalar ``reset_offset`` is added everywhere.
    reset_offset : float
        Extra offset in DN added on top of the file's.
    science_superbias : str or None
        Name of the superbias subtracted from the data, passed to
        `calibrated_pedestal_map`.

    Returns
    -------
    ndarray of float32, shape (nrows, ncols)
        Prior mean of the reset level in DN; zero where the map has no
        value, plus the offsets.
    """
    tree = read_calibration(calib_path)
    mean = calibrated_pedestal_map(tree, shape, fastaxis, slowaxis, highpass=highpass, science_superbias=science_superbias)
    level = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32) + np.float32(reset_offset)
    file_offset = float(tree.get("reset_offset", 0.0) or 0.0)
    scales = tree.get("t1_scale_channels")
    offsets = tree.get("reset_offset_channels")
    if per_channel_timing and scales is not None:
        offsets = offsets if offsets is not None else [file_offset] * len(scales)
        for (rows, cols), offset in zip(readout.channel_blocks(tuple(shape), fastaxis, slowaxis), offsets):
            level[rows, cols] += np.float32(offset)
    else:
        level += np.float32(file_offset)
    return level


def _first_resultant(model, use_zeroframe):
    """The first resultant the fit uses, its time since reset, a usable mask, the later-group slope (DN/s) and t1.

    Only the first integration is used.

    Parameters
    ----------
    model : `~stdatamodels.jwst.datamodels.RampModel`
        The ramp; ``data``, ``groupdq``, ``pixeldq``, ``zeroframe`` and the
        exposure timing keywords are read.
    use_zeroframe : bool
        Take the zeroframe as the first resultant when the model has one
        with any non-zero pixel; otherwise group 0 is used.

    Returns
    -------
    first : ndarray of float64, shape (nrows, ncols)
        The first resultant in DN (zeroframe or group 0).
    t_first : float
        Its time since reset in seconds: the frame time for the zeroframe,
        else the first group's mean time.
    clean : ndarray of bool, shape (nrows, ncols)
        Pixels with no group flag, no DO_NOT_USE pixel flag, a usable
        first resultant and finite values.
    rest : ndarray of float64, shape (nrows, ncols)
        Slope of groups 1 onwards in DN/s.
    t1 : float
        Mean time since reset of the first group in seconds.
    """
    from .stream_clean import slope_per_group

    exposure = model.meta.exposure
    t1 = first_group_time(model)
    data = np.asarray(model.data[0], dtype=np.float64)
    groupdq = np.asarray(model.groupdq[0])
    zeroframe = getattr(model, "zeroframe", None)
    if use_zeroframe and zeroframe is not None and np.any(zeroframe[0] != 0):
        first = np.asarray(zeroframe[0], dtype=np.float64)
        t_first = float(exposure.frame_time)
        usable = first != 0
    else:
        first = data[0]
        t_first = t1
        usable = np.ones(first.shape, dtype=bool)
    rest = slope_per_group(data[1:], groupdq[1:]) / float(exposure.group_time)
    clean = np.all(groupdq == 0, axis=0) & ((model.pixeldq & dqflags.pixel["DO_NOT_USE"]) == 0) & usable
    clean &= np.isfinite(rest) & np.isfinite(first)
    return first, t_first, clean, rest, t1


def star_anchored_scales(model, base_level, blocks, use_zeroframe, min_bright=500, fallback=None,
                         bright_range=(50.0, 2000.0), nbins=6, min_per_bin=20, resultant=None):
    """The first-read scale per channel from the exposure's own bright, unsaturated pixels.

    With ``y = first - prior - a * t_first`` (the first resultant's residual
    against nominal timing) and ``x = a`` the slope of the later groups,
    ``y = b - (1 - k) * t1 * x``: the slope of ``y`` against ``x`` over
    pixels whose nominal first-resultant signal lies in ``bright_range`` DN
    gives ``k``, from the medians of ``y`` in logarithmic bins of ``x``.  The
    scale measured on one readout pattern has not carried to others (0.90
    on BRIGHT1, 0.92 on SHALLOW4, 1.01 on MEDIUM8), and a scale wrong by
    0.04 biases the rate of every pixel the reset point weighs on by about
    a percent.  Channels with fewer than ``min_bright`` such pixels, or whose
    fit leaves fewer than 3 bins, keep their ``fallback`` scale.

    Parameters
    ----------
    model : `~stdatamodels.jwst.datamodels.RampModel`
        The ramp, read through `_first_resultant`.
    base_level : float or ndarray, shape (nrows, ncols)
        Prior mean of the reset level in DN, subtracted from the first
        resultant.
    blocks : sequence of (slice, slice)
        Science-frame ``(rows, cols)`` of each output channel.
    use_zeroframe : bool
        Passed to `_first_resultant`.
    resultant : tuple or None
        The output of `_first_resultant` for ``model``, when the caller has it
        already; None computes it here.
    min_bright : int
        Fewer bright pixels than this in a channel keeps its fallback scale.
    fallback : sequence of float or None
        Scale per channel to keep when the fit is not possible; None means
        1.0 everywhere.
    bright_range : tuple of float
        ``(low, high)`` in DN of the nominal first-resultant signal for a
        pixel to count as bright.
    nbins : int
        Number of logarithmic bins of the slope between the ends of
        ``bright_range``.
    min_per_bin : int
        Bins with fewer pixels are left out of the line fit.

    Returns
    -------
    list of float
        First-read scale per channel, each fitted value clipped to
        ``[0.7, 1.1]``.
    """
    first, t_first, clean, rest, t1 = resultant if resultant is not None else _first_resultant(model, use_zeroframe)
    base = np.broadcast_to(np.asarray(base_level, dtype=np.float64), first.shape)
    nominal = rest * t_first
    scales = []
    fallback = [1.0] * len(blocks) if fallback is None else list(fallback)
    lo, hi = bright_range
    for (rows, cols), kept in zip(blocks, fallback):
        x = rest[rows, cols]
        y = first[rows, cols] - base[rows, cols] - nominal[rows, cols]
        bright = clean[rows, cols] & (nominal[rows, cols] > lo) & (nominal[rows, cols] < hi)
        if bright.sum() < min_bright:
            log.warning(f"Only {int(bright.sum())} bright pixels in a channel; its first-read scale stays at {kept:.4f}")
            scales.append(float(kept))
            continue
        edges = np.geomspace(lo, hi, nbins + 1) / t_first
        xs, ys, ws = [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            sel = bright & (x >= a) & (x < b)
            if sel.sum() >= min_per_bin:
                xs.append(np.median(x[sel]))
                ys.append(np.median(y[sel]))
                ws.append(sel.sum())
        if len(xs) < 3:
            log.warning("Too few bright bins in a channel; its first-read scale stays at the file's")
            scales.append(float(kept))
            continue
        slope, _intercept = np.polyfit(np.array(xs), np.array(ys), 1, w=np.sqrt(np.array(ws, dtype=float)))
        k = 1.0 + slope / t1
        scales.append(float(np.clip(k, 0.7, 1.1)))
    return scales


def sky_anchored_offsets(
    model, base_level, t1_scales, blocks, use_zeroframe, max_signal=50.0, min_pixels=1000, fallback=None, resultant=None
):
    """The reset offset per channel from the exposure's own sky pixels, in DN.

    For each channel, the median over faint, unflagged pixels of the first
    resultant (the zeroframe when it is used, else group 0) minus the prior
    map minus the signal expected there from the slope of the later groups:
    ``a * (t_first - (1 - k) * t1)``, with ``k`` the channel's first-read
    scale.  The reset level drifts by 10 DN and more between epochs with the
    same superbias, so a calibration constant cannot serve; a few DN is a
    few percent of the first read on a faint sky.  Channels with fewer than
    ``min_pixels`` sky pixels keep their ``fallback`` offset (the
    calibration file's) with a warning.

    Parameters
    ----------
    model : `~stdatamodels.jwst.datamodels.RampModel`
        The ramp, read through `_first_resultant`.
    base_level : float or ndarray, shape (nrows, ncols)
        Prior mean of the reset level in DN without any scalar offset.
    t1_scales : sequence of float
        First-read scale ``k`` per channel.
    blocks : sequence of (slice, slice)
        Science-frame ``(rows, cols)`` of each output channel.
    use_zeroframe : bool
        Passed to `_first_resultant`.
    max_signal : float
        Pixels whose expected first-resultant signal is below this (and
        above ``-0.4 * max_signal``), in DN, count as sky.
    resultant : tuple or None
        The output of `_first_resultant` for ``model``, when the caller has it
        already; None computes it here.
    min_pixels : int
        Fewer sky pixels than this in a channel keeps its fallback offset.
    fallback : sequence of float or None
        Offset per channel in DN to keep when there are too few sky pixels;
        None means 0.0 everywhere.

    Returns
    -------
    list of float
        Reset offset per channel in DN.
    """
    first, t_first, clean, rest, t1 = resultant if resultant is not None else _first_resultant(model, use_zeroframe)
    base = np.broadcast_to(np.asarray(base_level, dtype=np.float64), first.shape)
    offsets = []
    fallback = [0.0] * len(blocks) if fallback is None else list(fallback)
    for (rows, cols), k, kept in zip(blocks, t1_scales, fallback):
        signal = rest[rows, cols] * (t_first - (1.0 - float(k)) * t1)
        faint = clean[rows, cols] & (signal < max_signal) & (signal > -0.4 * max_signal)
        if faint.sum() < min_pixels:
            log.warning(f"Only {int(faint.sum())} sky pixels in a channel; its reset offset stays at the file's {kept:+.2f} DN")
            offsets.append(float(kept))
            continue
        residual = first[rows, cols] - base[rows, cols] - signal
        offsets.append(float(np.median(residual[faint])))
    return offsets


def first_group_time(model):
    """Mean time since reset of the first group's reads, in seconds (the stcal convention).

    Parameters
    ----------
    model : `~stdatamodels.jwst.datamodels.RampModel`
        Supplies ``meta.exposure.frame_time`` and ``meta.exposure.nframes``.

    Returns
    -------
    float
        ``frame_time * (nframes + 1) / 2`` in seconds.
    """
    exposure = model.meta.exposure
    return float(exposure.frame_time) * (int(exposure.nframes) + 1) / 2.0


def rescale_t1(t1_scale, t1_calibration, t1_data):
    """Carry a first-read scale measured at one first-group time to another, as a fixed delay in seconds.

    Parameters
    ----------
    t1_scale : float
        First-read scale measured on the calibration exposure.
    t1_calibration : float
        First-group time of the calibration exposure in seconds.
    t1_data : float
        First-group time of the exposure being fitted in seconds.

    Returns
    -------
    float
        ``1 - (1 - t1_scale) * t1_calibration / t1_data``.
    """
    delay = (1.0 - float(t1_scale)) * float(t1_calibration)
    return 1.0 - delay / float(t1_data)


def highpass_prior(pedestal_mean, fastaxis, slowaxis):
    """Remove the slow structure of a pedestal map: per-channel row medians, even-odd and per-column offsets.

    The reset-anchored fit uses group 0 minus this map as a difference.
    When ``stream_clean`` cleans the first read (``clean_first_group``) it
    removes the read's row structure, which includes the real row pattern
    of the reset level; the map must lose the same structure or the fit
    inherits it with the opposite sign.  The row pattern of a calibrated
    map is about 3 DN rms on NRCA1 and repeats between darks, so it is
    not noise, but it is in both the first read and the map and cancels
    only if both are treated alike.  The per-channel mean of the map is
    kept, since the sky-anchored reset offsets are defined relative to it.

    Parameters
    ----------
    pedestal_mean : ndarray, shape (nrows, ncols)
        Pedestal map in the science frame, DN, NaN where unknown.
    fastaxis : int
        ``meta.subarray.fastaxis`` of the data.
    slowaxis : int
        ``meta.subarray.slowaxis`` of the data.

    Returns
    -------
    ndarray of float32, shape (nrows, ncols)
        The map minus its slow structure, DN; NaN where the input was.
    """
    from .stream_clean import estimate_noise

    finite = np.isfinite(pedestal_mean)
    image = np.where(finite, pedestal_mean, 0.0).astype(np.float32)
    slow = estimate_noise(
        image, finite.copy(), fastaxis, slowaxis, "row_median", even_odd_columns=True, column_offsets=True
    )
    blocks = readout.channel_blocks(image.shape, fastaxis, slowaxis)
    for rows, cols in blocks:
        inside = finite[rows, cols]
        if np.any(inside):
            slow[rows, cols] -= np.mean(slow[rows, cols][inside])
    return np.where(finite, image - slow, np.nan).astype(np.float32)


def write_resetfit(path, integ_info, sigma_b, t1_scale, detector, reset_level=0.0, t1_scales=None, reset_offsets=None):
    """Write the reset diagnostics of a fit to an ASDF file.

    Parameters
    ----------
    path : str
        Output ASDF file path.
    integ_info : dict
        Per-integration results of `likely_ramp_fit_reset`; the
        ``used_reset`` and ``d0_resid`` planes, shape
        (nints, nrows, ncols), are written along with their mean fraction
        and mean squared value.
    sigma_b : float or ndarray, shape (nrows, ncols)
        Reset-level scatter used by the fit, DN.
    t1_scale : float
        First-read scale used by the fit.
    detector : str
        Detector name.
    reset_level : float or ndarray, shape (nrows, ncols)
        Prior mean of the reset level used by the fit, DN, before any
        per-channel offset; its median plus the mean channel offset is
        stored as ``reset_level_median``.
    t1_scales : sequence of float or None
        First-read scale per output channel when the fit used them.
    reset_offsets : sequence of float or None
        Reset offset per output channel in DN when the fit used them,
        added to ``reset_level``.
    """
    used = np.asarray(integ_info["used_reset"])
    resid = np.asarray(integ_info["d0_resid"], dtype=np.float32)
    finite = np.isfinite(resid) & used
    tree = {
        "detector": detector,
        "sigma_b": np.asarray(sigma_b, dtype=np.float32),
        "t1_scale": float(t1_scale),
        "reset_level": np.asarray(reset_level, dtype=np.float32),
        "reset_level_median": float(np.nanmedian(reset_level)) + (float(np.mean(reset_offsets)) if reset_offsets else 0.0),
        "t1_scale_channels": None if t1_scales is None else [float(v) for v in t1_scales],
        "reset_offset_channels": None if reset_offsets is None else [float(v) for v in reset_offsets],
        "used_reset": used,
        "d0_resid": resid,
        "used_reset_fraction": float(used.mean()),
        "d0_chisq": float(np.mean(resid[finite] ** 2)) if finite.any() else float("nan"),
    }
    with asdf.AsdfFile(tree) as handle:
        add_history(
            handle,
            f"Reset-anchored likelihood fit diagnostics for {detector}: sigma_b median "
            f"{float(np.nanmedian(sigma_b)):.2f} DN, t1_scale {float(t1_scale):.4f}",
        )
        handle.write_to(path)

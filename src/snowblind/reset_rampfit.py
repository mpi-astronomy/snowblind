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
from jwst.stpipe import Step
from stcal.ramp_fitting.ramp_fit import create_ramp_fit_class
from stdatamodels.jwst.datamodels import dqflags

from . import readout
from .likely_fit_reset import MIN_NGROUPS, likely_ramp_fit_reset, likely_ramp_fit_reset_blocks
from .reset_calib import add_history, read_calibration, superbias_basename, superbias_difference  # noqa: F401

__all__ = ["ResetRampFitStep", "write_resetfit"]

log = logging.getLogger(__name__)


class ResetRampFitStep(Step):
    """Fit ramps by maximum likelihood with a Gaussian prior on the reset level.

    Input is a `~stdatamodels.jwst.datamodels.RampModel` after the jump step
    (and after `StreamCleanStep` with ``clean_first_group`` when both are
    used).  Output is an `~stdatamodels.jwst.datamodels.ImageModel` and a
    `~stdatamodels.jwst.datamodels.CubeModel` as from the jwst ramp_fit step.
    ``meta.cal_step.ramp_fit`` is set to COMPLETE as well so that downstream
    steps treat the result as a rate product.
    """

    class_alias = "reset_rampfit"

    @classmethod
    def get_stpipe_loggers(cls):
        """The jwst loggers plus snowblind's, so that this package's messages reach the step log."""
        return super().get_stpipe_loggers() + ("snowblind",)

    spec = """
        reset_calib = string(default=None)  # ASDF from snowblind-reset-calib; required unless sigma_b is set
        sigma_b = float(default=None)  # DN; scalar reset scatter, overrides the calibration file
        t1_scale = float(default=None)  # fraction of the nominal first interval the first read integrates; overrides the file
        per_channel_timing = boolean(default=True)  # use the file's per-channel t1_scale and reset_offset when present
        per_pixel_sigma_b = boolean(default=False)  # use the per-pixel sigma_b image from the calibration file
        use_pedestal_mean = boolean(default=False)  # prior mean of the reset level from the calibration file's pedestal_mean map
        prior_highpass = boolean(default=True)  # remove the pedestal map's row, even-odd and column structure, as stream_clean removes it from the first read
        reset_offset = float(default=0.0)  # DN added to the prior mean of the reset level, on top of the file's value
        use_zeroframe = boolean(default=True)  # use the zeroframe when present
        rejection_threshold = float(default=None)  # sigma threshold of the likelihood jump search; None for stcal default
        save_opt = boolean(default=False)  # save used_reset and d0_resid as a _resetfit ASDF product
        skip = boolean(default=False)
    """  # noqa: E501

    reference_file_types = ["readnoise", "gain", "superbias"]

    def process(self, step_input):
        """Fit the ramps of a group-level product.

        Parameters
        ----------
        step_input : str or `~stdatamodels.jwst.datamodels.RampModel`

        Returns
        -------
        out_model : `~stdatamodels.jwst.datamodels.ImageModel`
        int_model : `~stdatamodels.jwst.datamodels.CubeModel`
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

        log.info(
            f"Reset-anchored likelihood fit: sigma_b "
            f"{'per pixel, median ' if np.ndim(sigma_b) == 2 else ''}"
            f"{float(np.nanmedian(sigma_b)):.3g} DN, t1_scale {t1_scale:.4f}, "
            f"reset level median {float(np.nanmedian(reset_level)):.2f} DN"
            + (f", per channel t1_scale {channel_timing[0]} offsets {channel_timing[1]}" if channel_timing else "")
        )
        if channel_timing:
            t1_scales, offsets = channel_timing
            blocks = readout.channel_blocks(result.data.shape[-2:], result.meta.subarray.fastaxis, result.meta.subarray.slowaxis)
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
            write_resetfit(path, integ_info, sigma_b, t1_scale, result.meta.instrument.detector, reset_level)
            log.info(f"Saved reset diagnostics to {path}")

        out_model = create_image_model(result, image_info)
        out_model.meta.bunit_data = "DN/s"
        out_model.meta.bunit_err = "DN/s"
        out_model.meta.cal_step.ramp_fit = "COMPLETE"
        setattr(out_model.meta.cal_step, self.class_alias, "COMPLETE")

        int_times_stripe = getattr(result, "int_times_stripe", None)
        if int_times_stripe is not None and len(int_times_stripe) == 0:
            int_times_stripe = None
        int_model = create_integration_model(result, integ_info, result.int_times, int_times_stripe)
        int_model.meta.bunit_data = "DN/s"
        int_model.meta.bunit_err = "DN/s"
        int_model.meta.cal_step.ramp_fit = "COMPLETE"
        setattr(int_model.meta.cal_step, self.class_alias, "COMPLETE")

        if result is not step_input:
            del result
        return out_model, int_model

    def _reset_prior(self, model):
        """Return ``(sigma_b, t1_scale, reset_level, channel_timing)`` from the parameters and the calibration file.

        ``channel_timing`` is ``(t1_scales, reset_offsets)`` per channel when
        the file carries them and ``per_channel_timing`` is set and no
        ``t1_scale`` override was given; otherwise None.  The step's
        ``reset_offset`` is added on top of the file's offsets.
        """
        t1_scale = self.t1_scale
        sigma_b = self.sigma_b
        reset_level = np.float32(self.reset_offset)
        channel_timing = None
        if self.use_pedestal_mean and self.reset_calib is None:
            raise ValueError("use_pedestal_mean needs reset_calib")
        if self.reset_calib is not None:
            tree = read_calibration(self.reset_calib)
            file_offset = float(tree.get("reset_offset", 0.0) or 0.0)
            reset_level = np.float32(self.reset_offset + file_offset)
            scales = tree.get("t1_scale_channels")
            offsets = tree.get("reset_offset_channels")
            if self.per_channel_timing and self.t1_scale is None and scales is not None:
                offsets = offsets if offsets is not None else [file_offset] * len(scales)
                # the per-channel offsets replace the file's scalar offset; the step's offset stays on top
                channel_timing = ([float(v) for v in scales], [float(v) - file_offset for v in offsets])
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
            if t1_scale is None:
                t1_scale = float(tree.get("t1_scale", 1.0))
            t1_data = first_group_time(model)
            t1_cal = tree.get("t1_seconds")
            if t1_cal is not None and t1_data > 0:
                # the shortfall is a delay in seconds; carry it to this exposure's first-group time
                t1_scale = rescale_t1(t1_scale, float(t1_cal), t1_data)
                if channel_timing is not None:
                    channel_timing = ([rescale_t1(k, float(t1_cal), t1_data) for k in channel_timing[0]], channel_timing[1])
                log.info(
                    f"First-read timing carried from a first-group time of {float(t1_cal):.2f} s to {t1_data:.2f} s: "
                    f"t1_scale {t1_scale:.4f}"
                )
            elif t1_cal is None and tree.get("t1_scale") is not None and abs(float(tree.get("t1_scale", 1.0)) - 1.0) > 1e-6:
                log.warning(
                    "The calibration's t1_scale has no first-group time recorded; it is used as it is, which is only right "
                    "for exposures with the readout pattern it was measured on"
                )
            if sigma_b is None:
                if self.per_pixel_sigma_b:
                    sigma_b = np.array(tree["sigma_b"], dtype=np.float64)
                    if sigma_b.shape != model.data.shape[-2:]:
                        raise ValueError(
                            f"Per-pixel sigma_b shape {sigma_b.shape} does not match data {model.data.shape[-2:]}"
                        )
                    positive = np.isfinite(sigma_b) & (sigma_b > 0)
                    if not positive.any():
                        raise ValueError("Per-pixel sigma_b has no positive finite value")
                    sigma_b = np.where(positive, sigma_b, float(np.median(sigma_b[positive])))
                else:
                    sigma_b = float(tree["sigma_b_median"])
        if sigma_b is None:
            raise ValueError("reset_rampfit needs either reset_calib or sigma_b")
        if np.ndim(sigma_b) == 0 and not (np.isfinite(sigma_b) and sigma_b > 0):
            raise ValueError(f"sigma_b must be positive and finite, got {sigma_b}")
        if t1_scale is None:
            t1_scale = 1.0
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


def first_group_time(model):
    """Mean time since reset of the first group's reads, in seconds (the stcal convention)."""
    exposure = model.meta.exposure
    return float(exposure.frame_time) * (int(exposure.nframes) + 1) / 2.0


def rescale_t1(t1_scale, t1_calibration, t1_data):
    """Carry a first-read scale measured at one first-group time to another, as a fixed delay in seconds."""
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


def write_resetfit(path, integ_info, sigma_b, t1_scale, detector, reset_level=0.0):
    """Write the reset diagnostics of a fit to an ASDF file."""
    used = np.asarray(integ_info["used_reset"])
    resid = np.asarray(integ_info["d0_resid"], dtype=np.float32)
    finite = np.isfinite(resid) & used
    tree = {
        "detector": detector,
        "sigma_b": np.asarray(sigma_b, dtype=np.float32),
        "t1_scale": float(t1_scale),
        "reset_level": np.asarray(reset_level, dtype=np.float32),
        "reset_level_median": float(np.nanmedian(reset_level)),
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

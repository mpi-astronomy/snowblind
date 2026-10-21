"""Calibrate the reset-level uncertainty ``sigma_b`` for `ResetRampFitStep`.

``sigma_b`` is the single-integration scatter of the true reset level about
the superbias, per pixel, in DN.  It is measured from dark integrations
through the OLS ramp fit's optional ``pedestal`` output, which is the fitted
ramp extrapolated to the reset.  On a dark the pedestal is the reset
residual itself.

The pedestal computed by the stcal C fitter is
``group0 - slope * ((nframes + 1) / 2 + dropframes) / (nframes + groupgap)``
in group-time units, and it is written as exactly zero where the
integration is saturated or the fit failed.  Those zeros are treated as
missing here.

Inputs are ``_jump`` files, or ``_uncal`` files that are run in memory
through the Detector1 steps up to ``jump`` with SIRS reference pixel
correction.  Nothing but the calibration ASDF is written to disk.
"""
import argparse
import datetime
import logging
import os
import sys

import asdf
import numpy as np
from astropy.stats import sigma_clipped_stats
from jwst import __version__ as jwst_version
from jwst.ramp_fitting.ramp_fit_step import get_reference_file_subarrays
from jwst.stpipe import Step
from stcal import __version__ as stcal_version
from stcal.ramp_fitting import ramp_fit
from stdatamodels.jwst import datamodels
from stdatamodels.jwst.datamodels import dqflags

from . import readout

__all__ = [
    "run_detector1_to_jump",
    "pedestal_cube",
    "remove_channel_offsets",
    "sigma_b_from_pedestals",
    "write_calibration",
    "update_timing",
    "read_calibration",
    "main",
]

log = logging.getLogger(__name__)

DETECTOR1_TO_JUMP = (
    ("jwst.group_scale", "GroupScaleStep", {}),
    ("jwst.dq_init", "DQInitStep", {}),
    ("jwst.saturation", "SaturationStep", {}),
    ("jwst.superbias", "SuperBiasStep", {}),
    ("jwst.refpix", "RefPixStep", {"refpix_algorithm": "sirs"}),
    ("jwst.linearity", "LinearityStep", {}),
    ("jwst.dark_current", "DarkCurrentStep", {}),
    ("jwst.jump", "JumpStep", {}),
)


def run_detector1_to_jump(model, max_groups=None, max_ints=None, refpix_algorithm="sirs"):
    """Run the Detector1 steps up to and including jump, in memory, with SIRS refpix.

    Parameters
    ----------
    model : `~stdatamodels.jwst.datamodels.RampModel` or str
        An ``_uncal`` ramp or its path.
    max_groups : int or None
        If given, only the first ``max_groups`` groups are kept, to limit
        memory on long ramps.
    max_ints : int or None
        If given, only the first ``max_ints`` integrations are kept.
    refpix_algorithm : str
        ``"sirs"`` (default) or ``"median"``, passed to the refpix step.

    Returns
    -------
    `~stdatamodels.jwst.datamodels.RampModel`
        The ``_jump`` equivalent.
    """
    import importlib

    # Open as RampModel: uncal files open as Level1bModel with uint16 data,
    # which the Detector1 steps refuse.  RampModel converts to float32 and
    # creates groupdq and pixeldq.
    with datamodels.RampModel(model) as opened:
        nints, ngroups = opened.data.shape[:2]
        keep_ints = nints if max_ints is None else min(max_ints, nints)
        keep_groups = ngroups if max_groups is None else min(max_groups, ngroups)
        if (keep_ints, keep_groups) != (nints, ngroups):
            trimmed = datamodels.RampModel(opened.data[:keep_ints, :keep_groups].astype(np.float32))
            trimmed.update(opened)
            trimmed.groupdq = np.zeros(trimmed.data.shape, dtype=np.uint8)
            trimmed.pixeldq = np.zeros(trimmed.data.shape[-2:], dtype=np.uint32)
            trimmed.meta.exposure.ngroups = keep_groups
            trimmed.meta.exposure.nints = keep_ints
            result = trimmed
        else:
            result = opened.copy()

    for module_name, class_name, params in DETECTOR1_TO_JUMP:
        step_class = getattr(importlib.import_module(module_name), class_name)
        if class_name == "RefPixStep":
            params = {**params, "refpix_algorithm": refpix_algorithm}
        result = step_class.call(result, **params)
    return result


def pedestal_cube(jump_model, readnoise_2d=None, gain_2d=None):
    """Return the OLS pedestal for every integration of a ``_jump`` ramp.

    Runs the stcal OLS_C fitter with the optional output enabled and
    one-group suppression disabled, exactly as the plan prescribes.  The
    read noise and gain come from CRDS unless both arrays are given.

    Returns
    -------
    pedestal : ndarray of float32, shape (nints, nrows, ncols)
        ``opt_info["pedestal"]`` with zeros (the fitter's marker for an
        unusable integration) replaced by NaN.
    read_noise : ndarray of float32, shape (nrows, ncols)
        Single-read noise in DN from the readnoise reference file, which
        holds the CDS noise.
    """
    if readnoise_2d is None or gain_2d is None:
        step = Step()
        readnoise_filename = step.get_reference_file(jump_model, "readnoise")
        gain_filename = step.get_reference_file(jump_model, "gain")
        with (
            datamodels.ReadnoiseModel(readnoise_filename) as readnoise_model,
            datamodels.GainModel(gain_filename) as gain_model,
        ):
            readnoise_2d, gain_2d = get_reference_file_subarrays(jump_model, readnoise_model, gain_model)
    readnoise_2d = np.asarray(readnoise_2d, dtype=np.float32)
    gain_2d = np.asarray(gain_2d, dtype=np.float32)

    # stcal scales the read-noise array in place (by gain / sqrt(2 nframes)
    # for OLS), so hand it a copy and keep the reference-file values.
    _image_info, _integ_info, opt_info = ramp_fit.ramp_fit(
        jump_model,
        True,
        readnoise_2d.copy(),
        gain_2d.copy(),
        "OLS_C",
        "optimal",
        "1",
        dqflags.pixel,
        suppress_one_group=False,
    )
    pedestal = np.array(opt_info["pedestal"], dtype=np.float32)
    pedestal[pedestal == 0.0] = np.nan
    read_noise = (readnoise_2d / np.sqrt(2.0)).astype(np.float32)
    return pedestal, read_noise


def remove_channel_offsets(pedestals, fastaxis, slowaxis):
    """Subtract the per-integration median of each output channel.

    Frame-level offsets are the refpix step's job and must not be counted
    in ``sigma_b``.  The median is taken over all finite pixels of a channel
    in the detector frame.

    Parameters
    ----------
    pedestals : ndarray, shape (nints, nrows, ncols)
        Pedestal planes in the science frame, NaN where unusable.
    fastaxis, slowaxis : int
        Detector readout axis keywords.

    Returns
    -------
    ndarray of float32
        Pedestals with channel medians removed, in the science frame.
    """
    det = np.array(readout.to_detector_frame(pedestals, fastaxis, slowaxis), dtype=np.float32)
    for columns in readout.channel_slices(det.shape[-1]):
        block = det[..., columns]
        medians = np.nanmedian(block.reshape(block.shape[0], -1), axis=1)
        det[..., columns] = block - medians[:, np.newaxis, np.newaxis]
    return np.ascontiguousarray(readout.to_science_frame(det, fastaxis, slowaxis))


def sigma_b_from_pedestals(pedestals, sigma=3.0, read_noise=None, nframes=1):
    """Measure the per-pixel reset scatter from a stack of pedestal planes.

    The pedestal is the first group extrapolated to the reset, so its
    scatter contains the read noise of the first group as well as the reset
    scatter.  The likelihood fitter already accounts for the first group's
    read noise through its own term, so that part is removed here when
    ``read_noise`` is given: ``sigma_b**2 = scatter**2 - read_noise**2 / nframes``.

    Parameters
    ----------
    pedestals : ndarray, shape (n, nrows, ncols)
        Channel-offset-corrected pedestals, NaN where unusable.
    sigma : float
        Clipping threshold along the stack axis.
    read_noise : ndarray or None
        Single-read noise per pixel in DN.  None skips the correction.
    nframes : int
        Frames averaged into the first group.

    Returns
    -------
    sigma_b : ndarray of float32
        Reset scatter per pixel.  NaN where fewer than three finite values
        remain.
    mean : ndarray of float32
        Sigma-clipped mean per pixel, the systematic superbias error at this
        epoch.  Diagnostic only.
    count : ndarray of int32
        Number of finite values per pixel.
    scatter : ndarray of float32
        The raw sigma-clipped standard deviation of the pedestal.
    """
    count = np.isfinite(pedestals).sum(axis=0).astype(np.int32)
    mean, _median, std = sigma_clipped_stats(pedestals, sigma=sigma, maxiters=5, axis=0)
    scatter = np.where(count >= 3, std, np.nan).astype(np.float32)
    mean = np.where(count >= 3, mean, np.nan).astype(np.float32)
    if read_noise is None:
        sigma_b = scatter.copy()
    else:
        variance = scatter.astype(np.float64) ** 2 - np.asarray(read_noise, dtype=np.float64) ** 2 / nframes
        sigma_b = np.sqrt(np.clip(variance, 0.0, None)).astype(np.float32)
    return sigma_b, mean, count, scatter


def _snowblind_version():
    """Package version, looked up lazily to avoid a circular import at package load."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("snowblind")
    except PackageNotFoundError:
        return "dev"


def superbias_basename(name):
    """Base name of a reference file as stored in ``meta.ref_file`` (``crds://`` prefix and directories removed)."""
    if name is None:
        return None
    name = str(name)
    if name in ("", "N/A", "None"):
        return None
    return os.path.basename(name.replace("crds://", ""))


def write_calibration(
    path,
    sigma_b,
    mean,
    count,
    detector,
    input_files,
    t1_scale=1.0,
    refpix_algorithm="sirs",
    scatter=None,
    read_noise=None,
    reset_offset=0.0,
    t1_scale_channels=None,
    reset_offset_channels=None,
    timing_provenance=None,
    superbias=None,
):
    """Write the calibration ASDF file read by `ResetRampFitStep`.

    ``superbias`` is the name of the superbias reference file that was
    subtracted from the darks.  The pedestal map is a residual relative to
    it, and the step corrects the prior when a science exposure had a
    different superbias subtracted.

    ``t1_scale`` and ``reset_offset`` are the detector-wide first-read model
    ``R1 = reset_offset + t1_scale * a * t1``; the ``*_channels`` lists hold
    the same per output channel and take precedence in the step when present.
    """
    tree = {
        "detector": detector,
        "sigma_b": sigma_b,
        "sigma_b_median": float(np.nanmedian(sigma_b)),
        "pedestal_mean": mean,
        "pedestal_scatter": scatter if scatter is not None else sigma_b,
        "read_noise_single_median": float(np.nanmedian(read_noise)) if read_noise is not None else None,
        "n_integrations": count,
        "n_integrations_max": int(count.max()),
        "t1_scale": float(t1_scale),
        "reset_offset": float(reset_offset),
        "t1_scale_channels": None if t1_scale_channels is None else [float(v) for v in t1_scale_channels],
        "reset_offset_channels": None if reset_offset_channels is None else [float(v) for v in reset_offset_channels],
        "timing_provenance": timing_provenance,
        "provenance": {
            "input_files": [os.path.basename(f) for f in input_files],
            "refpix_algorithm": refpix_algorithm,
            "superbias": superbias,
            "jwst_version": jwst_version,
            "stcal_version": stcal_version,
            "snowblind_version": _snowblind_version(),
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        },
    }
    with asdf.AsdfFile(tree) as handle:
        handle.write_to(path)


def update_timing(path, t1_scale, reset_offset, t1_scale_channels=None, reset_offset_channels=None, provenance=None):
    """Rewrite a calibration file with new first-read timing parameters, keeping everything else."""
    tree = read_calibration(path)
    tree["t1_scale"] = float(t1_scale)
    tree["reset_offset"] = float(reset_offset)
    tree["t1_scale_channels"] = None if t1_scale_channels is None else [float(v) for v in t1_scale_channels]
    tree["reset_offset_channels"] = None if reset_offset_channels is None else [float(v) for v in reset_offset_channels]
    tree["timing_provenance"] = provenance
    with asdf.AsdfFile(tree) as handle:
        handle.write_to(path)


def read_calibration(path):
    """Return the tree of a calibration ASDF file as a plain dict with arrays loaded."""
    with asdf.open(path, lazy_load=False, memmap=False) as handle:
        tree = dict(handle.tree)
        tree.pop("asdf_library", None)
        tree.pop("history", None)
        for key in ("sigma_b", "pedestal_mean", "pedestal_scatter", "n_integrations"):
            if key in tree:
                tree[key] = np.array(tree[key])
    return tree


def calibrate(files, output, max_groups=None):
    """Measure ``sigma_b`` from a list of dark files for one detector and write the ASDF."""
    superbias_names = set()
    stacks = []
    detector = None
    fastaxis = slowaxis = None
    read_noise = None
    nframes = 1
    for path in files:
        log.info(f"Processing {path}")
        with datamodels.open(path) as opened:
            is_jump = str(opened.meta.cal_step.jump).upper() == "COMPLETE"
            this_detector = opened.meta.instrument.detector
            fastaxis = opened.meta.subarray.fastaxis
            slowaxis = opened.meta.subarray.slowaxis
            nframes = opened.meta.exposure.nframes
        if detector is None:
            detector = this_detector
        elif this_detector != detector:
            raise ValueError(f"{path} is {this_detector}; expected {detector}")

        model = datamodels.RampModel(path) if is_jump else run_detector1_to_jump(path, max_groups=max_groups)
        superbias_names.add(superbias_basename(model.meta.ref_file.superbias.name))
        pedestal, read_noise = pedestal_cube(model)
        model.close()
        stacks.append(remove_channel_offsets(pedestal, fastaxis, slowaxis))

    pedestals = np.concatenate(stacks, axis=0)
    superbias_names.discard(None)
    if len(superbias_names) > 1:
        raise ValueError(f"The darks had different superbias files subtracted: {sorted(superbias_names)}")
    superbias = superbias_names.pop() if superbias_names else None
    log.info(f"Superbias subtracted from the darks: {superbias}")
    sigma_b, mean, count, scatter = sigma_b_from_pedestals(pedestals, read_noise=read_noise, nframes=nframes)
    log.info(
        f"{detector}: median pedestal scatter = {np.nanmedian(scatter):.3f} DN, "
        f"median sigma_b = {np.nanmedian(sigma_b):.3f} DN after removing read noise "
        f"(median single read {np.nanmedian(read_noise):.3f} DN / sqrt({nframes})), "
        f"from {count.max()} integrations"
    )
    write_calibration(
        output, sigma_b, mean, count, detector, files, scatter=scatter, read_noise=read_noise, superbias=superbias
    )
    return sigma_b, mean, count


def main(argv=None):
    """Console script ``snowblind-reset-calib``."""
    parser = argparse.ArgumentParser(description="Measure the reset-level scatter sigma_b from dark ramps.")
    parser.add_argument("files", nargs="+", help="_jump or _uncal dark files for one detector")
    parser.add_argument("--output", "-o", required=True, help="output ASDF file")
    parser.add_argument("--max-groups", type=int, default=None, help="keep only the first N groups of uncal input")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s %(message)s")
    calibrate(args.files, args.output, max_groups=args.max_groups)


if __name__ == "__main__":
    main()

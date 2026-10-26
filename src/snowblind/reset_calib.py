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
import re
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


def run_detector1_to_jump(
    model, max_groups=None, max_ints=None, refpix_algorithm="sirs", snowblind=False, jump_params=None
):
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
    snowblind : bool
        Also run `~snowblind.SnowblindStep` after the jump step, growing the
        jump-detected snowballs to their actual extent (the jump step's own
        expansion is a circle of 1.75 to 2 times the core and leaves the
        halos below the jump threshold in the rate image).  The jump step's
        ``expand_large_events`` is then switched off, since Snowblind grows
        the raw jump regions and would otherwise grow the circles again.
    jump_params : dict or None
        Jump step parameters that override the CRDS parameter reference,
        for example ``{"min_jump_area": 8, "expand_factor": 2.0}``.

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
            zeroframe = getattr(opened, "zeroframe", None)
            if zeroframe is not None and np.ndim(zeroframe) == 3 and np.size(zeroframe) > 0:
                trimmed.zeroframe = np.array(zeroframe[:keep_ints], dtype=np.float32)
            int_times = getattr(opened, "int_times", None)
            if int_times is not None and len(int_times) > 0:
                trimmed.int_times = int_times[:keep_ints]
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
        if class_name == "JumpStep":
            params = {**params, **(jump_params or {})}
            if snowblind:
                params["expand_large_events"] = False
            if params:
                log.info(f"Jump step parameters overriding the reference: {params}")
        result = step_class.call(result, **params)
    if snowblind:
        from .snowblind import SnowblindStep

        result = SnowblindStep.call(result)
    return result


def pedestal_cube(jump_model, readnoise_2d=None, gain_2d=None):
    """Return the OLS pedestal for every integration of a ``_jump`` ramp.

    Runs the stcal OLS_C fitter with the optional output enabled and
    one-group suppression disabled, exactly as the plan prescribes.  The
    read noise and gain come from CRDS unless both arrays are given.

    Parameters
    ----------
    jump_model : `~stdatamodels.jwst.datamodels.RampModel`
        A ramp after the jump step, with ``groupdq`` and ``pixeldq`` set.
    readnoise_2d : ndarray or None, shape (nrows, ncols)
        CDS read noise per pixel in DN, as in the readnoise reference file.
        When this or ``gain_2d`` is None both are fetched from CRDS for the
        model and cut to its subarray.
    gain_2d : ndarray or None, shape (nrows, ncols)
        Gain per pixel in e/DN.

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
    fastaxis : int
        Detector readout axis keyword ``meta.subarray.fastaxis``.
    slowaxis : int
        Detector readout axis keyword ``meta.subarray.slowaxis``.

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
    # unusable integrations are NaN by design; pass them as a mask on a NaN-free array so astropy does not warn
    finite = np.isfinite(pedestals)
    mean, _median, std = sigma_clipped_stats(
        np.where(finite, pedestals, 0.0), mask=~finite, sigma=sigma, maxiters=5, axis=0
    )
    scatter = np.where(count >= 3, std, np.nan).astype(np.float32)
    mean = np.where(count >= 3, mean, np.nan).astype(np.float32)
    if read_noise is None:
        sigma_b = scatter.copy()
    else:
        variance = scatter.astype(np.float64) ** 2 - np.asarray(read_noise, dtype=np.float64) ** 2 / nframes
        sigma_b = np.sqrt(np.clip(variance, 0.0, None)).astype(np.float32)
    return sigma_b, mean, count, scatter


def _snowblind_version():
    """Package version, looked up lazily to avoid a circular import at package load.

    Returns
    -------
    str
        The installed version of ``snowblind``, or ``"dev"`` when the
        package metadata is not found.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("snowblind")
    except PackageNotFoundError:
        return "dev"


def software_entry():
    """The ``software`` record for ASDF history entries written by snowblind.

    Returns
    -------
    dict
        Keys ``name``, ``version``, ``homepage`` and ``author``, as
        ``asdf.AsdfFile.add_history_entry`` expects.
    """
    return {
        "name": "snowblind",
        "version": _snowblind_version(),
        "homepage": "https://github.com/mpi-astronomy/snowblind",
        "author": "James E. Davies",
    }


def add_history(handle, description):
    """Add a history entry to an ASDF file handle, naming snowblind as the software.

    Parameters
    ----------
    handle : `asdf.AsdfFile`
        Open ASDF file to add the entry to.
    description : str
        Text of the history entry.
    """
    handle.add_history_entry(description, software=software_entry())


def reference_superbias(names):
    """The superbias the calibration is expressed against: the latest delivery among the darks' files, or None.

    Parameters
    ----------
    names : iterable of str or None
        Superbias reference file base names, one per dark; None entries are
        ignored.

    Returns
    -------
    str or None
        The name with the largest serial number before ``.fits``, or None
        when no name is known.
    """
    known = sorted({name for name in names if name is not None})
    if not known:
        return None

    def number(name):
        """Serial number of a reference file name, used as the sort key.

        Parameters
        ----------
        name : str
            Reference file base name such as ``jwst_nircam_superbias_0123.fits``.

        Returns
        -------
        int
            The digits before ``.fits``, or -1 when the name does not end
            in digits followed by ``.fits``.
        """
        digits = re.findall(r"(\d+)\.fits$", name)
        return int(digits[0]) if digits else -1

    return max(known, key=number)


def superbias_basename(name):
    """Base name of a reference file as stored in ``meta.ref_file`` (``crds://`` prefix and directories removed).

    Parameters
    ----------
    name : str or None
        Reference file name or path, possibly with a ``crds://`` prefix.

    Returns
    -------
    str or None
        The base name, or None when ``name`` is None, empty, ``"N/A"`` or
        ``"None"``.
    """
    if name is None:
        return None
    name = str(name)
    if name in ("", "N/A", "None"):
        return None
    return os.path.basename(name.replace("crds://", ""))


def superbias_difference(calibration_superbias, science_superbias, shape, locate=None):
    """``calibration minus science`` superbias images, or None when they are the same file or unknown.

    The pedestal map is the mean reset level of the darks after their
    superbias was subtracted, so for a science exposure that had another
    superbias subtracted the prior is the map plus this difference.  The
    names are resolved through the CRDS cache (``crds.locate_file``) unless
    ``locate`` is given.

    Parameters
    ----------
    calibration_superbias : str or None
        Name of the superbias subtracted from the calibration darks, as
        recorded in the calibration file.
    science_superbias : str or None
        Name of the superbias subtracted from the science data
        (``meta.ref_file.superbias.name``).
    shape : tuple of int
        ``(nrows, ncols)`` of the data; both superbias images must match it.
    locate : callable or None
        Maps a reference file base name to a path on disk.  Default is
        ``crds.locate_file(name, "jwst")``.

    Returns
    -------
    ndarray of float32, shape (nrows, ncols) or None
        Calibration superbias minus science superbias in DN, or None when
        either name is unknown or both are the same file.
    """
    cal = superbias_basename(calibration_superbias)
    sci = superbias_basename(science_superbias)
    if cal is None or sci is None:
        if cal is None and sci is not None:
            log.warning("The calibration file does not record its superbias; the prior cannot be checked against the data's")
        return None
    if cal == sci:
        return None
    if locate is None:
        from crds import locate_file

        def locate(name):
            """Path of a reference file in the CRDS cache.

            Parameters
            ----------
            name : str
                Reference file base name.

            Returns
            -------
            str
                The path ``crds.locate_file`` gives for the ``jwst``
                observatory.
            """
            return locate_file(name, "jwst")

    from astropy.io import fits

    images = []
    for name in (cal, sci):
        path = locate(name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Superbias {name} is not in the CRDS cache ({path}); fetch it to correct the prior")
        with fits.open(path) as handle:
            images.append(np.asarray(handle["SCI"].data, dtype=np.float32))
    for image in images:
        if image.shape != tuple(shape):
            raise ValueError(f"Superbias shape {image.shape} does not match data {tuple(shape)}")
    log.info(f"Prior corrected from superbias {cal} (calibration) to {sci} (data)")
    return images[0] - images[1]


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
    superbias_inputs=None,
):
    """Write the calibration ASDF file read by `ResetRampFitStep`.

    ``superbias`` is the name of the superbias reference file that was
    subtracted from the darks.  The pedestal map is a residual relative to
    it, and the step corrects the prior when a science exposure had a
    different superbias subtracted.

    ``t1_scale`` and ``reset_offset`` are the detector-wide first-read model
    ``R1 = reset_offset + t1_scale * a * t1``; the ``*_channels`` lists hold
    the same per output channel and take precedence in the step when present.

    Parameters
    ----------
    path : str
        Output ASDF file path.
    sigma_b : ndarray of float32, shape (nrows, ncols)
        Reset scatter per pixel in DN; its median is also stored.
    mean : ndarray of float32, shape (nrows, ncols)
        Sigma-clipped mean pedestal per pixel in DN, stored as
        ``pedestal_mean``.
    count : ndarray of int32, shape (nrows, ncols)
        Number of integrations per pixel, stored as ``n_integrations``.
    detector : str
        Detector name the calibration applies to.
    input_files : sequence of str
        Paths of the dark files; their base names go into the provenance.
    t1_scale : float
        Detector-wide first-read scale.
    refpix_algorithm : str
        Reference pixel algorithm the darks were processed with, recorded
        in the provenance.
    scatter : ndarray of float32 or None, shape (nrows, ncols)
        Raw sigma-clipped pedestal scatter in DN, stored as
        ``pedestal_scatter``; ``sigma_b`` is stored instead when None.
    read_noise : ndarray or None, shape (nrows, ncols)
        Single-read noise per pixel in DN; only its median is stored.
    reset_offset : float
        Detector-wide reset offset in DN.
    t1_scale_channels : sequence of float or None
        First-read scale per output channel.
    reset_offset_channels : sequence of float or None
        Reset offset per output channel in DN.
    timing_provenance : dict or None
        Where the timing parameters were measured, stored as it is.
    superbias : str or None
        Name of the superbias reference the pedestal map is relative to.
    superbias_inputs : sequence of str or None
        All superbias names found among the darks.
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
            "superbias_inputs": None if superbias_inputs is None else list(superbias_inputs),
            "jwst_version": jwst_version,
            "stcal_version": stcal_version,
            "snowblind_version": _snowblind_version(),
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        },
    }
    with asdf.AsdfFile(tree) as handle:
        add_history(
            handle,
            f"Reset-level calibration for {detector} from {int(count.max())} dark integrations of "
            f"{', '.join(os.path.basename(f) for f in input_files)} (refpix {refpix_algorithm}, superbias {superbias}): "
            f"median sigma_b {float(np.nanmedian(sigma_b)):.2f} DN",
        )
        handle.write_to(path)


def update_timing(
    path, t1_scale, reset_offset, t1_scale_channels=None, reset_offset_channels=None, provenance=None, t1_seconds=None
):
    """Rewrite a calibration file's first-read timing parameters in place, keeping everything else including its history.

    ``t1_seconds`` is the mean time since reset of the first group of the
    exposure the timing was measured on.  With it the step can carry the
    first-read shortfall to exposures with another readout pattern as a
    delay in seconds, ``(1 - t1_scale) * t1_seconds``, which is what the
    data show it to be (0.6 s on NIRCam: BRIGHT1 gives t1_scale 0.94 at a
    first-group time of 10.7 s, MEDIUM8 gives 1.0 at 48 s).

    Parameters
    ----------
    path : str
        Calibration ASDF file to update in place.
    t1_scale : float
        Detector-wide first-read scale.
    reset_offset : float
        Detector-wide reset offset in DN.
    t1_scale_channels : sequence of float or None
        First-read scale per output channel; None clears the entry.
    reset_offset_channels : sequence of float or None
        Reset offset per output channel in DN; None clears the entry.
    provenance : dict or None
        Where the timing was measured, stored as ``timing_provenance``.  Its
        ``file`` entry, when present, is named in the history entry.
    t1_seconds : float or None
        Mean time since reset of the first group of the exposure the timing
        was measured on, in seconds.
    """
    with asdf.open(path, mode="rw") as handle:
        tree = handle.tree
        tree["t1_scale"] = float(t1_scale)
        tree["t1_seconds"] = None if t1_seconds is None else float(t1_seconds)
        tree["reset_offset"] = float(reset_offset)
        tree["t1_scale_channels"] = None if t1_scale_channels is None else [float(v) for v in t1_scale_channels]
        tree["reset_offset_channels"] = None if reset_offset_channels is None else [float(v) for v in reset_offset_channels]
        tree["timing_provenance"] = provenance
        source = provenance.get("file", "unknown data") if isinstance(provenance, dict) else "unknown data"
        channels = ""
        if t1_scale_channels is not None:
            channels = f"; per channel t1_scale {[round(float(v), 4) for v in t1_scale_channels]}"
        add_history(
            handle,
            f"First-read timing from {source}: t1_scale {float(t1_scale):.4f}"
            + ("" if t1_seconds is None else f" at a first-group time of {float(t1_seconds):.2f} s")
            + f", reset_offset {float(reset_offset):.2f} DN{channels}",
        )
        handle.update()


DEFAULTS_PATTERN = "reset_calib_{detector}.asdf"


def default_calibration(detector):
    """The packaged scalar reset calibration of a detector, shaped like a calibration tree without the maps.

    ``snowblind/data/reset_calib_<detector>.asdf``, one file per detector
    (the ten NIRCam detectors and NIRISS), holds the median reset scatter
    ``sigma_b``, the first-read scale and reset offset (detector-wide and
    per output channel), the first-group time the timing was measured at
    and the provenance of the dark and field exposures
    (``scripts/package_reset_defaults.py`` writes them from a directory of
    calibration files, and one detector can be updated alone).
    `ResetRampFitStep` falls back on the detector's file when it is given
    neither ``reset_calib`` nor ``sigma_b``.

    Parameters
    ----------
    detector : str
        Detector name as in ``meta.instrument.detector``, e.g. ``NRCA1``;
        case does not matter.

    Returns
    -------
    dict
        The entry with the keys `read_calibration` returns for scalars
        (``detector``, ``sigma_b``, ``sigma_b_median``, ``t1_scale``,
        ``reset_offset``, ``t1_scale_channels``, ``reset_offset_channels``,
        ``t1_seconds``, ``provenance``, ``timing_provenance``) plus the
        file's ``source`` and ``generated``; no pedestal maps.

    Raises
    ------
    ValueError
        If the package has no file for the detector.
    """
    from importlib.resources import files

    key = str(detector).upper()
    data_dir = files("snowblind").joinpath("data")
    path = data_dir.joinpath(DEFAULTS_PATTERN.format(detector=key.lower()))
    if not path.is_file():
        available = sorted(
            entry.name[len("reset_calib_"):-len(".asdf")].upper()
            for entry in data_dir.iterdir()
            if entry.name.startswith("reset_calib_") and entry.name.endswith(".asdf")
        )
        raise ValueError(
            f"No packaged reset calibration for detector {detector!r} (available: {', '.join(available)}); "
            "measure one with snowblind-reset-calib and pass it as reset_calib"
        )
    with asdf.open(str(path), lazy_load=False, memmap=False) as handle:
        entry = {k: v for k, v in handle.tree.items() if k not in ("asdf_library", "history")}
    return {
        "detector": key,
        "sigma_b": entry["sigma_b"],
        "sigma_b_median": entry["sigma_b"],
        "t1_scale": entry["t1_scale"],
        "reset_offset": entry["reset_offset"],
        "t1_scale_channels": entry["t1_scale_channels"],
        "reset_offset_channels": entry["reset_offset_channels"],
        "t1_seconds": entry["t1_seconds"],
        "provenance": entry["provenance"],
        "timing_provenance": entry["timing_provenance"],
        "source": entry["source"],
        "generated": entry["generated"],
    }


def read_calibration(path):
    """Return the tree of a calibration ASDF file as a plain dict with arrays loaded.

    Parameters
    ----------
    path : str
        Calibration ASDF file written by `write_calibration`.

    Returns
    -------
    dict
        The tree without ``asdf_library`` and ``history``; ``sigma_b``,
        ``pedestal_mean``, ``pedestal_scatter`` and ``n_integrations`` are
        in-memory ``ndarray`` copies.
    """
    with asdf.open(path, lazy_load=False, memmap=False) as handle:
        tree = dict(handle.tree)
        tree.pop("asdf_library", None)
        tree.pop("history", None)
        for key in ("sigma_b", "pedestal_mean", "pedestal_scatter", "n_integrations"):
            if key in tree:
                tree[key] = np.array(tree[key])
    return tree


def calibrate(files, output, max_groups=None):
    """Measure ``sigma_b`` from a list of dark files for one detector and write the ASDF.

    Parameters
    ----------
    files : sequence of str
        ``_jump`` or ``_uncal`` dark files, all of the same detector.  An
        ``_uncal`` file is run through `run_detector1_to_jump` first.
    output : str
        Path of the calibration ASDF file to write.
    max_groups : int or None
        Passed to `run_detector1_to_jump` for ``_uncal`` inputs: keep only
        the first ``max_groups`` groups.

    Returns
    -------
    sigma_b : ndarray of float32, shape (nrows, ncols)
        Reset scatter per pixel in DN.
    mean : ndarray of float32, shape (nrows, ncols)
        Sigma-clipped mean pedestal per pixel in DN.
    count : ndarray of int32, shape (nrows, ncols)
        Number of finite pedestal values per pixel.
    """
    inputs = []  # (pedestal cube, superbias name, path)
    detector = None
    fastaxis = slowaxis = None
    read_noise = None
    nframes = None
    for path in files:
        log.info(f"Processing {path}")
        with datamodels.open(path) as opened:
            is_jump = str(opened.meta.cal_step.jump).upper() == "COMPLETE"
            this_detector = opened.meta.instrument.detector
            this_nframes = opened.meta.exposure.nframes
            fastaxis = opened.meta.subarray.fastaxis
            slowaxis = opened.meta.subarray.slowaxis
        if detector is None:
            detector, nframes = this_detector, this_nframes
        elif this_detector != detector:
            raise ValueError(f"{path} is {this_detector}; expected {detector}")
        elif this_nframes != nframes:
            # the read noise removed from the pedestal scatter scales with the frames per group
            raise ValueError(f"{path} has {this_nframes} frames per group; the first dark has {nframes}")

        model = datamodels.RampModel(path) if is_jump else run_detector1_to_jump(path, max_groups=max_groups)
        name = superbias_basename(model.meta.ref_file.superbias.name)
        pedestal, read_noise = pedestal_cube(model)
        model.close()
        inputs.append((pedestal, name, path))

    names = sorted({name for _pedestal, name, _path in inputs if name is not None})
    superbias = reference_superbias(names)
    log.info(f"Superbias of the calibration: {superbias}")
    if len(names) > 1:
        # A superbias tracks the reset level of its epoch, so each dark's pedestal is kept as the residual
        # against its own superbias; adding the difference of the references would add the drift back.
        log.warning(f"The darks had different superbias files subtracted ({', '.join(names)}); residuals stacked as they are")
    stacks = [remove_channel_offsets(pedestal, fastaxis, slowaxis) for pedestal, _name, _path in inputs]
    pedestals = np.concatenate(stacks, axis=0)
    sigma_b, mean, count, scatter = sigma_b_from_pedestals(pedestals, read_noise=read_noise, nframes=nframes)
    log.info(
        f"{detector}: median pedestal scatter = {np.nanmedian(scatter):.3f} DN, "
        f"median sigma_b = {np.nanmedian(sigma_b):.3f} DN after removing read noise "
        f"(median single read {np.nanmedian(read_noise):.3f} DN / sqrt({nframes})), "
        f"from {count.max()} integrations"
    )
    write_calibration(
        output, sigma_b, mean, count, detector, files, scatter=scatter, read_noise=read_noise, superbias=superbias,
        superbias_inputs=names,
    )
    return sigma_b, mean, count


def main(argv=None):
    """Console script ``snowblind-reset-calib``.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments without the program name; None reads
        ``sys.argv``.  Accepts the dark files, ``--output`` and
        ``--max-groups``.
    """
    parser = argparse.ArgumentParser(description="Measure the reset-level scatter sigma_b from dark ramps.")
    parser.add_argument("files", nargs="+", help="_jump or _uncal dark files for one detector")
    parser.add_argument("--output", "-o", required=True, help="output ASDF file")
    parser.add_argument("--max-groups", type=int, default=None, help="keep only the first N groups of uncal input")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s %(message)s")
    calibrate(args.files, args.output, max_groups=args.max_groups)


if __name__ == "__main__":
    main()

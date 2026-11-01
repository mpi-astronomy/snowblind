"""Experiment: the stream clean before or after jump detection, and how low a second-pass threshold can go.

The ramp is run through Detector1 up to jump (SIRS refpix, jump skipped)
and then given three sets of first-pass flags:

- ``user``: James's jump settings, the jump step at 5 sigma with neighbours
  from 20 sigma, no after-jump flagging and no ``expand_large_events``,
  followed by ``SnowblindStep(min_radius=3, after_jumps=5)`` and
  ``JumpPlusStep``;
- ``high``: the same at ``--first-threshold`` sigma (7 by default; the
  three- and four-group thresholds are raised to it as well);
- ``none``: no jump flags, saturation only.

Question 1, how sensitive the clean is to jumps it is not told about: the
ramp is cleaned with each set.  The raw data are the same, so the
difference of two cleaned ramps is the difference of their noise
estimates.  It is summarised over the background pixels per group
difference and in the OLS rate (all with the ``user`` DQ), and in the
readout lines of each large event (a jump region of ``--event-area``
pixels or more in the ``user`` flags) against lines elsewhere in the
same channel.

Questions 2 and 3, how low a second pass can go and which detector does
better: on the raw ramp and on the ramp cleaned after the ``high`` first
pass, the ``high`` flags are turned into DO_NOT_USE and a second pass runs
at each of ``--thresholds``: the jump step's two-point test alone (no
neighbours, snowballs or after-jump flags) and the likelihood search of
the LIKELY fit (stcal ``mask_jumps``, the fitramp algorithm, without the
reset difference of ``ResetRampFitStep``).  Cosmic rays only raise the
signal, so flags on differences below the pixel's median difference count
the noise false flags; flags above minus flags below count the real jumps.
Also recorded: flags next to a first-pass flag (wings), flags crowding
into one readout line of a channel (1/f), and flags on source pixels.

Each cleaned ramp is also fitted on its own (OLS, ``user`` DQ) and its
rate summarised by the scatter of pixels, of readout-line means per
channel and of column means about the background median.  With
``--reset-fit`` the ``user`` and ``high`` chains are fitted with
``ResetRampFitStep`` at its default jump-search threshold and at 4 sigma:
its log line of new jumps, the flagged fraction and the upward and
downward rate outliers are recorded.  The outliers are also counted on
the same pixels and with the same noise scale for every run (the
``high_default`` run's unflagged background pixels and err), with the
pixels each run flagged kept in, and pixel by pixel between pairs of runs
(``reset_fit_common`` in the summary).

Output: ``summary.json``, ``cutouts.npz`` (with the three rates, and the
reset fits' rates, errs and flagged pixels) and, with
the sweep, ``flags.npz`` (every second-pass flag, for
``exp_jump_order_pixels.py``) in ``results/<date>_jump_order/<detector>/``.
"""
import argparse
import datetime
import glob
import json
import logging
import os
import time
import warnings

import numpy as np
import scipy.special
import scipy.stats
from jwst import datamodels
from jwst.jump import JumpStep
from jwst.ramp_fitting.ramp_fit_step import get_reference_file_subarrays
from scipy import ndimage
from stcal.ramp_fitting.likely_algo_classes import Covar
from stcal.ramp_fitting.likely_fit import get_readtimes, mask_jumps
from stcal.ramp_fitting.ramp_fit import create_ramp_fit_class
from stdatamodels.jwst.datamodels import dqflags

from snowblind import JumpPlusStep, ResetRampFitStep, SnowblindStep, StreamCleanStep, readout
from snowblind.reset_calib import DETECTOR1_TO_JUMP
from snowblind.stream_clean import slope_per_group

DNU = dqflags.group["DO_NOT_USE"]
SAT = dqflags.group["SATURATED"]
JUMP = dqflags.group["JUMP_DET"]
BORDER = 4
CROWDED_LINE = 5  # flags in one readout line of one channel and difference that count as crowded


def run_to_jump_input(path, max_groups, max_ints):
    """The Detector1 steps of ``run_detector1_to_jump`` (SIRS refpix) up to dark_current, without the jump step.

    ``JumpStep.call(..., skip=True)`` does not skip: stpipe sets skip to False
    for a step run standalone, so the jump step is left out of the loop instead.
    """
    import importlib

    with datamodels.RampModel(path) as opened:
        keep_ints = opened.data.shape[0] if max_ints is None else min(max_ints, opened.data.shape[0])
        keep_groups = opened.data.shape[1] if max_groups is None else min(max_groups, opened.data.shape[1])
        result = datamodels.RampModel(opened.data[:keep_ints, :keep_groups].astype(np.float32))
        result.update(opened)
        zeroframe = getattr(opened, "zeroframe", None)
        if zeroframe is not None and np.ndim(zeroframe) == 3 and np.size(zeroframe) > 0:
            result.zeroframe = np.array(zeroframe[:keep_ints], dtype=np.float32)
        int_times = getattr(opened, "int_times", None)
        if int_times is not None and len(int_times) > 0:
            result.int_times = int_times[:keep_ints]
        result.groupdq = np.zeros(result.data.shape, dtype=np.uint8)
        result.pixeldq = np.zeros(result.data.shape[-2:], dtype=np.uint32)
        result.meta.exposure.ngroups = keep_groups
        result.meta.exposure.nints = keep_ints
    for module_name, class_name, params in DETECTOR1_TO_JUMP:
        if class_name == "JumpStep":
            continue
        step_class = getattr(importlib.import_module(module_name), class_name)
        result = step_class.call(result, **params)
    return result


def first_pass(raw, threshold):
    """James's jump settings at ``threshold`` sigma, then SnowblindStep and JumpPlusStep; returns the groupdq."""
    jumped = JumpStep.call(
        raw.copy(),
        rejection_threshold=threshold,
        four_group_rejection_threshold=max(5.0, threshold),
        three_group_rejection_threshold=max(6.0, threshold),
        flag_4_neighbors=True,
        min_jump_to_flag_neighbors=20.0,
        after_jump_flag_time1=0.0,
        expand_large_events=False,
    )
    snowed = SnowblindStep.call(jumped, min_radius=3, after_jumps=5)
    jumped.close()
    plussed = JumpPlusStep.call(snowed)
    snowed.close()
    groupdq = np.array(plussed.groupdq)
    plussed.close()
    return groupdq


def with_groupdq(model, groupdq):
    """A copy of ``model`` with its groupdq replaced."""
    copy = model.copy()
    copy.groupdq = groupdq.copy()
    return copy


def clean(model, groupdq, dark, out_dir, save_mask):
    """StreamCleanStep defaults (background None on a dark) on ``model`` with ``groupdq``.

    Returns the cleaned data, the cleaned zeroframe (None without one; the
    step cleans it with the first group, and the reset fit uses it as a
    resultant) and, with ``save_mask``, the background mask.
    """
    flagged = with_groupdq(model, groupdq)
    kwargs = {"background_method": None} if dark else {}
    cleaned = StreamCleanStep.call(flagged, save_mask=save_mask, output_dir=out_dir, **kwargs)
    flagged.close()
    data = np.array(cleaned.data, dtype=np.float32)
    zeroframe = getattr(cleaned, "zeroframe", None)
    zeroframe = None if zeroframe is None or np.size(zeroframe) == 0 else np.array(zeroframe, dtype=np.float32)
    cleaned.close()
    mask = None
    if save_mask:
        path = sorted(glob.glob(os.path.join(out_dir, "*_mask.fits")), key=os.path.getmtime)[-1]
        with datamodels.open(path) as mask_model:
            mask = np.asarray(mask_model.data).astype(bool)
        os.replace(path, os.path.join(out_dir, "stream_clean_mask.fits"))
    return data, zeroframe, mask


def difference(cube, j):
    """Group difference j of one integration's cube (group 0 itself for j == 0)."""
    if j == 0:
        return cube[0].astype(np.float64)
    return cube[j].astype(np.float64) - cube[j - 1].astype(np.float64)


def interior(shape):
    """True away from the reference-pixel border."""
    inside = np.zeros(shape, dtype=bool)
    inside[BORDER:-BORDER, BORDER:-BORDER] = True
    return inside


def lines_view(array, fastaxis):
    """``array`` (..., nrows, ncols) with readout lines along the last axis."""
    return array if abs(fastaxis) == 1 else np.swapaxes(array, -1, -2)


def large_events(groupdq, min_area):
    """Connected JUMP_DET regions of at least ``min_area`` pixels: list of (integration, group, slices, mask)."""
    events = []
    nints, ngroups = groupdq.shape[:2]
    for i in range(nints):
        for j in range(1, ngroups):
            labels, n = ndimage.label((groupdq[i, j] & JUMP) > 0)
            if n == 0:
                continue
            areas = ndimage.sum_labels(np.ones_like(labels), labels, index=np.arange(1, n + 1))
            for label, box in enumerate(ndimage.find_objects(labels), start=1):
                if areas[label - 1] >= min_area:
                    events.append((i, j, box, labels[box] == label))
    return events


def band_means(image, usable, events_here, blocks, fastaxis, rng, ncontrol=20):
    """Mean of ``image`` in the readout lines of each event and in random control bands of the same channel.

    ``events_here`` holds (slices, mask) of the events in this difference.
    Returns two lists: one band mean per event, and the control band means.
    """
    image_l = lines_view(image, fastaxis)
    usable_l = lines_view(usable, fastaxis)
    event_means, control_means = [], []
    for box, _mask in events_here:
        box_l = box if abs(fastaxis) == 1 else box[::-1]
        centre = (box_l[1].start + box_l[1].stop) // 2
        block = next(b for b in (blk if abs(fastaxis) == 1 else blk[::-1] for blk in blocks) if b[1].start <= centre < b[1].stop)
        lines = slice(box_l[0].start, box_l[0].stop)
        sel = usable_l[lines, block[1]]
        if sel.sum() > 50:
            event_means.append(float(image_l[lines, block[1]][sel].mean()))
        height = lines.stop - lines.start
        for _ in range(ncontrol):
            start = int(rng.integers(BORDER, image_l.shape[0] - BORDER - height))
            if abs(start - lines.start) < 2 * height:
                continue
            sel = usable_l[start:start + height, block[1]]
            if sel.sum() > 50:
                control_means.append(float(image_l[start:start + height, block[1]][sel].mean()))
    return event_means, control_means


def compare_cleans(a, b, groupdq, background, events, blocks, fastaxis, group_time, rng):
    """Summarise ``a - b`` (two cleaned ramps of the same data) per difference and in the rate."""
    nints, ngroups = a.shape[:2]
    inside = interior(a.shape[-2:])
    sq, count, peak = 0.0, 0, 0.0
    event_means, control_means = [], []
    for i in range(nints):
        for j in range(ngroups):
            change = difference(a[i], j) - difference(b[i], j)
            flagged = groupdq[i, j] != 0
            near = ndimage.binary_dilation(flagged, iterations=5)
            usable = background & inside & ~near
            values = change[usable]
            if values.size == 0:
                continue
            sq += float(np.sum(values**2))
            count += values.size
            peak = max(peak, float(np.max(np.abs(values))))
            here = [(box, mask) for (ei, ej, box, mask) in events if ei == i and ej == j]
            if here:
                ev, ctl = band_means(change, usable, here, blocks, fastaxis, rng)
                event_means += ev
                control_means += ctl
    rate_a = np.mean([slope_per_group(a[i], groupdq[i]) for i in range(nints)], axis=0) / group_time
    rate_b = np.mean([slope_per_group(b[i], groupdq[i]) for i in range(nints)], axis=0) / group_time
    rate_change = rate_a - rate_b
    any_flag = np.any(groupdq != 0, axis=(0, 1))
    rate_usable = background & inside & ~ndimage.binary_dilation(any_flag, iterations=5)
    rate_values = rate_change[rate_usable]
    rate_event, rate_control = [], []
    for i in range(nints):
        here = [(box, mask) for (ei, _ej, box, mask) in events if ei == i]
        if here:
            ev, ctl = band_means(rate_change, rate_usable, here, blocks, fastaxis, rng)
            rate_event += ev
            rate_control += ctl
    return {
        "difference_rms_dn": float(np.sqrt(sq / max(count, 1))),
        "difference_max_abs_dn": peak,
        "event_band_mean_dn": summarise(event_means),
        "control_band_mean_dn": summarise(control_means),
        "rate_rms_dn_s": float(np.sqrt(np.mean(rate_values**2))),
        "rate_max_abs_dn_s": float(np.max(np.abs(rate_values))),
        "rate_event_band_mean_dn_s": summarise(rate_event),
        "rate_control_band_mean_dn_s": summarise(rate_control),
    }, rate_change


def flat_fielded(rate, flat):
    """``rate / flat`` where the flat is usable, NaN elsewhere; ``rate`` itself when ``flat`` is None (a dark)."""
    if flat is None:
        return rate
    ok = np.isfinite(flat) & (flat > 0)
    return np.where(ok, rate / np.where(ok, flat, 1.0), np.nan)


def rate_quality(rate, background, groupdq, blocks, fastaxis, flat=None):
    """Pixel rms, readout-line mean rms (per channel) and column mean rms of a rate about its background median.

    Uses background pixels away from the border and from any flag, with
    pixels beyond 5 robust sigma of the median left out, in DN/s; on a sky
    field the rate is flat-fielded first, since sky times the detector
    response is not flat.
    """
    rate = flat_fielded(rate, flat)
    any_flag = np.any(groupdq != 0, axis=(0, 1))
    good = background & interior(rate.shape) & ~any_flag & np.isfinite(rate)
    values = rate[good]
    centre = np.median(values)
    spread = 1.4826 * np.median(np.abs(values - centre))
    good &= np.abs(rate - centre) < 5 * spread
    residual = np.where(good, rate - centre, 0.0)
    residual_l, good_l = lines_view(residual, fastaxis), lines_view(good, fastaxis)
    line_means = []
    for block in blocks:
        block_l = block if abs(fastaxis) == 1 else block[::-1]
        n = good_l[:, block_l[1]].sum(axis=1)
        s = residual_l[:, block_l[1]].sum(axis=1)
        line_means.append(s[n > 50] / n[n > 50])
    line_means = np.concatenate(line_means)
    n = good_l.sum(axis=0)
    column_means = residual_l.sum(axis=0)[n > 50] / n[n > 50]
    return {
        "pixel_rms_dn_s": float(np.sqrt(np.mean(residual[good] ** 2))),
        "line_mean_rms_dn_s": float(np.sqrt(np.mean(line_means**2))),
        "column_mean_rms_dn_s": float(np.sqrt(np.mean(column_means**2))),
    }


def second_jump_pass(raw, cube, groupdq, threshold):
    """A second jump step on a cleaned cube, James's settings at ``threshold`` and JumpPlus on its new flags only.

    The first-pass flags stay as they are (running JumpPlusStep again would
    carry them one more group); new flags are also set in the next group when
    groups average more than one frame, as JumpPlusStep does.
    """
    model = with_groupdq(raw, groupdq)
    model.data = cube
    out = JumpStep.call(
        model,
        rejection_threshold=threshold,
        four_group_rejection_threshold=max(5.0, threshold),
        three_group_rejection_threshold=max(6.0, threshold),
        flag_4_neighbors=True,
        min_jump_to_flag_neighbors=20.0,
        after_jump_flag_time1=0.0,
        expand_large_events=False,
    )
    new = (np.asarray(out.groupdq) & JUMP) & ~(groupdq & JUMP)
    out.close()
    model.close()
    if raw.meta.exposure.nframes > 1:
        carried = np.roll(new, 1, axis=1)
        carried[:, 0] = 0
        new = new | carried
    return (groupdq | new).astype(groupdq.dtype)


class _Messages(logging.Handler):
    """Collect the messages of one logger."""

    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def reset_fit(raw, cube, zeroframe, groupdq, fit_params, background, blocks, fastaxis, flat=None):
    """ResetRampFitStep on a cleaned cube: its new-jump log line, flagged fraction and rate outliers.

    ``fit_params`` are passed to the step.  Outliers are background pixels
    without an input flag whose (flat-fielded) rate is more than 5 of its own
    err away from the median of a 5x5 neighbourhood; unflagged jumps make the
    upward ones outnumber the downward ones.  This count leaves out the
    pixels the fit flagged, which favours a run that flags more; see
    `common_outliers` for the count that does not.  The rate, its err (both
    flat-fielded) and the pixels with JUMP_DET in the rate DQ are returned
    as well, for `common_outliers`.
    """
    model = with_groupdq(raw, groupdq)
    model.data = cube
    if zeroframe is not None:
        model.zeroframe = zeroframe
    handler = _Messages()
    logger = logging.getLogger("snowblind.reset_rampfit")
    logger.addHandler(handler)
    try:
        rate, rateints = ResetRampFitStep.call(model, **fit_params)
    finally:
        logger.removeHandler(handler)
    model.close()
    data, err, dq = np.asarray(rate.data), np.asarray(rate.err), np.asarray(rate.dq)
    rate.close()
    rateints.close()
    data, err = flat_fielded(data, flat), flat_fielded(err, flat)
    input_jump = np.any((groupdq & JUMP) > 0, axis=(0, 1))
    any_input_flag = np.any(groupdq != 0, axis=(0, 1))
    good = background & interior(data.shape) & ~any_input_flag & np.isfinite(data) & (err > 0)
    z = (data - ndimage.median_filter(np.where(np.isfinite(data), data, 0.0), size=5)) / np.where(err > 0, err, 1.0)
    fit_jump = ((dq & dqflags.pixel["JUMP_DET"]) > 0) & ~input_jump
    found = [m for m in handler.messages if "jump search found" in m]
    return {
        "log": found[0] if found else None,
        "new_jump_pixels": int(fit_jump.sum()),
        "jump_fraction": float(((dq & dqflags.pixel["JUMP_DET"]) > 0).mean()),
        "outliers_up": int(np.count_nonzero(good & ~fit_jump & (z > 5))),
        "outliers_down": int(np.count_nonzero(good & ~fit_jump & (z < -5))),
        "quality": rate_quality(data, background, groupdq, blocks, fastaxis),
        "rate": data.astype(np.float32),
        "err": err.astype(np.float32),
        "flagged": (dq & dqflags.pixel["JUMP_DET"]) > 0,
    }


# pairs of runs compared pixel by pixel in common_outliers: (run, against)
OUTLIER_PAIRS = (
    ("high_default", "high_off"),
    ("high_5", "high_off"),
    ("high_4", "high_default"),
    ("high_twopoint4_default", "high_default"),
    ("user_default", "high_default"),
    ("user_4", "high_4"),
    ("user_default_nozeroframe", "user_default"),
)


def common_outliers(rates, errs, flagged, reference, background, reference_groupdq):
    """Rate outliers of every run on the same pixels and with the same noise scale.

    The pixels are the background pixels away from the border that
    ``reference``'s input DQ leaves without any flag, with a finite rate in
    every run; the noise scale is ``reference``'s err.  z is a run's rate
    minus the median of its 5x5 neighbourhood over that err.  Pixels a run
    flagged stay in, so a pixel flagged but not fixed counts as a failure,
    and the extra noise of dropped differences shows as extra outliers
    rather than as a larger err.

    For each pair of `OUTLIER_PAIRS` the pixels one run fixes and the other
    does not: an outlier (|z| > 5) in the second run with |z| < 3 in the
    first is fixed by the first, the reverse is new in the first; the gap
    keeps pixels near 5 that move by a little from counting.

    Returns
    -------
    per_run : dict
        Per run: pixels, up (z > 5), down (z < -5), the same among pixels
        the run flagged (up_flagged, down_flagged), and the robust sigma of z.
    pairs : dict
        Per pair: fixed_up, fixed_down, new_up, new_down.
    """
    scale = errs[reference]
    usable = background & interior(scale.shape) & ~np.any(reference_groupdq != 0, axis=(0, 1)) & (scale > 0)
    for rate in rates.values():
        usable &= np.isfinite(rate)
    z = {}
    per_run = {}
    for name, rate in rates.items():
        local = ndimage.median_filter(np.where(np.isfinite(rate), rate, 0.0), size=5)
        z[name] = np.where(usable, (rate - local) / np.where(scale > 0, scale, 1.0), 0.0)
        values = z[name][usable]
        up, down = usable & (z[name] > 5), usable & (z[name] < -5)
        per_run[name] = {
            "pixels": int(usable.sum()),
            "up": int(up.sum()),
            "down": int(down.sum()),
            "up_flagged": int((up & flagged[name]).sum()),
            "down_flagged": int((down & flagged[name]).sum()),
            "z_sigma": float(1.4826 * np.median(np.abs(values - np.median(values)))),
        }
    pairs = {}
    for a, b in OUTLIER_PAIRS:
        if a not in z or b not in z:
            continue
        fixed = usable & (np.abs(z[b]) > 5) & (np.abs(z[a]) < 3)
        new = usable & (np.abs(z[a]) > 5) & (np.abs(z[b]) < 3)
        pairs[f"{a}_vs_{b}"] = {
            "fixed_up": int((fixed & (z[b] > 0)).sum()),
            "fixed_down": int((fixed & (z[b] < 0)).sum()),
            "new_up": int((new & (z[a] > 0)).sum()),
            "new_down": int((new & (z[a] < 0)).sum()),
        }
    return per_run, pairs


def summarise(values):
    """Count, mean, rms and the most negative and positive value of a list."""
    if not values:
        return {"n": 0}
    v = np.asarray(values)
    return {"n": int(v.size), "mean": float(v.mean()), "rms": float(np.sqrt(np.mean(v**2))),
            "min": float(v.min()), "max": float(v.max())}


def second_pass_dq(groupdq):
    """First-pass jump flags turned into DO_NOT_USE, so that a second pass neither uses nor re-finds them."""
    dq = groupdq.copy()
    dq[(dq & JUMP) > 0] |= DNU
    dq &= np.uint8(~JUMP & 0xFF)
    return dq


def two_point_flags(model, threshold):
    """The jump step's two-point test alone at ``threshold``: new JUMP_DET flags per group, bool."""
    out = JumpStep.call(
        model.copy(),
        rejection_threshold=threshold,
        four_group_rejection_threshold=max(5.0, threshold),
        three_group_rejection_threshold=max(6.0, threshold),
        flag_4_neighbors=False,
        expand_large_events=False,
        after_jump_flag_dn1=0.0,
        after_jump_flag_time1=0.0,
        after_jump_flag_dn2=0.0,
        after_jump_flag_time2=0.0,
    )
    flags = (np.asarray(out.groupdq) & JUMP) > 0
    out.close()
    return flags


def likelihood_flags(model, threshold, readnoise_2d, gain_2d, rows):
    """stcal's likelihood jump search (as in ``likely_ramp_fit``, no zeroframe) at ``threshold`` on ``rows``."""
    ramp_data = create_ramp_fit_class(model, "LIKELY", dqflags.pixel, suppress_one_group=False)
    covar = Covar(get_readtimes(ramp_data))
    readnoise = readnoise_2d / np.sqrt(2.0)
    one = threshold**2
    two = scipy.stats.chi2.isf(scipy.special.erfc(threshold / np.sqrt(2.0)), 2)
    flags = np.zeros(model.data.shape, dtype=bool)
    for i in range(model.data.shape[0]):
        data = ramp_data.data[i]
        gdq = ramp_data.groupdq[i]
        for row in range(rows.start, rows.stop):
            diff = (data[1:, row] - data[:-1, row]) / covar.delta_t[:, np.newaxis]
            d2use = diffs2use_row(gdq[:, row])
            before = d2use.copy()
            d2use, _ = mask_jumps(
                diff, covar, readnoise[row], gain_2d[row],
                threshold_one_omit=one, threshold_two_omit=two, diffs2use=d2use,
            )
            flags[i, 1:, row, :] = (before ^ d2use) > 0
    return flags


def diffs2use_row(gdq_row):
    """stcal ``determine_diffs2use`` for one row: a difference is unusable when either of its groups has any flag.

    stcal allocates a full-frame array on every call; this is the same rule on the row alone.
    """
    d2use = np.ones((gdq_row.shape[0] - 1, gdq_row.shape[1]), dtype=np.uint8)
    d2use[gdq_row[1:] != 0] = 0
    d2use[gdq_row[:-1] != 0] = 0
    return d2use


def median_differences(data, groupdq):
    """Per pixel and integration, the median of the differences not touching a DO_NOT_USE or SATURATED group."""
    bad = (groupdq & (DNU | SAT)) > 0
    diffs = data[:, 1:].astype(np.float32) - data[:, :-1].astype(np.float32)
    usable = ~(bad[:, 1:] | bad[:, :-1])
    diffs[~usable] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(diffs, axis=1)
    return median, usable


def tally(flags, data, median, usable, first_jump, background, rows, fastaxis, blocks, threshold):
    """Counts of second-pass flags by sign, location and crowding, restricted to ``rows``."""
    region = np.zeros(background.shape, dtype=bool)
    region[rows] = True
    region &= interior(background.shape)
    out = {"threshold": threshold}
    totals = {k: 0 for k in ("bg_up", "bg_down", "src_up", "src_down", "wing_up", "wing_down", "bg_samples")}
    crowded_flags, all_bg_flags, crowded_lines = 0, 0, 0
    nints, ngroups = data.shape[:2]
    for i in range(nints):
        for g in range(1, ngroups):
            d = data[i, g].astype(np.float32) - data[i, g - 1].astype(np.float32)
            new = flags[i, g] & region
            up = new & (d > median[i])
            down = new & (d < median[i])
            near = ndimage.binary_dilation((first_jump[i, g]) > 0, structure=np.ones((3, 3), bool))
            totals["bg_up"] += int(np.count_nonzero(up & background & ~near))
            totals["bg_down"] += int(np.count_nonzero(down & background & ~near))
            totals["src_up"] += int(np.count_nonzero(up & ~background))
            totals["src_down"] += int(np.count_nonzero(down & ~background))
            totals["wing_up"] += int(np.count_nonzero(up & background & near))
            totals["wing_down"] += int(np.count_nonzero(down & background & near))
            totals["bg_samples"] += int(np.count_nonzero(usable[i, g - 1] & region & background))
            bg_new = lines_view(new & background & ~near, fastaxis)
            for block in blocks:
                block_l = block if abs(fastaxis) == 1 else block[::-1]
                per_line = bg_new[:, block_l[1]].sum(axis=1)
                crowded = per_line >= CROWDED_LINE
                crowded_flags += int(per_line[crowded].sum())
                crowded_lines += int(crowded.sum())
                all_bg_flags += int(per_line.sum())
    out.update(totals)
    out["expected_down"] = float(totals["bg_samples"] * 0.5 * scipy.special.erfc(threshold / np.sqrt(2.0)))
    out["real_estimate"] = totals["bg_up"] - totals["bg_down"]
    out["crowded_lines"] = crowded_lines
    out["crowded_fraction"] = float(crowded_flags / all_bg_flags) if all_bg_flags else 0.0
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file")
    parser.add_argument("--ngroups", type=int, default=None)
    parser.add_argument("--nints", type=int, default=1)
    parser.add_argument("--first-threshold", type=float, default=7.0)
    parser.add_argument("--thresholds", default="3.5,4,4.5,5,6")
    parser.add_argument("--rows", default=None, help="start:stop of the rows the second pass is tallied on")
    parser.add_argument("--event-area", type=int, default=200)
    parser.add_argument("--out", default=None)
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--reset-fit", action="store_true", help="also run ResetRampFitStep on the user and high chains")
    parser.add_argument(
        "--second-pass", type=float, action="append", default=[],
        help="with --reset-fit, also a chain with a second jump step at this threshold on the cleaned high ramp",
    )
    parser.add_argument("--reset-chains", default=None, help="with --reset-fit, comma-separated chain names to run (default all)")
    parser.add_argument(
        "--fits-only", action="store_true",
        help="only what --reset-fit needs: no clean without flags, no clean comparisons or OLS rates, no sweep (less memory)",
    )
    args = parser.parse_args()
    thresholds = [float(t) for t in args.thresholds.split(",")]
    rng = np.random.default_rng(20261028)

    raw = run_to_jump_input(args.file, args.ngroups, args.nints)
    assert not np.any(np.asarray(raw.groupdq) & JUMP), "the input to the first passes must carry no jump flags"
    detector = raw.meta.instrument.detector
    out_root = args.out or os.path.join("results", f"{datetime.date.today().isoformat()}_jump_order")
    out_dir = os.path.join(out_root, detector)
    os.makedirs(out_dir, exist_ok=True)
    fastaxis, slowaxis = raw.meta.subarray.fastaxis, raw.meta.subarray.slowaxis
    blocks = readout.channel_blocks(raw.data.shape[-2:], fastaxis, slowaxis)
    group_time = raw.meta.exposure.group_time
    dark = "DARK" in str(raw.meta.exposure.type).upper()
    nrows = raw.data.shape[-2]
    rows = slice(0, nrows) if args.rows is None else slice(*[int(v) for v in args.rows.split(":")])
    summary = {
        "file": os.path.basename(args.file), "detector": detector, "shape": list(raw.data.shape),
        "readpatt": raw.meta.exposure.readpatt, "nframes": raw.meta.exposure.nframes,
        "first_threshold": args.first_threshold, "rows": [rows.start, rows.stop],
    }

    t0 = time.time()
    dq_user = first_pass(raw, 5.0)
    dq_high = first_pass(raw, args.first_threshold)
    dq_none = np.array(raw.groupdq)
    for name, dq in (("user", dq_user), ("high", dq_high)):
        summary[f"first_pass_{name}_flags"] = int(np.count_nonzero((dq & JUMP) > 0))
    print(f"first passes done in {time.time() - t0:.0f} s", flush=True)

    t0 = time.time()
    clean_high, zf_high, background = clean(raw, dq_high, dark, out_dir, save_mask=True)
    clean_user, zf_user, _ = clean(raw, dq_user, dark, out_dir, save_mask=False)
    clean_none = None if args.fits_only else clean(raw, dq_none, dark, out_dir, save_mask=False)[0]
    print(f"cleans done in {time.time() - t0:.0f} s", flush=True)

    summary["background_fraction"] = float(background.mean())
    events = []
    if not args.fits_only:
        events = large_events(dq_user, args.event_area)
        summary["large_events"] = len(events)
        summary["none_vs_high"], rate_none_high = compare_cleans(
            clean_none, clean_high, dq_user, background, events, blocks, fastaxis, group_time, rng
        )
        summary["user_vs_high"], rate_user_high = compare_cleans(
            clean_user, clean_high, dq_user, background, events, blocks, fastaxis, group_time, rng
        )
        print(json.dumps({k: summary[k] for k in ("large_events", "none_vs_high", "user_vs_high")}, indent=1), flush=True)

    # each cleaned rate on its own, all with the user DQ: on a dark the truth is zero, on a sky field flat once
    # flat-fielded
    flat = None
    if not dark:
        with datamodels.FlatModel(StreamCleanStep().get_reference_file(raw, "flat")) as flat_model:
            flat = np.array(flat_model.data, dtype=np.float32)
    rates = {}
    summary["rate_quality"] = {}
    for name, cube in () if args.fits_only else (("high", clean_high), ("user", clean_user), ("none", clean_none)):
        rates[name] = np.mean([slope_per_group(cube[i], dq_user[i]) for i in range(cube.shape[0])], axis=0) / group_time
        summary["rate_quality"][name] = rate_quality(rates[name], background, dq_user, blocks, fastaxis, flat)
    print(json.dumps(summary["rate_quality"], indent=1), flush=True)

    fit_rates, fit_errs, fit_flagged = {}, {}, {}
    # only the reference chain's input DQ is kept, for the comparison of the chains
    reference_dq = None
    if args.reset_fit:
        summary["reset_fit"] = {}
        # the user chains first, so that their cleaned cube can go before the high ones run
        chains = [
            ("user_default", clean_user, zf_user, dq_user, {}),
            ("user_4", clean_user, zf_user, dq_user, {"rejection_threshold": 4.0}),
            ("user_default_nozeroframe", clean_user, zf_user, dq_user, {"use_zeroframe": False}),
            ("high_default", clean_high, zf_high, dq_high, {}),
            ("high_4", clean_high, zf_high, dq_high, {"rejection_threshold": 4.0}),
        ]
        # the fit's own search higher, and in effect off (no jump the 7 sigma pass missed is that large)
        chains.append(("high_5", clean_high, zf_high, dq_high, {"rejection_threshold": 5.0}))
        chains.append(("high_off", clean_high, zf_high, dq_high, {"rejection_threshold": 1000.0}))
        for k in args.second_pass:
            dq_second = second_jump_pass(raw, clean_high, dq_high, k)
            chains.append((f"high_twopoint{k:g}_default", clean_high, zf_high, dq_second, {}))
        wanted = None if args.reset_chains is None else set(args.reset_chains.split(","))
        if args.fits_only:
            del clean_user, zf_user
        while chains:
            name, cube, zf, dq, fit_params = chains.pop(0)
            if wanted is not None and name not in wanted:
                continue
            record = reset_fit(raw, cube, zf, dq, fit_params, background, blocks, fastaxis, flat)
            del cube, zf
            fit_rates[name] = record.pop("rate")
            fit_errs[name] = record.pop("err")
            fit_flagged[name] = record.pop("flagged")
            if name == "high_default":
                reference_dq = dq
            summary["reset_fit"][name] = record
            print(name, json.dumps(record), flush=True)
        if "high_default" in fit_rates:
            per_run, pairs = common_outliers(
                fit_rates, fit_errs, fit_flagged, "high_default", background, reference_dq
            )
            summary["reset_fit_common"] = {"reference": "high_default", "per_run": per_run, "pairs": pairs}
            print(json.dumps(summary["reset_fit_common"], indent=1), flush=True)

    cutouts = {f"rate_{name}": rate.astype(np.float32) for name, rate in rates.items()}
    cutouts.update({f"fit_rate_{name}": rate for name, rate in fit_rates.items()})
    cutouts.update({f"fit_err_{name}": err for name, err in fit_errs.items()})
    cutouts.update({f"fit_flagged_{name}": np.packbits(flags) for name, flags in fit_flagged.items()})
    biggest = sorted(events, key=lambda e: -int(e[3].sum()))[:3]
    for n, (i, j, box, _mask) in enumerate(biggest):
        cy = (box[0].start + box[0].stop) // 2
        cx = (box[1].start + box[1].stop) // 2
        y0, x0 = max(cy - 128, 0), max(cx - 128, 0)
        cut = (slice(y0, y0 + 256), slice(x0, x0 + 256))
        cutouts[f"event{n}_difference"] = (difference(clean_none[i], j) - difference(clean_high[i], j))[cut].astype(np.float32)
        cutouts[f"event{n}_rate"] = rate_none_high[cut].astype(np.float32)
        cutouts[f"event{n}_raw"] = difference(np.asarray(raw.data[i]), j)[cut].astype(np.float32)
        cutouts[f"event{n}_flags"] = dq_user[i, j][cut]
        cutouts[f"event{n}_origin"] = np.array([i, j, y0, x0])
    if not args.fits_only:
        cutouts["rate_none_minus_high"] = rate_none_high.astype(np.float32)
        cutouts["rate_user_minus_high"] = rate_user_high.astype(np.float32)
    np.savez_compressed(os.path.join(out_dir, "cutouts.npz"), **cutouts)
    clean_user = clean_none = None

    if not (args.skip_sweep or args.fits_only):
        with datamodels.ReadnoiseModel(JumpStep().get_reference_file(raw, "readnoise")) as rn_model, \
                datamodels.GainModel(JumpStep().get_reference_file(raw, "gain")) as gain_model:
            readnoise_2d, gain_2d = get_reference_file_subarrays(raw, rn_model, gain_model)
        dq2 = second_pass_dq(dq_high)
        first_jump = (dq_high & JUMP) > 0
        sweep = {}
        # every second-pass flag, for the per-pixel follow-up: (integration, group, row, column) and the
        # flagged difference minus the pixel's median difference
        stored = {
            "pixeldq": np.asarray(raw.pixeldq), "readnoise": readnoise_2d.astype(np.float32),
            "background": background, "first_jump": np.argwhere(first_jump).astype(np.int16),
        }
        for name, data in (("raw", np.asarray(raw.data, dtype=np.float32)), ("cleaned", clean_high)):
            model = with_groupdq(raw, dq2)
            model.data = data
            median, usable = median_differences(data, dq2)
            sweep[name] = {"two_point": [], "likelihood": []}
            for k in thresholds:
                for detector_name in ("two_point", "likelihood"):
                    t0 = time.time()
                    if detector_name == "two_point":
                        flags = two_point_flags(model, k)
                    else:
                        flags = likelihood_flags(model, k, readnoise_2d, gain_2d, rows)
                    row = tally(flags, data, median, usable, first_jump, background, rows, fastaxis, blocks, k)
                    row["seconds"] = round(time.time() - t0, 1)
                    sweep[name][detector_name].append(row)
                    print(name, detector_name, json.dumps(row), flush=True)
                    where = np.argwhere(flags[:, 1:])
                    i, g, y, x = where.T
                    excess = data[i, g + 1, y, x] - data[i, g, y, x] - median[i, y, x]
                    stored[f"{name}_{detector_name}_{k:g}_where"] = (where + [0, 1, 0, 0]).astype(np.int16)
                    stored[f"{name}_{detector_name}_{k:g}_excess"] = excess.astype(np.float32)
            model.close()
        summary["sweep"] = sweep
        np.savez_compressed(os.path.join(out_dir, "flags.npz"), **stored)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()

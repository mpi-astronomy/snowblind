"""Likelihood ramp fitting with the reset level as an extra resultant.

Forked from ``stcal/src/stcal/ramp_fitting/likely_fit.py`` at stcal 1.19.0
(commit 6cddca3, 2026-06-23): ``likely_ramp_fit``, ``mask_jumps``,
``fit_ramps`` and ``compute_alphas_betas``.  Everything else is imported
from stcal.  Changes, all inside the forked functions (plan section B.2):

1. Read times get a first entry for the reset resultant (``N = 1``) at
   time ``(1 - t1_scale) * t_1``, which is zero for the nominal timing.  A
   ``t1_scale`` below one models a first read that holds less signal than
   its nominal interval predicts, without touching any later interval.
2. A plane holding the prior mean of the reset level (zero by default, or
   a per-pixel superbias-error map) is prepended to the data of each
   integration, with a DQ plane that is good except where the pixel DQ has
   DO_NOT_USE.
3. When a zeroframe is used, its DQ is SATURATED where the zeroframe value
   is exactly zero (the pipeline's marker for an unusable zeroframe pixel)
   and good elsewhere, instead of a copy of group 1's DQ.
4. The read-noise variance of the first difference becomes
   ``(sigma_b**2 + sigma_r**2 / N_1) / dt_0**2``, written into the first
   entry of ``alpha_readnoise`` in ``compute_alphas_betas_reset``.  A
   scalar ``sigma_b`` is broadcast to every pixel.  (The plan's scalar
   prototype used the reset resultant's effective read count instead; that
   made the prior scale with each pixel's read noise and was retired once
   the per-pixel form existed.)
5. A dropped first difference is recorded in a ``used_reset`` plane rather
   than as a JUMP_DET flag on the first group, and the normalized residual
   of the first difference is returned as ``d0_resid``.

Snowball expansion (``flag_large_events``) is not carried over.
"""
import copy
import logging
import warnings

import numpy as np
import scipy.special
import scipy.stats
from stcal.ramp_fitting import utils
from stcal.ramp_fitting.likely_algo_classes import Covar, IntegInfo
from stcal.ramp_fitting.likely_fit import (
    SQRT2,
    compute_image_info,
    compute_jump_detects,
    compute_PhiDs,
    compute_Phis,
    compute_phis,
    compute_ThetaDs,
    compute_Thetas,
    compute_thetas,
    determine_diffs2use,
    get_ramp_result,
    get_readtimes,
    initial_count_rate_guess,
    matrix_computations,
)

__all__ = [
    "likely_ramp_fit_reset",
    "likely_ramp_fit_reset_blocks",
    "ResetIntegInfo",
    "MIN_NGROUPS",
    "STCAL_PINNED_COMMIT",
]

log = logging.getLogger(__name__)

STCAL_PINNED_COMMIT = "6cddca3"
# stcal needs four resultants; the reset resultant supplies one of them.
MIN_NGROUPS = 3


class ResetIntegInfo(IntegInfo):
    """IntegInfo with two extra planes: whether the reset difference was used, and its residual."""

    def __init__(self, nints, nrows, ncols):
        """Allocate stcal's per-integration output arrays plus the two reset planes.

        Parameters
        ----------
        nints : int
            Number of integrations.
        nrows : int
            Number of image rows.
        ncols : int
            Number of image columns.
        """
        super().__init__(nints, nrows, ncols)
        self.used_reset = np.zeros((nints, nrows, ncols), dtype=bool)
        self.d0_resid = np.full((nints, nrows, ncols), np.nan, dtype=np.float32)

    def prepare_info(self):
        """Arrange output arrays as a dict, as the ramp fit step expects, plus the two extra planes.

        Returns
        -------
        dict
            stcal's ``integ_info`` dictionary with the added keys
            ``used_reset`` (bool, shape (nints, nrows, ncols)) and
            ``d0_resid`` (float32, same shape).
        """
        info = super().prepare_info()
        info["used_reset"] = self.used_reset
        info["d0_resid"] = self.d0_resid
        return info

    def get_results(self, result, integ, row, used_reset=None):
        """Store a row's results, including the reset diagnostics.

        Parameters
        ----------
        result : RampResult
            Fit of one row from `fit_ramps_reset`; its ``d0_resid``
            attribute, when present, is copied into the ``d0_resid`` plane.
        integ : int
            Integration index.
        row : int
            Row index.
        used_reset : ndarray or None, shape (ncols,)
            Whether the reset difference was kept for each pixel of the row;
            None leaves the ``used_reset`` plane unchanged.
        """
        super().get_results(result, integ, row)
        if used_reset is not None:
            self.used_reset[integ, row, :] = used_reset.astype(bool)
        if getattr(result, "d0_resid", None) is not None:
            self.d0_resid[integ, row, :] = result.d0_resid


def likely_ramp_fit_reset(
    ramp_data, readnoise_2d, gain_2d, sigma_b, t1_scale=1.0, use_zeroframe=True, reset_level=0.0
):
    """Likelihood ramp fit with a Gaussian prior on the reset level.

    Parameters
    ----------
    ramp_data : RampData
        As built by ``stcal.ramp_fitting.ramp_fit.create_ramp_fit_class``.
    readnoise_2d : ndarray
        CDS read noise per pixel, DN (the readnoise reference file).
    gain_2d : ndarray
        Gain per pixel, e/DN.
    sigma_b : float or ndarray
        Reset-level scatter about the superbias in DN, a scalar or an
        (nrows, ncols) array.
    t1_scale : float
        Fraction of the nominal first interval over which the first read
        actually accumulates signal; the reset resultant is placed at
        ``(1 - t1_scale)`` times the first group's mean time.
    use_zeroframe : bool
        Use ``ramp_data.zeroframe`` when present and the first group has at
        least two reads.
    reset_level : float or ndarray
        Mean of the reset-level prior in DN: the value of the prepended
        resultant.  Zero after superbias subtraction, or a per-pixel map of
        the superbias error, optionally with a constant offset.

    Returns
    -------
    image_info : dict
        As stcal's ``likely_ramp_fit``.
    integ_info : dict
        As stcal's, plus ``used_reset`` (bool) and ``d0_resid`` (float32).
    opt_info : None
    """
    nints, ngroups, nrows, ncols = ramp_data.data.shape
    if ngroups < MIN_NGROUPS:
        raise ValueError(f"Reset-anchored likelihood fit requires at least {MIN_NGROUPS} groups.")

    readtimes = [np.atleast_1d(np.asarray(times, dtype=float)) for times in get_readtimes(ramp_data)]
    first_group_time = float(np.mean(readtimes[0]))

    zeroframe_ok = False
    if use_zeroframe and ramp_data.zeroframe is not None:
        if len(readtimes[0]) < 2:
            log.warning("Zero frame is present, but the first group has < 2 reads; it will be ignored.")
        else:
            rt_zero = readtimes[0][0]
            readtimes[0] = readtimes[0][1:]
            readtimes.insert(0, np.array([rt_zero]))
            zeroframe_ok = True

    # Change 1: the reset resultant.  With t1_scale = k < 1 the first read
    # holds k times the signal the nominal interval predicts, as if the
    # integration started (1 - k) t_1 after the nominal reset.  Placing the
    # reset resultant there shortens only the reset interval; every later
    # interval is unchanged.  Scaling the first group's read times instead
    # would also stretch the first ordinary interval and bias that difference.
    reset_time = (1.0 - t1_scale) * first_group_time
    first_time = float(readtimes[0][0])
    if reset_time >= first_time:
        log.warning(
            f"First-read scale {t1_scale:.3f} puts the reset {reset_time:.2f} s after the nominal reset, at or after "
            f"the first resultant at {first_time:.2f} s; placing it at half that time instead"
        )
        reset_time = 0.5 * first_time
    readtimes.insert(0, np.array([reset_time]))
    covar = Covar(readtimes)

    readnoise_2d = np.asarray(readnoise_2d, dtype=float) / SQRT2
    sigma_b = np.asarray(sigma_b, dtype=float)
    if sigma_b.ndim == 2 and sigma_b.shape != (nrows, ncols):
        raise ValueError(f"sigma_b shape {sigma_b.shape} does not match the data {(nrows, ncols)}")
    # Change 4: a scalar sigma_b is the same prior for every pixel.  The
    # plan's scalar prototype folded it into the reset resultant's effective
    # read count, which made the prior scale with each pixel's read noise;
    # broadcasting into the per-pixel form keeps one physics for both cases.
    sigma_b = np.broadcast_to(sigma_b, (nrows, ncols)).astype(float)
    reset_level = np.asarray(reset_level, dtype=np.float32)
    if reset_level.ndim == 2 and reset_level.shape != (nrows, ncols):
        raise ValueError(f"reset_level shape {reset_level.shape} does not match the data {(nrows, ncols)}")
    reset_plane = np.broadcast_to(reset_level, (nrows, ncols)).astype(np.float32)

    integ_class = ResetIntegInfo(nints, nrows, ncols)

    for integ in range(nints):
        data = ramp_data.data[integ]
        gdq = ramp_data.groupdq[integ].copy()
        pdq = ramp_data.pixeldq.copy()

        if zeroframe_ok:
            zf = ramp_data.zeroframe[integ]
            # Change 3: zeroframe DQ from the zeroframe itself.
            zfdq = np.where(zf == 0, ramp_data.flags_saturated, 0).astype(gdq.dtype)
            first_group = (data[0] * ramp_data.nframes - zf) / (ramp_data.nframes - 1)
            data = np.vstack([zf[None, :, :], first_group[None, :, :], data[1:]])
            gdq = np.vstack([zfdq[None, :, :], gdq])

        # Change 2: the reset plane.
        reset_dq = np.where(
            (pdq & ramp_data.flags_do_not_use) > 0, ramp_data.flags_do_not_use, 0
        ).astype(gdq.dtype)
        data = np.vstack([reset_plane[None, :, :], data])
        gdq = np.vstack([reset_dq[None, :, :], gdq])

        diff = (data[1:] - data[:-1]) / covar.delta_t[:, np.newaxis, np.newaxis]
        alldiffs2use = np.ones(diff.shape, np.uint8)
        allrateguesses = np.zeros((nrows, ncols))

        for row in range(nrows):
            d2use = determine_diffs2use(row, diff, gdq)
            d2use_copy = d2use.copy()
            sb_row = sigma_b[row]
            if ramp_data.rejection_threshold is not None:
                threshold_one_omit = ramp_data.rejection_threshold**2
                pval = scipy.special.erfc(ramp_data.rejection_threshold / SQRT2)
                threshold_two_omit = scipy.stats.chi2.isf(pval, 2)
                if np.isinf(threshold_two_omit):
                    threshold_two_omit = threshold_one_omit + 10
                d2use, countrates = mask_jumps_reset(
                    diff[:, row],
                    covar,
                    readnoise_2d[row],
                    gain_2d[row],
                    sigma_b=sb_row,
                    threshold_one_omit=threshold_one_omit,
                    threshold_two_omit=threshold_two_omit,
                    diffs2use=d2use,
                )
            else:
                d2use, countrates = mask_jumps_reset(
                    diff[:, row], covar, readnoise_2d[row], gain_2d[row], sigma_b=sb_row, diffs2use=d2use
                )

            # Change 5: a dropped reset difference is not a jump on the first group.
            jump_locs = d2use_copy ^ d2use
            jump_locs[jump_locs > 0] = ramp_data.flags_jump_det
            gdq[2:, row, :] |= jump_locs[1:]

            alldiffs2use[:, row] = d2use
            allrateguesses[row] = countrates * (countrates > 0) + ramp_data.average_dark_current[row, :]

        for row in range(nrows):
            result = fit_ramps_reset(
                diff[:, row],
                covar,
                gain_2d[row],
                readnoise_2d[row],
                sigma_b=sigma_b[row],
                diffs2use=alldiffs2use[:, row],
                count_rate_guess=allrateguesses[row],
            )
            integ_class.get_results(result, integ, row, used_reset=alldiffs2use[0, row])

        # The reset plane is not a group: leave it out of the DQ compression.
        pdq = utils.dq_compress_sect(ramp_data, gdq[1:], pdq)
        integ_class.dq[integ, :, :] = pdq
        del gdq

    integ_info = integ_class.prepare_info()
    image_info = compute_image_info(integ_class, ramp_data)
    return image_info, integ_info, None


def likely_ramp_fit_reset_blocks(
    ramp_data, readnoise_2d, gain_2d, sigma_b, blocks, t1_scales, reset_levels, use_zeroframe=True
):
    """Run `likely_ramp_fit_reset` on image blocks with their own timing parameters and stitch the results.

    Used for per-channel ``t1_scale`` and reset offset: the covariance
    coefficients depend on the reset interval, which the fitter holds once
    per call, so each channel is fitted separately.

    Parameters
    ----------
    ramp_data : RampData
        Full-frame ramp data as built by
        ``stcal.ramp_fitting.ramp_fit.create_ramp_fit_class``; a shallow
        copy with its arrays sliced is handed to each block fit.
    readnoise_2d : ndarray, shape (nrows, ncols)
        CDS read noise per pixel, DN.
    gain_2d : ndarray, shape (nrows, ncols)
        Gain per pixel, e/DN.
    sigma_b : float or ndarray, shape (nrows, ncols)
        Reset-level scatter in DN, broadcast to the full frame and sliced
        per block.
    blocks : sequence of (slice, slice)
        Science-frame ``(rows, cols)`` of each block.
    t1_scales : sequence of float
        One per block.
    reset_levels : sequence of float or ndarray
        One per block; an array is the full-frame prior mean, sliced here.
    use_zeroframe : bool
        Passed to `likely_ramp_fit_reset`.

    Returns
    -------
    image_info, integ_info, None
        As `likely_ramp_fit_reset`, assembled over the full frame.
    """
    nints, _ngroups, nrows, ncols = ramp_data.data.shape
    sigma_b = np.broadcast_to(np.asarray(sigma_b, dtype=float), (nrows, ncols))
    image_info, integ_info = None, None
    for (rows, cols), t1_scale, reset_level in zip(blocks, t1_scales, reset_levels):
        sub = copy.copy(ramp_data)
        sub.data = ramp_data.data[:, :, rows, cols]
        sub.groupdq = ramp_data.groupdq[:, :, rows, cols]
        sub.pixeldq = ramp_data.pixeldq[rows, cols]
        sub.average_dark_current = ramp_data.average_dark_current[rows, cols]
        if ramp_data.zeroframe is not None:
            sub.zeroframe = ramp_data.zeroframe[:, rows, cols]
        level = np.asarray(reset_level, dtype=np.float32)
        if level.ndim == 2:
            level = level[rows, cols]
        block_image, block_integ, _ = likely_ramp_fit_reset(
            sub,
            np.asarray(readnoise_2d)[rows, cols],
            np.asarray(gain_2d)[rows, cols],
            sigma_b[rows, cols],
            t1_scale=t1_scale,
            use_zeroframe=use_zeroframe,
            reset_level=level,
        )
        if image_info is None:
            image_info = {k: np.zeros((nrows, ncols), dtype=v.dtype) for k, v in block_image.items()}
            integ_info = {k: np.zeros((nints, nrows, ncols), dtype=v.dtype) for k, v in block_integ.items()}
        for key, value in block_image.items():
            image_info[key][rows, cols] = value
        for key, value in block_integ.items():
            integ_info[key][:, rows, cols] = value
    return image_info, integ_info, None


def mask_jumps_reset(
    diffs,
    covar,
    rnoise,
    gain,
    sigma_b=None,
    threshold_one_omit=20.25,
    threshold_two_omit=23.8,
    diffs2use=None,
):
    """Likelihood-based iterative jump detection (stcal ``mask_jumps`` with ``sigma_b`` threaded through).

    Parameters
    ----------
    diffs : ndarray, shape (ndiffs, ncols)
        Resultant differences of one row divided by their time intervals,
        DN/s; the first entry is the reset difference.
    covar : Covar
        Read-time covariance coefficients from stcal's ``Covar``, built with
        the reset resultant as the first read time.
    rnoise : ndarray, shape (ncols,)
        Single-read noise per pixel of the row, DN.
    gain : ndarray, shape (ncols,)
        Gain per pixel of the row, e/DN.
    sigma_b : ndarray of shape (ncols,) or None
        Per-pixel reset scatter for the row; None leaves stcal's read-noise
        term for the first difference unchanged.
    threshold_one_omit : float
        Minimum chi-squared improvement for dropping one difference.
    threshold_two_omit : float
        Minimum chi-squared improvement for dropping two adjacent
        differences.
    diffs2use : ndarray of shape (ndiffs, ncols) or None
        Initial mask of usable differences (1 usable, 0 not); None uses all.
        Modified in place.

    Returns
    -------
    diffs2use : ndarray of uint8, shape (ndiffs, ncols)
        The mask with the detected jumps removed.
    countrate : ndarray, shape (ncols,)
        Count rate per pixel from the last fit, DN/s.
    """
    loc_diff = diffs

    one_omit_ok = covar.Nreads[1:] * covar.Nreads[:-1] >= 1
    one_omit_ok[0] = one_omit_ok[-1] = True
    two_omit_ok = covar.Nreads[1:-1] > 1

    if diffs2use is None:
        diffs2use = np.ones(loc_diff.shape, np.uint8)

    count_rate_guess = np.median(loc_diff, axis=0)[np.newaxis, :]
    count_rate_guess *= count_rate_guess > 0

    recheck = np.ones(loc_diff.shape[1]) == 1
    dropped = np.ones(loc_diff.shape[1]) == 0

    for j in range(loc_diff.shape[0]):
        if j == 0:
            result = fit_ramps_reset(
                loc_diff,
                covar,
                gain,
                rnoise,
                sigma_b=sigma_b,
                count_rate_guess=count_rate_guess,
                diffs2use=diffs2use,
                detect_jumps=True,
            )
            countrate = result.countrate * 1.0
        else:
            result = fit_ramps_reset(
                loc_diff[:, recheck],
                covar,
                gain[recheck],
                rnoise[recheck],
                sigma_b=None if sigma_b is None else sigma_b[recheck],
                count_rate_guess=count_rate_guess[:, recheck],
                diffs2use=diffs2use[:, recheck],
                detect_jumps=True,
            )

        dchisq_two = result.chisq - result.chisq_two_omit
        dchisq_one = result.chisq - result.chisq_one_omit

        best_dchisq_one = np.amax(dchisq_one * one_omit_ok[:, np.newaxis], axis=0)
        best_dchisq_two = np.amax(dchisq_two * two_omit_ok[:, np.newaxis], axis=0)

        onedropbetter = best_dchisq_one - threshold_one_omit > best_dchisq_two - threshold_two_omit

        best_dchisq = best_dchisq_one * (best_dchisq_one > threshold_one_omit) * onedropbetter
        best_dchisq += best_dchisq_two * (best_dchisq_two > threshold_two_omit) * (~onedropbetter)

        best_dchisq[best_dchisq == 0] = np.nan

        dropone = (dchisq_one == best_dchisq) & (onedropbetter)
        droptwo = (dchisq_two == best_dchisq) & (~onedropbetter)

        # With only two usable differences (one ordinary difference plus the
        # reset difference) the one-omit search is degenerate: dropping either
        # leaves a perfect fit.  stcal never sees this case because such a ramp
        # has a single difference there.  Keep both and let the weights decide.
        too_few = np.sum(diffs2use[:, recheck], axis=0) <= 2
        dropone[:, too_few] = False
        droptwo[:, too_few] = False

        drop = np.any([np.sum(dropone, axis=0), np.sum(droptwo, axis=0)], axis=0)

        if np.sum(drop) == 0:
            break

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_cts = np.zeros(np.sum(recheck))
            i_d1 = np.sum(dropone, axis=0) > 0
            new_cts[i_d1] = np.sum(result.countrate_one_omit * dropone, axis=0)[i_d1]
            i_d2 = np.sum(droptwo, axis=0) > 0
            new_cts[i_d2] = np.sum(result.countrate_two_omit * droptwo, axis=0)[i_d2]

        countrate[recheck] *= drop == 0
        countrate[recheck] += new_cts

        diffs2use[:, recheck] *= ~dropone
        diffs2use[:-1, recheck] *= ~droptwo
        diffs2use[1:, recheck] *= ~droptwo

        dropped[:] = False
        dropped[recheck] = drop
        recheck[:] = dropped

        recheck[np.sum(diffs2use, axis=0) <= 3] = False

    return diffs2use, countrate


def fit_ramps_reset(
    diffs,
    covar,
    gain,
    rnoise,
    sigma_b=None,
    count_rate_guess=None,
    diffs2use=None,
    detect_jumps=False,
    rescale=True,
    dn_scale=10.0,
):
    """Fit ramps on a row of data (stcal ``fit_ramps`` plus ``sigma_b`` and the ``d0_resid`` output).

    Parameters are as in stcal; ``sigma_b`` is the per-pixel reset scatter
    for the row or None for the scalar form.  The returned ``RampResult``
    carries ``d0_resid``, the first difference minus the fitted rate in
    units of the first difference's standard deviation, NaN where the first
    difference was not used.

    Parameters
    ----------
    diffs : ndarray, shape (ndiffs, ncols)
        Resultant differences of one row divided by their time intervals,
        DN/s; the first entry is the reset difference.
    covar : Covar
        Read-time covariance coefficients from stcal's ``Covar``.
    gain : ndarray, shape (ncols,)
        Gain per pixel of the row, e/DN.
    rnoise : ndarray, shape (ncols,)
        Single-read noise per pixel of the row, DN.
    sigma_b : ndarray of shape (ncols,) or None
        Per-pixel reset scatter for the row in DN, or None for the scalar
        form (stcal's unchanged read-noise term).
    count_rate_guess : ndarray or None, shape (ncols,) or (1, ncols)
        Count rate used for the photon-noise part of the covariance, DN/s;
        None takes stcal's initial guess from the differences.
    diffs2use : ndarray or None, shape (ndiffs, ncols)
        Mask of usable differences (1 usable, 0 not); None uses all.
    detect_jumps : bool
        Also compute the one- and two-omit chi-squared values that
        `mask_jumps_reset` searches.
    rescale : bool
        Rescale the covariance to keep the recursion numerically stable, as
        in stcal.
    dn_scale : float
        How often (in differences) the rescaling is applied.

    Returns
    -------
    RampResult
        stcal's result for the row with the extra attribute ``d0_resid``.
    """
    if diffs2use is None:
        diffs2use = np.ones(diffs.shape, np.uint8)

    if count_rate_guess is None:
        count_rate_guess = initial_count_rate_guess(diffs, diffs2use)

    alpha_tuple, beta_tuple, scale = compute_alphas_betas_reset(
        count_rate_guess, gain, rnoise, covar, rescale, diffs, dn_scale, sigma_b
    )
    alpha, alpha_phnoise, alpha_readnoise = alpha_tuple
    beta, beta_phnoise, beta_readnoise = beta_tuple

    ndiffs, npix = diffs.shape

    diff_mask = diffs * diffs2use
    beta = beta * diffs2use[1:] * diffs2use[:-1]

    theta = compute_thetas(ndiffs, npix, alpha, beta)
    phi = compute_phis(ndiffs, npix, alpha, beta)

    sgn = np.ones((ndiffs, npix))
    sgn[::2] = -1

    Phi = compute_Phis(ndiffs, npix, beta, phi, sgn)  # noqa: N806
    PhiD = compute_PhiDs(ndiffs, npix, beta, phi, sgn, diff_mask)  # noqa: N806
    Theta = compute_Thetas(ndiffs, npix, beta, theta, sgn)  # noqa: N806
    ThetaD = compute_ThetaDs(ndiffs, npix, beta, theta, sgn, diff_mask)  # noqa: N806

    dB, dC, A, B, C = matrix_computations(  # noqa: N806
        ndiffs, npix, sgn, diff_mask, diffs2use, beta, phi, Phi, PhiD, theta, Theta, ThetaD
    )

    result = get_ramp_result(dC, A, B, C, scale, alpha_phnoise, alpha_readnoise, beta_phnoise, beta_readnoise)

    # Change 5: normalized residual of the reset difference.
    with np.errstate(invalid="ignore", divide="ignore"):
        var0 = alpha_phnoise[0] + alpha_readnoise[0]
        resid = (diffs[0] - result.countrate) / np.sqrt(var0)
    result.d0_resid = np.where(diffs2use[0] > 0, resid, np.nan)

    if detect_jumps:
        result = compute_jump_detects(result, ndiffs, diffs2use, dC, dB, A, B, C, scale, beta, phi, theta)

    return result


def compute_alphas_betas_reset(count_rate_guess, gain, rnoise, covar, rescale, diffs, dn_scale, sigma_b=None):
    """Covariance diagonals (stcal ``compute_alphas_betas``) with the per-pixel reset prior.

    With ``sigma_b`` given, the read-noise part of the first diagonal
    element is replaced by ``(sigma_b**2 + rnoise**2 / N_1) / dt_0**2``.

    Parameters
    ----------
    count_rate_guess : ndarray, shape (ncols,) or (1, ncols)
        Count rate per pixel for the photon-noise terms, DN/s.
    gain : ndarray, shape (ncols,)
        Gain per pixel, e/DN.
    rnoise : ndarray, shape (ncols,)
        Single-read noise per pixel, DN.
    covar : Covar
        Read-time coefficients ``alpha_phnoise``, ``alpha_readnoise``,
        ``beta_phnoise``, ``beta_readnoise``, ``Nreads`` and ``delta_t``.
    rescale : bool
        Rescale ``alpha`` and ``beta`` by a running factor so that the
        theta recursion stays of order one.
    diffs : ndarray, shape (ndiffs, ncols)
        Differences; only their shape is used.
    dn_scale : float
        The rescaling factor is refreshed every ``int(dn_scale)`` differences.
    sigma_b : ndarray of shape (ncols,) or None
        Per-pixel reset scatter in DN; None keeps stcal's first element.

    Returns
    -------
    alpha_tuple : tuple of ndarray
        ``(alpha, alpha_phnoise, alpha_readnoise)``, the covariance diagonal
        and its photon- and read-noise parts, shape (ndiffs, ncols); only
        ``alpha`` is rescaled.
    beta_tuple : tuple of ndarray
        ``(beta, beta_phnoise, beta_readnoise)``, the off-diagonal and its
        parts, shape (ndiffs - 1, ncols); only ``beta`` is rescaled.
    scale : ndarray or int
        The rescaling factor per pixel, or 1 when ``rescale`` is False.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
        warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)

        alpha_phnoise = count_rate_guess / gain * covar.alpha_phnoise[:, np.newaxis]
        alpha_readnoise = rnoise**2 * covar.alpha_readnoise[:, np.newaxis]
        if sigma_b is not None:
            # Change 4, per-pixel form.
            alpha_readnoise = np.array(alpha_readnoise, dtype=float)
            alpha_readnoise[0] = (sigma_b**2 + rnoise**2 / covar.Nreads[1]) / covar.delta_t[0] ** 2
        alpha = alpha_phnoise + alpha_readnoise

        beta_phnoise = count_rate_guess / gain * covar.beta_phnoise[:, np.newaxis]
        beta_readnoise = rnoise**2 * covar.beta_readnoise[:, np.newaxis]
        beta = beta_phnoise + beta_readnoise

    ndiffs, npix = diffs.shape

    if rescale:
        theta = np.ones((ndiffs + 1, npix))
        theta[1] = alpha[0]

        scale = theta[0] * 1
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
            warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)

            for i in range(2, ndiffs + 1):
                theta[i] = alpha[i - 1] / scale * theta[i - 1] - beta[i - 2] ** 2 / scale**2 * theta[i - 2]

                if i % int(dn_scale) == 0 or i == ndiffs:
                    f = theta[i] ** (1 / i)
                    scale *= f
                    tmp = theta[i] / f
                    theta[i - 1] /= tmp
                    theta[i - 2] /= tmp / f
                    theta[i] = 1
    else:
        scale = 1

    alpha /= scale
    beta /= scale

    alpha_tuple = (alpha, alpha_phnoise, alpha_readnoise)
    beta_tuple = (beta, beta_phnoise, beta_readnoise)

    return alpha_tuple, beta_tuple, scale

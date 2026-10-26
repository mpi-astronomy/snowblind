import numpy as np
import pytest
from stcal.ramp_fitting.likely_algo_classes import Covar
from stcal.ramp_fitting.likely_fit import get_readtimes, likely_ramp_fit
from stcal.ramp_fitting.ramp_fit_class import RampData
from stdatamodels.jwst.datamodels import dqflags

from snowblind.likely_fit_reset import fit_ramps_reset, likely_ramp_fit_reset, likely_ramp_fit_reset_blocks

DNU = dqflags.pixel["DO_NOT_USE"]
SAT = dqflags.pixel["SATURATED"]
JUMP = dqflags.pixel["JUMP_DET"]

FRAME_TIME = 10.73677
READNOISE_CDS = 7.0
GAIN = 2.0
SIGMA_B = 15.0


def blank_ramp_data(ngroups, ncols=1, nframes=1, groupgap=0, zeroframe=False):
    """RampData of one integration and one row, with uniform read noise and gain."""
    data = np.zeros((1, ngroups, 1, ncols), dtype=np.float32)
    gdq = np.zeros((1, ngroups, 1, ncols), dtype=np.uint8)
    pdq = np.zeros((1, ncols), dtype=np.uint32)
    dark = np.zeros((1, ncols), dtype=np.float32)
    ramp_data = RampData()
    zf = np.zeros((1, 1, ncols), dtype=np.float32) if zeroframe else None
    ramp_data.set_arrays(data=data, groupdq=gdq, pixeldq=pdq, average_dark_current=dark, zeroframe=zf)
    ramp_data.set_meta(
        name="NIRCAM",
        frame_time=FRAME_TIME,
        group_time=(nframes + groupgap) * FRAME_TIME,
        groupgap=groupgap,
        nframes=nframes,
        drop_frames1=None,
    )
    ramp_data.set_dqflags(dqflags.pixel)
    readnoise = np.full((1, ncols), READNOISE_CDS, dtype=np.float32)
    gain = np.full((1, ncols), GAIN, dtype=np.float32)
    return ramp_data, readnoise, gain


def group_times(ramp_data):
    """Mean time of each group since reset, with the stcal convention of the first read at one frame time."""
    return np.array([np.mean(t) for t in get_readtimes(ramp_data)])


def linear_ramp(ramp_data, rate, reset=0.0):
    """Fill the data with reset + rate * t for every pixel."""
    t = group_times(ramp_data)
    ramp_data.data[0, :, 0, :] = (reset + rate * t)[:, None]


def test_all_good_ramp_matches_stcal_closely():
    """likely_ramp_fit_reset on a noiseless 10-group ramp of 20 DN/s returns the stcal rate within 1e-6 relative."""
    ramp_data, readnoise, gain = blank_ramp_data(10)
    linear_ramp(ramp_data, 20.0)

    image_info, integ_info, _ = likely_ramp_fit_reset(ramp_data, readnoise, gain, SIGMA_B)
    stcal_info, _, _ = likely_ramp_fit(ramp_data, readnoise, gain)

    assert image_info["slope"][0, 0] == pytest.approx(stcal_info["slope"][0, 0], rel=1e-6)
    assert integ_info["used_reset"][0, 0, 0]


def test_all_good_ramp_variance_shift_matches_weights():
    """likely_ramp_fit_reset reports a read-noise variance equal to 1 / sum(C^-1) of the full tridiagonal covariance."""
    ramp_data, readnoise, gain = blank_ramp_data(6)
    linear_ramp(ramp_data, 0.0)
    readtimes = [np.array([0.0])] + [np.atleast_1d(t) for t in get_readtimes(ramp_data)]
    covar = Covar(readtimes)
    rn = READNOISE_CDS / np.sqrt(2.0)
    alpha = rn**2 * covar.alpha_readnoise
    alpha[0] = (SIGMA_B**2 + rn**2) / covar.delta_t[0] ** 2
    beta = rn**2 * covar.beta_readnoise
    matrix = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
    expected = 1.0 / np.sum(np.linalg.inv(matrix))

    _image_info, integ_info, _ = likely_ramp_fit_reset(ramp_data, readnoise, gain, np.full((1, 1), SIGMA_B))

    assert integ_info["var_rnoise"][0, 0, 0] == pytest.approx(expected, rel=1e-6)


def test_fit_ramps_reset_equals_generalized_least_squares():
    """fit_ramps_reset with a fixed count-rate guess equals the explicit GLS solution with the same covariance."""
    ramp_data, _readnoise, _gain = blank_ramp_data(8)
    readtimes = [np.array([0.0])] + [np.atleast_1d(t) for t in get_readtimes(ramp_data)]
    covar = Covar(readtimes)
    rng = np.random.default_rng(1)
    t = np.concatenate([[0.0], group_times(ramp_data)])
    data = 12.0 * t + rng.normal(scale=3.0, size=t.size)
    diffs = (np.diff(data) / covar.delta_t)[:, None]
    rn = READNOISE_CDS / np.sqrt(2.0)
    guess = np.array([12.0])

    result = fit_ramps_reset(
        diffs, covar, np.array([GAIN]), np.array([rn]), sigma_b=np.array([SIGMA_B]), count_rate_guess=guess
    )

    alpha = 12.0 / GAIN * covar.alpha_phnoise + rn**2 * covar.alpha_readnoise
    alpha[0] = 12.0 / GAIN * covar.alpha_phnoise[0] + (SIGMA_B**2 + rn**2) / covar.delta_t[0] ** 2
    beta = 12.0 / GAIN * covar.beta_phnoise + rn**2 * covar.beta_readnoise
    cinv = np.linalg.inv(np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1))
    ones = np.ones(alpha.size)
    expected = ones @ cinv @ diffs[:, 0] / (ones @ cinv @ ones)
    assert result.countrate[0] == pytest.approx(expected, rel=1e-8)
    assert result.uncert[0] == pytest.approx(np.sqrt(1.0 / (ones @ cinv @ ones)), rel=1e-6)


def test_only_first_group_good_gives_rate_from_reset():
    """likely_ramp_fit_reset with groups 2 onward saturated returns rate R1 / t1 and the predicted variance."""
    ramp_data, readnoise, gain = blank_ramp_data(5)
    linear_ramp(ramp_data, 30.0)
    ramp_data.groupdq[0, 1:, 0, 0] = SAT
    t1 = group_times(ramp_data)[0]
    r1 = ramp_data.data[0, 0, 0, 0]
    rn = READNOISE_CDS / np.sqrt(2.0)

    image_info, integ_info, _ = likely_ramp_fit_reset(ramp_data, readnoise, gain, np.full((1, 1), SIGMA_B))

    assert image_info["slope"][0, 0] == pytest.approx(r1 / t1, rel=1e-6)
    expected_var = (SIGMA_B**2 + rn**2 + (r1 / t1) * t1 / GAIN) / t1**2
    assert image_info["var_rnoise"][0, 0] + image_info["var_poisson"][0, 0] == pytest.approx(expected_var, rel=1e-4)
    assert integ_info["used_reset"][0, 0, 0]


def test_only_zeroframe_good_gives_rate_from_reset():
    """likely_ramp_fit_reset with every group saturated but a valid zeroframe returns zeroframe / frame_time."""
    ramp_data, readnoise, gain = blank_ramp_data(5, nframes=4, zeroframe=True)
    linear_ramp(ramp_data, 30.0)
    ramp_data.groupdq[0, :, 0, 0] = SAT
    ramp_data.zeroframe[0, 0, 0] = 30.0 * FRAME_TIME + 2.0
    rn = READNOISE_CDS / np.sqrt(2.0)

    image_info, integ_info, _ = likely_ramp_fit_reset(ramp_data, readnoise, gain, np.full((1, 1), SIGMA_B))

    rate = (30.0 * FRAME_TIME + 2.0) / FRAME_TIME
    assert image_info["slope"][0, 0] == pytest.approx(rate, rel=1e-6)
    expected_var = (SIGMA_B**2 + rn**2 + rate * FRAME_TIME / GAIN) / FRAME_TIME**2
    assert image_info["var_rnoise"][0, 0] + image_info["var_poisson"][0, 0] == pytest.approx(expected_var, rel=1e-4)
    assert integ_info["used_reset"][0, 0, 0]


def test_zero_valued_zeroframe_is_unusable():
    """likely_ramp_fit_reset with every group saturated and a zeroframe of exactly 0 flags the pixel DO_NOT_USE."""
    ramp_data, readnoise, gain = blank_ramp_data(5, nframes=4, zeroframe=True)
    linear_ramp(ramp_data, 30.0)
    ramp_data.groupdq[0, :, 0, 0] = SAT
    ramp_data.zeroframe[0, 0, 0] = 0.0

    image_info, _integ_info, _ = likely_ramp_fit_reset(ramp_data, readnoise, gain, np.full((1, 1), SIGMA_B))

    assert image_info["dq"][0, 0] & DNU


def test_reset_far_off_is_dropped():
    """likely_ramp_fit_reset on a ramp whose reset is 10 sigma_b off drops d_0 and returns the stcal rate."""
    ramp_data, readnoise, gain = blank_ramp_data(10)
    linear_ramp(ramp_data, 20.0, reset=10.0 * SIGMA_B)

    image_info, integ_info, _ = likely_ramp_fit_reset(ramp_data, readnoise, gain, SIGMA_B)
    stcal_info, _, _ = likely_ramp_fit(ramp_data, readnoise, gain)

    assert not integ_info["used_reset"][0, 0, 0]
    assert image_info["slope"][0, 0] == pytest.approx(stcal_info["slope"][0, 0], rel=1e-9)
    assert not image_info["dq"][0, 0] & JUMP


def test_dropped_reset_is_not_a_jump_but_real_jump_is():
    """likely_ramp_fit_reset flags a 500 DN jump at group 5 as JUMP_DET and keeps d_0 when the reset is on."""
    ramp_data, readnoise, gain = blank_ramp_data(10)
    linear_ramp(ramp_data, 20.0)
    ramp_data.data[0, 5:, 0, 0] += 500.0

    image_info, integ_info, _ = likely_ramp_fit_reset(ramp_data, readnoise, gain, SIGMA_B)

    assert image_info["dq"][0, 0] & JUMP
    assert integ_info["used_reset"][0, 0, 0]
    assert image_info["slope"][0, 0] == pytest.approx(20.0, rel=1e-6)


def test_scalar_and_uniform_per_pixel_sigma_b_agree():
    """likely_ramp_fit_reset gives the same rate and variance for scalar sigma_b and a uniform sigma_b image."""
    ramp_data, readnoise, gain = blank_ramp_data(6, ncols=3)
    linear_ramp(ramp_data, 15.0)
    ramp_data.data[0, 0, 0, :] += np.array([5.0, -3.0, 8.0], dtype=np.float32)

    scalar_info, _, _ = likely_ramp_fit_reset(ramp_data, readnoise, gain, SIGMA_B)
    pixel_info, _, _ = likely_ramp_fit_reset(ramp_data, readnoise, gain, np.full((1, 3), SIGMA_B))

    np.testing.assert_allclose(scalar_info["slope"], pixel_info["slope"], rtol=1e-6)
    np.testing.assert_allclose(scalar_info["var_rnoise"], pixel_info["var_rnoise"], rtol=1e-6)


def test_per_pixel_sigma_b_changes_variance_pixel_by_pixel():
    """likely_ramp_fit_reset with sigma_b 10 and 30 in two one-group pixels reports the matching variances."""
    ramp_data, readnoise, gain = blank_ramp_data(5, ncols=2)
    linear_ramp(ramp_data, 30.0)
    ramp_data.groupdq[0, 1:, 0, :] = SAT
    t1 = group_times(ramp_data)[0]
    rn = READNOISE_CDS / np.sqrt(2.0)
    sigma_b = np.array([[10.0, 30.0]])

    image_info, _, _ = likely_ramp_fit_reset(ramp_data, readnoise, gain, sigma_b)

    expected = (sigma_b[0] ** 2 + rn**2) / t1**2
    np.testing.assert_allclose(image_info["var_rnoise"][0], expected, rtol=1e-5)


def test_t1_scale_shifts_one_group_rate():
    """likely_ramp_fit_reset with t1_scale 1.1 on a one-good-group pixel returns R1 / (1.1 t1)."""
    ramp_data, readnoise, gain = blank_ramp_data(5)
    linear_ramp(ramp_data, 30.0)
    ramp_data.groupdq[0, 1:, 0, 0] = SAT
    t1 = group_times(ramp_data)[0]
    r1 = ramp_data.data[0, 0, 0, 0]

    image_info, _, _ = likely_ramp_fit_reset(ramp_data, readnoise, gain, SIGMA_B, t1_scale=1.1)

    assert image_info["slope"][0, 0] == pytest.approx(r1 / (1.1 * t1), rel=1e-6)


def test_d0_resid_has_unit_scatter():
    """likely_ramp_fit_reset on 3000 noisy pixels with reset scatter sigma_b gives d0_resid with std between 0.85 and 1.1."""
    rng = np.random.default_rng(2)
    ncols = 3000
    ramp_data, readnoise, gain = blank_ramp_data(8, ncols=ncols)
    t = group_times(ramp_data)
    reset = rng.normal(scale=SIGMA_B, size=ncols)
    noise = rng.normal(scale=READNOISE_CDS / np.sqrt(2.0), size=(t.size, ncols))
    ramp_data.data[0, :, 0, :] = reset[None, :] + 5.0 * t[:, None] + noise

    _image_info, integ_info, _ = likely_ramp_fit_reset(ramp_data, readnoise, gain, SIGMA_B, use_zeroframe=False)

    resid = integ_info["d0_resid"][0, 0]
    assert abs(np.nanmean(resid)) < 0.1
    assert 0.85 < np.nanstd(resid) < 1.1


def test_too_few_groups_raises():
    """likely_ramp_fit_reset with two groups raises ValueError."""
    ramp_data, readnoise, gain = blank_ramp_data(2)

    with pytest.raises(ValueError, match="at least 3 groups"):
        likely_ramp_fit_reset(ramp_data, readnoise, gain, SIGMA_B)


def test_three_groups_are_enough():
    """likely_ramp_fit_reset fits a three-group ramp, which stcal alone would refuse."""
    ramp_data, readnoise, gain = blank_ramp_data(3)
    linear_ramp(ramp_data, 20.0)

    image_info, _, _ = likely_ramp_fit_reset(ramp_data, readnoise, gain, SIGMA_B)

    assert image_info["slope"][0, 0] == pytest.approx(20.0, rel=1e-6)


def test_two_differences_are_both_kept_even_when_they_disagree():
    """likely_ramp_fit_reset with two usable differences 20 sigma_b apart keeps both and returns a rate between them."""
    ramp_data, readnoise, gain = blank_ramp_data(5)
    linear_ramp(ramp_data, 100.0, reset=-20.0 * SIGMA_B)
    ramp_data.groupdq[0, 2:, 0, 0] = SAT
    t = group_times(ramp_data)
    d0 = ramp_data.data[0, 0, 0, 0] / t[0]
    d1 = (ramp_data.data[0, 1, 0, 0] - ramp_data.data[0, 0, 0, 0]) / (t[1] - t[0])

    image_info, integ_info, _ = likely_ramp_fit_reset(ramp_data, readnoise, gain, SIGMA_B)

    assert integ_info["used_reset"][0, 0, 0]
    assert min(d0, d1) < image_info["slope"][0, 0] < max(d0, d1)
    assert not image_info["dq"][0, 0] & JUMP


def test_reset_level_shifts_one_group_rate():
    """likely_ramp_fit_reset with reset_level -4 on a one-good-group pixel returns (R1 + 4) / t1."""
    ramp_data, readnoise, gain = blank_ramp_data(5)
    linear_ramp(ramp_data, 30.0)
    ramp_data.groupdq[0, 1:, 0, 0] = SAT
    t1 = group_times(ramp_data)[0]
    r1 = ramp_data.data[0, 0, 0, 0]

    image_info, _, _ = likely_ramp_fit_reset(ramp_data, readnoise, gain, SIGMA_B, reset_level=-4.0)

    assert image_info["slope"][0, 0] == pytest.approx((r1 + 4.0) / t1, rel=1e-6)


def test_per_pixel_reset_level_map():
    """likely_ramp_fit_reset with a reset_level map of [0, 10] on two one-good-group pixels returns (R1 - map) / t1."""
    ramp_data, readnoise, gain = blank_ramp_data(5, ncols=2)
    linear_ramp(ramp_data, 30.0)
    ramp_data.groupdq[0, 1:, 0, :] = SAT
    t1 = group_times(ramp_data)[0]
    level = np.array([[0.0, 10.0]], dtype=np.float32)

    image_info, _, _ = likely_ramp_fit_reset(ramp_data, readnoise, gain, SIGMA_B, reset_level=level)

    expected = (ramp_data.data[0, 0, 0, :] - level[0]) / t1
    np.testing.assert_allclose(image_info["slope"][0], expected, rtol=1e-6)


def test_t1_scale_leaves_later_differences_alone():
    """likely_ramp_fit_reset with t1_scale 0.9 on a noiseless ramp whose reset came 0.1 t1 late returns the exact rate."""
    ramp_data, readnoise, gain = blank_ramp_data(8)
    t1 = group_times(ramp_data)[0]
    linear_ramp(ramp_data, 20.0, reset=-0.1 * 20.0 * t1)

    image_info, integ_info, _ = likely_ramp_fit_reset(ramp_data, readnoise, gain, SIGMA_B, t1_scale=0.9)

    assert image_info["slope"][0, 0] == pytest.approx(20.0, rel=1e-6)
    assert integ_info["used_reset"][0, 0, 0]
    assert abs(integ_info["d0_resid"][0, 0, 0]) < 1e-4


def test_blocks_driver_with_equal_timing_matches_single_fit():
    """likely_ramp_fit_reset_blocks with the same t1_scale in two blocks equals likely_ramp_fit_reset on the whole row."""
    ramp_data, readnoise, gain = blank_ramp_data(6, ncols=4)
    linear_ramp(ramp_data, 15.0, reset=-5.0)
    blocks = [(slice(0, 1), slice(0, 2)), (slice(0, 1), slice(2, 4))]

    whole, _, _ = likely_ramp_fit_reset(ramp_data, readnoise, gain, SIGMA_B, t1_scale=0.9, reset_level=-3.0)
    split, split_integ, _ = likely_ramp_fit_reset_blocks(
        ramp_data, readnoise, gain, SIGMA_B, blocks, [0.9, 0.9], [-3.0, -3.0]
    )

    np.testing.assert_allclose(split["slope"], whole["slope"], rtol=1e-6)
    np.testing.assert_allclose(split["var_rnoise"], whole["var_rnoise"], rtol=1e-6)
    assert split_integ["used_reset"].shape == (1, 1, 4)


def test_blocks_driver_applies_each_blocks_timing():
    """likely_ramp_fit_reset_blocks with t1_scale 0.8 and 1.0 gives one-group pixels R1 / (0.8 t1) and R1 / t1."""
    ramp_data, readnoise, gain = blank_ramp_data(5, ncols=2)
    linear_ramp(ramp_data, 30.0)
    ramp_data.groupdq[0, 1:, 0, :] = SAT
    t1 = group_times(ramp_data)[0]
    r1 = ramp_data.data[0, 0, 0, :]
    blocks = [(slice(0, 1), slice(0, 1)), (slice(0, 1), slice(1, 2))]

    image_info, _, _ = likely_ramp_fit_reset_blocks(ramp_data, readnoise, gain, SIGMA_B, blocks, [0.8, 1.0], [0.0, 0.0])

    np.testing.assert_allclose(image_info["slope"][0], [r1[0] / (0.8 * t1), r1[1] / t1], rtol=1e-6)

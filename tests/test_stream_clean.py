import numpy as np
import pytest
from jwst.clean_flicker_noise.clean_flicker_noise import median_clean
from stdatamodels.jwst import datamodels

from snowblind import StreamCleanStep, readout
from snowblind.stream_clean import (
    clean_difference,
    clean_ramp,
    dilate_sources,
    estimate_noise,
    column_offset_pattern,
    even_odd_pattern,
    masked_mean_background,
    exclude_reference_border,
    first_group_time_ratio,
    integration_backgrounds,
    prepare_difference,
    slope_per_group,
)


JUMP_DET = datamodels.dqflags.group["JUMP_DET"]
SATURATED = datamodels.dqflags.group["SATURATED"]
DO_NOT_USE = datamodels.dqflags.group["DO_NOT_USE"]

NY, NX = 64, 64


def make_ramp_model(nints=1, ngroups=4, fastaxis=-1, slowaxis=2):
    """Small full-frame-like NIRCam ramp with zero data and good DQ."""
    model = datamodels.RampModel((nints, ngroups, NY, NX))
    model.meta.instrument.name = "NIRCAM"
    model.meta.instrument.detector = "NRCA1"
    model.meta.exposure.type = "NRC_IMAGE"
    model.meta.exposure.nframes = 1
    model.meta.exposure.groupgap = 0
    model.meta.exposure.frame_time = 10.73677
    model.meta.exposure.group_time = 10.73677
    model.meta.subarray.name = "FULL"
    model.meta.subarray.fastaxis = fastaxis
    model.meta.subarray.slowaxis = slowaxis
    model.meta.filename = "jw00000000000_00000_00000_nrca1_jump.fits"
    return model


def row_stripes(rng, fastaxis=-1, slowaxis=2, scale=5.0):
    """Science-frame image that is constant along each row of each channel in the detector frame.

    Stripe values are uniform in [-scale, scale] so that no row is an outlier
    to the 3-sigma clipping inside the jwst background helper.
    """
    width = readout.channel_width(NX)
    det = np.zeros((NY, NX), dtype=np.float32)
    for columns in readout.channel_slices(NX):
        det[:, columns] = rng.uniform(-scale, scale, size=(NY, 1)).astype(np.float32)
    return np.ascontiguousarray(readout.to_science_frame(det, fastaxis, slowaxis)), width


def test_init():
    """StreamCleanStep() has the class alias stream_clean."""
    step = StreamCleanStep()

    assert step.class_alias == "stream_clean"


def test_row_median_matches_median_clean():
    """estimate_noise with row_median on a 64 x 2048 image equals image minus median_clean(fit_by_channel=True)."""
    rng = np.random.default_rng(1)
    image = rng.normal(size=(64, 2048)).astype(np.float32)
    mask = rng.random((64, 2048)) < 0.7

    noise = estimate_noise(image, mask, -1, 2, "row_median")

    expected = image - median_clean(image, mask, 2, fit_by_channel=True)
    np.testing.assert_allclose(noise, expected, atol=1e-5)


def test_row_median_matches_median_clean_flipped_geometry():
    """estimate_noise with row_median for fastaxis 1, slowaxis -2 equals image minus median_clean along axis 2."""
    rng = np.random.default_rng(2)
    image = rng.normal(size=(64, 2048)).astype(np.float32)
    mask = rng.random((64, 2048)) < 0.7

    noise = estimate_noise(image, mask, 1, -2, "row_median")

    expected = image - median_clean(image, mask, 2, fit_by_channel=True)
    np.testing.assert_allclose(noise, expected, atol=1e-5)


def test_row_median_recovers_stripes_exactly():
    """estimate_noise with row_median on pure per-channel row stripes returns the stripes."""
    rng = np.random.default_rng(3)
    stripes, _ = row_stripes(rng)
    mask = np.ones((NY, NX), dtype=bool)

    noise = estimate_noise(stripes, mask, -1, 2, "row_median")

    np.testing.assert_allclose(noise, stripes, atol=1e-6)


def test_row_median_zero_where_row_fully_masked():
    """estimate_noise with row_median returns 0 in a channel row whose pixels are all masked."""
    rng = np.random.default_rng(4)
    stripes, width = row_stripes(rng)
    mask = np.ones((NY, NX), dtype=bool)
    mask[10, :width] = False

    noise = estimate_noise(stripes, mask, -1, 2, "row_median")

    np.testing.assert_array_equal(noise[10, :width], 0.0)
    np.testing.assert_allclose(noise[10, width:], stripes[10, width:], atol=1e-6)


def test_estimate_noise_unknown_method():
    """estimate_noise with an unknown fit_method raises ValueError."""
    image = np.zeros((NY, NX), dtype=np.float32)
    mask = np.ones((NY, NX), dtype=bool)

    with pytest.raises(ValueError, match="Unknown fit_method"):
        estimate_noise(image, mask, -1, 2, "no_such_method")


def test_clean_difference_restores_background():
    """clean_difference with background_method='median' on stripes plus a constant returns a flat image at the background."""
    rng = np.random.default_rng(5)
    stripes, _ = row_stripes(rng)
    image = stripes + 100.0
    mask = np.ones((NY, NX), dtype=bool)

    cleaned, background = clean_difference(image, mask, -1, 2, fit_method="row_median")

    np.testing.assert_allclose(cleaned, background, atol=1e-4)
    assert background == pytest.approx(100.0, abs=1.0)


def test_clean_difference_keeps_nan():
    """clean_difference leaves NaN pixels NaN and excludes them from the mask."""
    rng = np.random.default_rng(6)
    stripes, _ = row_stripes(rng)
    stripes[3, 7] = np.nan
    mask = np.ones((NY, NX), dtype=bool)

    cleaned, _ = clean_difference(stripes, mask, -1, 2, fit_method="row_median", clip_sigma=0)

    assert np.isnan(cleaned[3, 7])
    assert not mask[3, 7]


def test_clean_difference_none_when_all_masked():
    """clean_difference with an all-False mask returns (None, None)."""
    image = np.ones((NY, NX), dtype=np.float32)
    mask = np.zeros((NY, NX), dtype=bool)

    cleaned, background = clean_difference(image, mask, -1, 2, fit_method="row_median")

    assert cleaned is None
    assert background is None


def striped_ramp(rng, ngroups=4, rate=10.0):
    """Ramp whose group differences are a constant rate plus independent row stripes; returns data and the stripes."""
    data = np.zeros((1, ngroups, NY, NX), dtype=np.float32)
    stripes = np.zeros((ngroups, NY, NX), dtype=np.float32)
    for j in range(ngroups):
        stripes[j], _ = row_stripes(rng)
        data[0, j] = (j + 1) * rate + stripes[: j + 1].sum(axis=0)
    return data, stripes


def test_clean_ramp_removes_stripes_from_differences():
    """clean_ramp with row_median leaves every group difference flat, at the rate to within the stripe median."""
    rng = np.random.default_rng(7)
    data, _ = striped_ramp(rng)
    groupdq = np.zeros(data.shape, dtype=np.uint8)
    mask = np.ones((NY, NX), dtype=bool)

    cleaned = clean_ramp(data, groupdq, mask, -1, 2, fit_method="row_median", clip_sigma=0)

    diffs = np.diff(cleaned[0], axis=0)
    assert np.all(np.ptp(diffs, axis=(1, 2)) < 1e-3)
    np.testing.assert_allclose(diffs.mean(axis=(1, 2)), 10.0, atol=1.0)


def test_clean_ramp_leaves_group0_when_clean_first_group_false():
    """clean_ramp with clean_first_group=False returns group 0 unchanged."""
    rng = np.random.default_rng(8)
    data, _ = striped_ramp(rng)
    groupdq = np.zeros(data.shape, dtype=np.uint8)
    mask = np.ones((NY, NX), dtype=bool)

    cleaned = clean_ramp(data, groupdq, mask, -1, 2, fit_method="row_median", clip_sigma=0, clean_first_group=False)

    np.testing.assert_array_equal(cleaned[0, 0], data[0, 0])


def test_clean_ramp_cleans_group0_when_clean_first_group_true():
    """clean_ramp with clean_first_group=True changes group 0 by exactly its row-median noise estimate."""
    rng = np.random.default_rng(9)
    data, _ = striped_ramp(rng)
    groupdq = np.zeros(data.shape, dtype=np.uint8)
    mask = np.ones((NY, NX), dtype=bool)

    cleaned = clean_ramp(data, groupdq, mask, -1, 2, fit_method="row_median", clip_sigma=0, clean_first_group=True)

    noise = estimate_noise(data[0, 0] - np.median(data[0, 0]), mask.copy(), -1, 2, "row_median")
    np.testing.assert_allclose(data[0, 0] - cleaned[0, 0], noise, atol=1e-3)


def test_clean_ramp_excludes_flagged_pixels_from_fit():
    """clean_ramp ignores a JUMP_DET pixel with a huge value, so its row is cleaned to the same level as other rows."""
    rng = np.random.default_rng(10)
    data, _ = striped_ramp(rng)
    data[0, 2:, 5, 5] += 1e6
    groupdq = np.zeros(data.shape, dtype=np.uint8)
    groupdq[0, 2, 5, 5] = JUMP_DET
    mask = np.ones((NY, NX), dtype=bool)

    cleaned = clean_ramp(data, groupdq, mask, -1, 2, fit_method="row_median", clip_sigma=0)

    diff = cleaned[0, 2] - cleaned[0, 1]
    np.testing.assert_allclose(diff[5, :5], diff[6, :5], atol=1e-3)


def test_clean_ramp_excludes_saturated_pixels_from_fit():
    """clean_ramp ignores a SATURATED pixel with a huge value, so its row is cleaned to the same level as other rows."""
    rng = np.random.default_rng(11)
    data, _ = striped_ramp(rng)
    data[0, 3, 9, 9] += 1e6
    groupdq = np.zeros(data.shape, dtype=np.uint8)
    groupdq[0, 3, 9, 9] = SATURATED
    mask = np.ones((NY, NX), dtype=bool)

    cleaned = clean_ramp(data, groupdq, mask, -1, 2, fit_method="row_median", clip_sigma=0)

    diff = cleaned[0, 3] - cleaned[0, 2]
    np.testing.assert_allclose(diff[9, :9], diff[8, :9], atol=1e-3)


def write_mask(tmp_path, mask):
    """Save a background mask as an ImageModel and return its path."""
    mask_model = datamodels.ImageModel(mask.astype(np.float32))
    path = tmp_path / "mask.fits"
    mask_model.save(path)
    return str(path)


def test_step_call_with_user_mask_cleans_and_completes(tmp_path):
    """StreamCleanStep.call with a user mask and row_median leaves flat interior group differences and sets COMPLETE."""
    rng = np.random.default_rng(12)
    data, _ = striped_ramp(rng)
    model = make_ramp_model()
    model.data = data
    mask_path = write_mask(tmp_path, np.ones((NY, NX), dtype=bool))

    result = StreamCleanStep.call(
        model, fit_method="row_median", user_mask=mask_path, background_method="median", clip_sigma=0
    )

    assert result.meta.cal_step.stream_clean == "COMPLETE"
    diffs = np.diff(result.data[0], axis=0)[:, 4:-4, 4:-4]
    assert np.all(np.ptp(diffs, axis=(1, 2)) < 1e-3)
    np.testing.assert_allclose(diffs.mean(axis=(1, 2)), 10.0, atol=1.0)


def test_step_skips_subarray():
    """StreamCleanStep.call on a subarray sets SKIPPED and leaves the data alone."""
    model = make_ramp_model()
    model.meta.subarray.name = "SUB64P"
    model.data[:] = 3.0

    result = StreamCleanStep.call(model, fit_method="row_median")

    assert result.meta.cal_step.stream_clean == "SKIPPED"
    np.testing.assert_array_equal(result.data, 3.0)


def test_step_skips_miri():
    """StreamCleanStep.call on MIRI data sets SKIPPED."""
    model = make_ramp_model()
    model.meta.instrument.name = "MIRI"

    result = StreamCleanStep.call(model, fit_method="row_median")

    assert result.meta.cal_step.stream_clean == "SKIPPED"


def gradient_ramp(rng, ngroups=8, rate=10.0):
    """Ramp whose differences are a smooth per-pixel gradient (same every group) plus independent 1 DN row stripes."""
    yy, xx = np.mgrid[0:NY, 0:NX]
    gradient = (rate + 0.5 * xx + 0.25 * yy).astype(np.float32)
    data = np.zeros((1, ngroups, NY, NX), dtype=np.float32)
    for j in range(ngroups):
        stripes, _ = row_stripes(rng, scale=1.0)
        data[0, j] = (data[0, j - 1] if j else 0.0) + gradient + stripes
    return data, gradient


# the background model's box median filter flattens the outer three 8-pixel boxes
INTERIOR = (slice(24, -24), slice(24, -24))


def test_integration_backgrounds_follow_the_gradient():
    """integration_backgrounds on a gradient-plus-stripes ramp returns one image within 0.5 DN of the gradient inside."""
    rng = np.random.default_rng(20)
    data, gradient = gradient_ramp(rng)
    groupdq = np.zeros(data.shape, dtype=np.uint8)
    mask = np.ones((NY, NX), dtype=bool)

    backgrounds = integration_backgrounds(data, groupdq, mask, background_box_size=(8, 8))

    assert len(backgrounds) == 1
    assert np.abs(backgrounds[0] - gradient)[INTERIOR].max() < 0.5


def test_integration_model_keeps_gradient_and_removes_stripes():
    """clean_ramp with background_method='integration_model' leaves interior differences equal to the gradient within 0.5 DN."""
    rng = np.random.default_rng(21)
    data, gradient = gradient_ramp(rng)
    groupdq = np.zeros(data.shape, dtype=np.uint8)
    mask = np.ones((NY, NX), dtype=bool)

    cleaned = clean_ramp(
        data, groupdq, mask, -1, 2, fit_method="row_median", background_method="integration_model",
        background_box_size=(8, 8), clip_sigma=0,
    )

    diffs = np.diff(cleaned[0], axis=0)
    assert np.abs(diffs - gradient)[(slice(None),) + INTERIOR].max() < 0.5


def test_integration_model_is_the_same_for_every_difference():
    """clean_ramp with integration_model removes the same background from each difference, so cleaned differences agree."""
    rng = np.random.default_rng(22)
    data, _gradient = gradient_ramp(rng)
    groupdq = np.zeros(data.shape, dtype=np.uint8)
    mask = np.ones((NY, NX), dtype=bool)

    cleaned = clean_ramp(
        data, groupdq, mask, -1, 2, fit_method="row_median", background_method="integration_model",
        background_box_size=(8, 8), clip_sigma=0,
    )

    diffs = np.diff(cleaned[0], axis=0)
    np.testing.assert_allclose(diffs[0], diffs[1], atol=1e-3)


def test_first_group_time_ratio_single_frame():
    """first_group_time_ratio is 1 for one frame per group and no group gap."""
    model = make_ramp_model()
    model.meta.exposure.nframes = 1
    model.meta.exposure.groupgap = 0
    model.meta.exposure.frame_time = 10.0
    model.meta.exposure.group_time = 10.0

    assert first_group_time_ratio(model) == 1.0


def test_first_group_time_ratio_frame_averaged():
    """first_group_time_ratio for 8 frames and a gap of 2 is 4.5 / 10."""
    model = make_ramp_model()
    model.meta.exposure.nframes = 8
    model.meta.exposure.groupgap = 2
    model.meta.exposure.frame_time = 10.0
    model.meta.exposure.group_time = 100.0

    assert first_group_time_ratio(model) == pytest.approx(0.45)


def test_clip_sigma_removes_outlier_from_mask():
    """clean_difference with clip_sigma=3 on stripes with one 1000 DN pixel masks that pixel."""
    rng = np.random.default_rng(30)
    stripes, _ = row_stripes(rng)
    stripes += rng.normal(scale=0.5, size=stripes.shape).astype(np.float32)
    stripes[7, 7] = 1000.0
    mask = np.ones((NY, NX), dtype=bool)

    clean_difference(stripes, mask, -1, 2, fit_method="row_median", clip_sigma=3.0)

    assert not mask[7, 7]


def test_wiener_channel_without_data_removes_nothing():
    """clean_ramp with wiener and a mask that blanks channel 0 leaves channel 0 unchanged instead of raising."""
    rng = np.random.default_rng(31)
    data, _ = striped_ramp(rng)
    groupdq = np.zeros(data.shape, dtype=np.uint8)
    mask = np.ones((NY, NX), dtype=bool)
    width = readout.channel_width(NX)
    mask_det = readout.to_detector_frame(mask, -1, 2)
    mask_det[:, :width] = False

    cleaned = clean_ramp(data, groupdq, mask, -1, 2, fit_method="wiener", background_method=None, max_lag=256)

    det_in = readout.to_detector_frame(data[0], -1, 2)
    det_out = readout.to_detector_frame(cleaned[0], -1, 2)
    np.testing.assert_allclose(det_out[:, :, :width], det_in[:, :, :width], atol=1e-4)


def test_prepare_difference_integration_model_needs_fixed_background():
    """prepare_difference with background_method='integration_model' and no fixed background raises ValueError."""
    image = np.ones((NY, NX), dtype=np.float32)
    mask = np.ones((NY, NX), dtype=bool)

    with pytest.raises(ValueError, match="integration_model"):
        prepare_difference(image, mask, background_method="integration_model")


def test_exposure_model_is_shared_across_integrations():
    """integration_backgrounds with per_exposure=True on two integrations returns the same model twice, close to the gradient."""
    rng = np.random.default_rng(40)
    data1, gradient = gradient_ramp(rng)
    data2, _ = gradient_ramp(rng)
    data = np.concatenate([data1, data2], axis=0)
    groupdq = np.zeros(data.shape, dtype=np.uint8)
    mask = np.ones((NY, NX), dtype=bool)

    backgrounds = integration_backgrounds(data, groupdq, mask, background_box_size=(8, 8), per_exposure=True)

    assert len(backgrounds) == 2
    np.testing.assert_array_equal(backgrounds[0], backgrounds[1])
    assert np.abs(backgrounds[0] - gradient)[INTERIOR].max() < 0.5


def test_exposure_model_cleans_both_integrations():
    """clean_ramp with background_method='exposure_model' leaves interior differences of both integrations at the gradient."""
    rng = np.random.default_rng(41)
    data1, gradient = gradient_ramp(rng)
    data2, _ = gradient_ramp(rng)
    data = np.concatenate([data1, data2], axis=0)
    groupdq = np.zeros(data.shape, dtype=np.uint8)
    mask = np.ones((NY, NX), dtype=bool)

    cleaned = clean_ramp(
        data, groupdq, mask, -1, 2, fit_method="row_median", background_method="exposure_model",
        background_box_size=(8, 8),
    )

    for i in range(2):
        diffs = np.diff(cleaned[i], axis=0)
        assert np.abs(diffs - gradient)[(slice(None),) + INTERIOR].max() < 0.5


def test_slope_per_group_is_the_least_squares_slope():
    """slope_per_group on a striped ramp equals numpy's per-pixel least-squares slope against the group index."""
    rng = np.random.default_rng(50)
    data, _ = striped_ramp(rng, ngroups=8, rate=3.0)
    groupdq = np.zeros(data.shape, dtype=np.uint8)

    slope = slope_per_group(data[0], groupdq[0])

    expected = np.polyfit(np.arange(8), data[0].reshape(8, -1).astype(np.float64), 1)[0].reshape(NY, NX)
    np.testing.assert_allclose(slope, expected, rtol=1e-5, atol=1e-4)


def test_slope_per_group_falls_back_where_flagged():
    """slope_per_group uses the mean of the usable differences for a pixel whose last groups are saturated."""
    data = np.zeros((1, 6, NY, NX), dtype=np.float32)
    for j in range(6):
        data[0, j] = 10.0 * j
    data[0, 3:, 2, 2] = 1e5
    groupdq = np.zeros(data.shape, dtype=np.uint8)
    groupdq[0, 3:, 2, 2] = SATURATED

    slope = slope_per_group(data[0], groupdq[0])

    assert slope[2, 2] == pytest.approx(10.0)
    assert slope[0, 0] == pytest.approx(10.0)


def test_exclude_reference_border_masks_four_pixels():
    """exclude_reference_border sets the outer four rows and columns False and leaves the interior as it was."""
    mask = np.ones((NY, NX), dtype=bool)
    mask[10, 10] = False

    out = exclude_reference_border(mask)

    assert not out[:4].any() and not out[-4:].any() and not out[:, :4].any() and not out[:, -4:].any()
    assert out[4:-4, 4:-4].sum() == (NY - 8) * (NX - 8) - 1
    assert mask[0, 0]


def test_dilate_sources_grows_masked_region():
    """dilate_sources with 2 pixels turns a single masked pixel into a diamond of 13 masked pixels."""
    mask = np.ones((NY, NX), dtype=bool)
    mask[20, 20] = False

    out = dilate_sources(mask, 2)

    assert (~out).sum() == 13
    assert not out[18, 20] and not out[20, 22] and out[18, 22]


def test_step_masks_reference_border_before_fitting(tmp_path):
    """StreamCleanStep.call with a user mask of all True still cleans the stripes without using the border rows."""
    rng = np.random.default_rng(60)
    data, _ = striped_ramp(rng)
    data[:, :, :4, :] += 1e4
    model = make_ramp_model()
    model.data = data
    mask_path = write_mask(tmp_path, np.ones((NY, NX), dtype=bool))

    result = StreamCleanStep.call(model, fit_method="row_median", user_mask=mask_path, background_method="median")

    diffs = np.diff(result.data[0], axis=0)[:, 4:-4, 4:-4]
    assert np.all(np.ptp(diffs, axis=(1, 2)) < 1e-3)


def test_step_second_mask_pass_runs_and_completes(tmp_path):
    """StreamCleanStep.call with mask_passes=2 and a user mask completes in one pass, since the user mask is fixed."""
    rng = np.random.default_rng(70)
    data, _ = striped_ramp(rng)
    model = make_ramp_model()
    model.data = data
    mask_path = write_mask(tmp_path, np.ones((NY, NX), dtype=bool))

    result = StreamCleanStep.call(
        model, fit_method="row_median", user_mask=mask_path, background_method="median", mask_passes=2
    )

    assert result.meta.cal_step.stream_clean == "COMPLETE"


def test_even_odd_pattern_recovers_alternating_offset():
    """even_odd_pattern on a channel with +0.5 on even and -0.5 on odd columns returns that pattern exactly."""
    image = np.zeros((16, 8), dtype=np.float32)
    image[:, ::2] = 0.5
    image[:, 1::2] = -0.5

    pattern = even_odd_pattern(image, np.ones(image.shape, dtype=bool))

    np.testing.assert_allclose(pattern, image)


def test_estimate_noise_even_odd_removes_alternation_per_channel():
    """estimate_noise with even_odd_columns=True removes a per-channel alternating column offset to 1e-6."""
    det = np.zeros((NY, NX), dtype=np.float32)
    for k, columns in enumerate(readout.channel_slices(NX)):
        det[:, columns.start::2] += 0.3 * (k + 1)
        det[:, columns.start + 1:columns.stop:2] -= 0.3 * (k + 1)
    image = np.ascontiguousarray(readout.to_science_frame(det, -1, 2))
    mask = np.ones((NY, NX), dtype=bool)

    noise = estimate_noise(image, mask, -1, 2, "row_median", even_odd_columns=True)

    np.testing.assert_allclose(image - noise, 0.0, atol=1e-6)


def test_estimate_noise_even_odd_off_leaves_alternation():
    """estimate_noise with even_odd_columns=False and row_median leaves an alternating column offset in place."""
    det = np.zeros((NY, NX), dtype=np.float32)
    det[:, ::2] = 0.5
    det[:, 1::2] = -0.5
    image = np.ascontiguousarray(readout.to_science_frame(det, -1, 2))
    mask = np.ones((NY, NX), dtype=bool)

    noise = estimate_noise(image, mask, -1, 2, "row_median", even_odd_columns=False)

    assert np.abs(image - noise).max() > 0.4


def test_column_offset_pattern_recovers_per_column_offsets():
    """column_offset_pattern on a channel whose columns hold 0, 1, ..., 7 returns that value down every column."""
    image = np.tile(np.arange(8, dtype=np.float32), (32, 1))

    pattern = column_offset_pattern(image, np.ones(image.shape, dtype=bool))

    np.testing.assert_allclose(pattern, image)


def test_column_offset_pattern_uses_only_masked_pixels():
    """column_offset_pattern ignores masked-out pixels: a column with value 1 and masked 100s returns 1."""
    image = np.ones((32, 4), dtype=np.float32)
    mask = np.ones(image.shape, dtype=bool)
    image[:8, 2] = 100.0
    mask[:8, 2] = False

    pattern = column_offset_pattern(image, mask)

    np.testing.assert_allclose(pattern, 1.0)


def test_column_offset_pattern_zero_for_sparse_column():
    """column_offset_pattern gives zero for a column with fewer than min_pixels usable pixels."""
    image = np.full((32, 4), 3.0, dtype=np.float32)
    mask = np.ones(image.shape, dtype=bool)
    mask[:-10, 1] = False

    pattern = column_offset_pattern(image, mask, min_pixels=16)

    np.testing.assert_allclose(pattern[:, 1], 0.0)
    np.testing.assert_allclose(pattern[:, 0], 3.0)


def test_estimate_noise_column_offsets_removes_per_column_offsets():
    """estimate_noise with column_offsets=True removes a different constant offset from every column to 1e-5."""
    rng = np.random.default_rng(3)
    det = np.tile(rng.normal(0.0, 1.0, NX).astype(np.float32), (NY, 1))
    image = np.ascontiguousarray(readout.to_science_frame(det, -1, 2))
    mask = np.ones((NY, NX), dtype=bool)

    noise = estimate_noise(image, mask, -1, 2, "row_median", column_offsets=True)

    np.testing.assert_allclose(image - noise, 0.0, atol=1e-5)


def test_estimate_noise_column_offsets_off_leaves_per_column_offsets():
    """estimate_noise with column_offsets=False and row_median leaves per-column offsets of unit scatter in place."""
    rng = np.random.default_rng(3)
    det = np.tile(rng.normal(0.0, 1.0, NX).astype(np.float32), (NY, 1))
    image = np.ascontiguousarray(readout.to_science_frame(det, -1, 2))
    mask = np.ones((NY, NX), dtype=bool)

    noise = estimate_noise(image, mask, -1, 2, "row_median", column_offsets=False)

    assert np.abs(image - noise).max() > 0.5


def test_masked_mean_background_ignores_masked_pixels():
    """masked_mean_background on a flat image of 2.0 with masked pixels at 100 returns 2.0 everywhere within 1e-3."""
    rng = np.random.default_rng(5)
    image = np.full((128, 128), 2.0, dtype=np.float32)
    mask = rng.uniform(size=image.shape) > 0.3
    image[~mask] = 100.0

    background = masked_mean_background(image, mask, (32, 32))

    np.testing.assert_allclose(background, 2.0, atol=1e-3)


def test_masked_mean_background_is_unbiased_for_skewed_values():
    """masked_mean_background on exponentially distributed values (mean 1) returns their mean to 2 percent, not the median."""
    rng = np.random.default_rng(6)
    image = rng.exponential(1.0, size=(256, 256)).astype(np.float32)
    mask = np.ones(image.shape, dtype=bool)

    background = masked_mean_background(image, mask, (32, 32))

    assert abs(background.mean() - image.mean()) < 0.02 * image.mean()


def test_estimate_noise_anchor_mean_matches_residual_mean_for_row_median():
    """estimate_noise with row_median and anchor_mean=True on a skewed image gives a noise estimate with the image mean."""
    rng = np.random.default_rng(7)
    image = rng.exponential(1.0, size=(NY, NX)).astype(np.float32)
    mask = np.ones((NY, NX), dtype=bool)

    noise = estimate_noise(image, mask, -1, 2, "row_median", anchor_mean=True)

    for columns in readout.channel_slices(NX):
        det_image = readout.to_detector_frame(image, -1, 2)[:, columns]
        det_noise = readout.to_detector_frame(noise, -1, 2)[:, columns]
        assert abs(det_noise.mean() - det_image.mean()) < 1e-4


def test_estimate_noise_row_median_without_anchor_is_below_mean_for_skewed_image():
    """estimate_noise with row_median and anchor_mean=False on an exponential image sits more than 0.2 below the image mean."""
    rng = np.random.default_rng(7)
    image = rng.exponential(1.0, size=(NY, NX)).astype(np.float32)
    mask = np.ones((NY, NX), dtype=bool)

    noise = estimate_noise(image, mask, -1, 2, "row_median", anchor_mean=False)

    assert image.mean() - noise.mean() > 0.2


def test_clean_ramp_first_group_against_reference_returns_reference_plus_cleaned_residual():
    """clean_ramp with first_group_reference and row_median on group 0 = reference + row pattern returns group 0 = reference."""
    rng = np.random.default_rng(13)
    reference = rng.normal(0.0, 10.0, (NY, NX)).astype(np.float32)
    rows = rng.normal(0.0, 3.0, NY).astype(np.float32)
    data = np.zeros((1, 3, NY, NX), dtype=np.float32)
    data[0, 0] = reference + rows[:, None]
    data[0, 1] = data[0, 0] + 1.0
    data[0, 2] = data[0, 1] + 1.0
    groupdq = np.zeros(data.shape, dtype=np.uint32)
    mask = np.ones((NY, NX), dtype=bool)

    cleaned = clean_ramp(
        data, groupdq, mask, -1, 2, fit_method="row_median", background_method=None, clean_first_group=True,
        first_group_reference=reference,
    )

    np.testing.assert_allclose(cleaned[0, 0], reference, atol=1e-4)


def test_clean_ramp_with_scene_background_keeps_the_scene_and_removes_rows():
    """clean_ramp with scene_background equal to the injected scene and row_median returns the scene ramp to 1e-4."""
    rng = np.random.default_rng(14)
    scene = rng.uniform(5.0, 50.0, (NY, NX)).astype(np.float32)
    rows = rng.normal(0.0, 3.0, (3, NY)).astype(np.float32)
    data = np.zeros((1, 4, NY, NX), dtype=np.float32)
    for j in range(1, 4):
        data[0, j] = data[0, j - 1] + scene + rows[j - 1][:, None]
    groupdq = np.zeros(data.shape, dtype=np.uint32)
    mask = np.ones((NY, NX), dtype=bool)

    cleaned = clean_ramp(
        data, groupdq, mask, -1, 2, fit_method="row_median", clean_first_group=False, scene_background=scene
    )

    for j in range(1, 4):
        np.testing.assert_allclose(cleaned[0, j] - cleaned[0, j - 1], scene, atol=1e-4)


def test_read_scene_model_reads_sci_extension(tmp_path):
    """read_scene_model on a FITS file with a SCI extension of shape (NY, NX) returns that array."""
    from astropy.io import fits

    from snowblind.stream_clean import read_scene_model

    image = np.arange(NY * NX, dtype=np.float32).reshape(NY, NX)
    path = tmp_path / "scene.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(image, name="SCI")]).writeto(path)

    scene = read_scene_model(str(path), (NY, NX))

    np.testing.assert_array_equal(scene, image)

from pathlib import Path

import numpy as np
import pytest
from astropy import coordinates as coord
from astropy.modeling import models
from gwcs import coordinate_frames as cf
from gwcs import wcs as gwcs_wcs
from jwst.datamodels import ModelLibrary
from jwst.resample.resample import ResampleImage
from jwst.resample.resample_step import GOOD_BITS
from stdatamodels.jwst.datamodels import ImageModel, dqflags
from stpipe.exceptions import ValidationError

pytest.importorskip("spizzle")

from snowblind.resample import SparseResampleImage, SparseResampleStep  # noqa: E402


SHAPE = (40, 50)
SCALE = 0.06 / 3600  # degrees per pixel
CRPIX = (24.5, 19.5)
CRVAL = (53.1, -27.8)
# x and y offsets in pixels and rotations in degrees of the dithered exposures
DITHERS = [(0.0, 0.0, 0.0), (3.3, 1.7, 0.0), (-2.6, 4.1, 0.4), (5.2, -3.4, -0.3)]
PLANES = ["data", "wht", "var_rnoise", "var_flat", "var_poisson", "err"]
RESAMPLE_KWARGS = {"weight_type": "ivm", "good_bits": GOOD_BITS, "blendheaders": False}


def tan_wcs(shape, crpix, crval, scale, angle=0.0):
    """A gwcs of a TAN projection with pixel ``crpix`` (x, y) at ``crval`` (ra, dec), ``scale`` degrees per pixel,
    rotated by ``angle`` degrees, with the bounding box and array shape of ``shape``."""
    c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    shift = models.Shift(-crpix[0]) & models.Shift(-crpix[1])
    cd = models.AffineTransformation2D(scale * np.array([[-c, s], [s, c]]), translation=[0, 0])
    det2sky = shift | cd | models.Pix2Sky_TAN() | models.RotateNative2Celestial(crval[0], crval[1], 180)
    detector = cf.Frame2D(name="detector", axes_names=("x", "y"))
    sky = cf.CelestialFrame(reference_frame=coord.ICRS(), name="world")
    wcs = gwcs_wcs.WCS([(detector, det2sky), (sky, None)])
    wcs.bounding_box = ((-0.5, shape[1] - 0.5), (-0.5, shape[0] - 0.5))
    wcs.array_shape = shape
    return wcs


def make_exposure(i, wcs, data, var_rnoise, var_flat, var_poisson):
    """An image model, exposure ``i`` with its own group, of the given arrays and WCS, with the metadata resampling
    reads."""
    model = ImageModel(data.shape)
    model.meta.wcs = wcs
    model.data = data.astype(np.float32)
    model.var_rnoise = var_rnoise.astype(np.float32)
    model.var_flat = var_flat.astype(np.float32)
    model.var_poisson = var_poisson.astype(np.float32)
    model.err = np.sqrt(model.var_rnoise + model.var_flat + model.var_poisson)
    model.dq = np.zeros(data.shape, dtype=np.uint32)
    model.meta.filename = f"exposure{i}_cal.fits"
    model.meta.group_id = f"group{i}"
    model.meta.bunit_data = "MJy/sr"
    model.meta.exposure.exposure_time = 100.0
    model.meta.exposure.measurement_time = 90.0
    model.meta.exposure.start_time = 60000.0 + i
    model.meta.exposure.end_time = 60000.0 + i + 0.001
    model.meta.exposure.duration = 110.0
    model.meta.photometry.pixelarea_steradians = np.deg2rad(SCALE) ** 2
    model.meta.photometry.pixelarea_arcsecsq = (SCALE * 3600) ** 2
    model.meta.background.subtracted = False
    model.meta.wcsinfo.ra_ref, model.meta.wcsinfo.dec_ref = wcs(*CRPIX)
    model.meta.wcsinfo.roll_ref = 0.0
    model.meta.wcsinfo.v3yangle = 0.0
    model.meta.wcsinfo.vparity = -1
    model.meta.wcsinfo.s_region = "POLYGON ICRS " + " ".join(f"{ra:.9f} {dec:.9f}" for ra, dec in wcs.footprint())
    return model


def dithered_exposures():
    """Four exposures of random data and variance planes at the offsets and rotations of DITHERS, each with one
    DO_NOT_USE pixel."""
    rng = np.random.default_rng(42)
    exposures = []
    for i, (dx, dy, angle) in enumerate(DITHERS):
        crval = (CRVAL[0] + dx * SCALE / np.cos(np.deg2rad(CRVAL[1])), CRVAL[1] + dy * SCALE)
        exposure = make_exposure(
            i,
            tan_wcs(SHAPE, CRPIX, crval, SCALE, angle),
            rng.normal(1.0, 0.3, SHAPE),
            rng.uniform(0.5, 1.5, SHAPE),
            rng.uniform(0.01, 0.03, SHAPE),
            rng.uniform(0.1, 0.4, SHAPE),
        )
        exposure.dq[5, 7] = dqflags.pixel["DO_NOT_USE"]
        exposures.append(exposure)
    return exposures


def uniform_exposure():
    """One exposure at CRVAL of data 1, VAR_RNOISE 1, VAR_FLAT 0.02 and VAR_POISSON 0.3 everywhere."""
    ones = np.ones(SHAPE)
    return make_exposure(0, tan_wcs(SHAPE, CRPIX, CRVAL, SCALE), ones, ones, 0.02 * ones, 0.3 * ones)


def resample_one(exposure, out_wcs, **kwargs):
    """SparseResampleImage of one exposure onto ``out_wcs`` with the variance planes and ERR; the resampler and its
    output model."""
    resamp = SparseResampleImage(
        ModelLibrary([exposure], on_disk=False),
        output_wcs={"wcs": out_wcs},
        enable_var=True,
        compute_err="from_var",
        **RESAMPLE_KWARGS,
        **kwargs,
    )
    return resamp, resamp.resample_many_to_one()


def inside_frame(model):
    """Pixels with weight, leaving out the outermost ring of the output frame."""
    inside = model.wht > 0
    inside[[0, -1], :] = False
    inside[:, [0, -1]] = False
    return inside


@pytest.fixture(scope="module")
def jwst_and_stcal_mode():
    """jwst's ResampleImage and SparseResampleImage with stcal's variance and every pixel's WCS, of the dithered
    exposures with the variance planes, ERR from them, and context."""
    kwargs = dict(RESAMPLE_KWARGS, enable_ctx=True, enable_var=True, compute_err="from_var")
    jwst_output = ResampleImage(ModelLibrary(dithered_exposures(), on_disk=False), **kwargs).resample_many_to_one()
    sparse_output = SparseResampleImage(
        ModelLibrary(dithered_exposures(), on_disk=False), variance="stcal", pixmap_stepsize=1, pixmap_order=1, **kwargs
    ).resample_many_to_one()
    return jwst_output, sparse_output


@pytest.mark.parametrize("plane", PLANES)
def test_stcal_mode_matches_jwst_inside_the_frame(jwst_and_stcal_mode, plane):
    """SparseResampleImage(variance="stcal", pixmap_stepsize=1) of the dithered exposures gives each plane of jwst's
    ResampleImage to float32 round-off (2e-6 relative), inside the outermost ring of the output frame.

    drizzle drops the input pixels centered outside the output frame, and jwst's frame leaves a few of a rotated
    exposure there, so that ring can differ."""
    jwst_output, sparse_output = jwst_and_stcal_mode
    inside = inside_frame(jwst_output)

    np.testing.assert_allclose(getattr(sparse_output, plane)[inside], getattr(jwst_output, plane)[inside], rtol=2e-6)


def test_stcal_mode_has_weight_where_jwst_has(jwst_and_stcal_mode):
    """SparseResampleImage(variance="stcal") of the dithered exposures has weight in the same pixels as jwst's."""
    jwst_output, sparse_output = jwst_and_stcal_mode

    np.testing.assert_array_equal(sparse_output.wht > 0, jwst_output.wht > 0)


def test_stcal_mode_context_matches_jwst(jwst_and_stcal_mode):
    """SparseResampleImage(enable_ctx=True) of the dithered exposures gives jwst's context array."""
    jwst_output, sparse_output = jwst_and_stcal_mode

    np.testing.assert_array_equal(sparse_output.con, jwst_output.con)


@pytest.mark.parametrize(
    "name", ["resample.pointings", "exposure.exposure_time", "exposure.measurement_time", "wcsinfo.s_region"]
)
def test_stcal_mode_metadata_matches_jwst(jwst_and_stcal_mode, name):
    """SparseResampleImage of the dithered exposures sets meta.<name> as jwst's ResampleImage does."""
    jwst_output, sparse_output = jwst_and_stcal_mode
    section, key = name.split(".")

    assert getattr(getattr(sparse_output.meta, section), key) == getattr(getattr(jwst_output.meta, section), key)


def test_stcal_mode_err_with_exptime_weights_matches_jwst_inside_the_frame():
    """SparseResampleImage(variance="stcal", weight_type="exptime", pixmap_stepsize=1) of the dithered exposures
    gives jwst's ERR to float32 round-off (2e-6 relative), inside the outermost ring of the output frame."""
    kwargs = dict(RESAMPLE_KWARGS, weight_type="exptime", enable_var=True, compute_err="from_var")
    jwst_output = ResampleImage(ModelLibrary(dithered_exposures(), on_disk=False), **kwargs).resample_many_to_one()
    sparse_output = SparseResampleImage(
        ModelLibrary(dithered_exposures(), on_disk=False), variance="stcal", pixmap_stepsize=1, pixmap_order=1, **kwargs
    ).resample_many_to_one()
    inside = inside_frame(jwst_output)

    np.testing.assert_allclose(sparse_output.err[inside], jwst_output.err[inside], rtol=2e-6)


def test_driz_err_matches_jwst_inside_the_frame():
    """SparseResampleImage(compute_err="driz_err", pixmap_stepsize=1) of the dithered exposures gives jwst's
    drizzled ERR to float32 round-off (2e-6 relative), inside the outermost ring of the output frame."""
    kwargs = dict(RESAMPLE_KWARGS, enable_var=False, compute_err="driz_err")
    jwst_output = ResampleImage(ModelLibrary(dithered_exposures(), on_disk=False), **kwargs).resample_many_to_one()
    sparse_output = SparseResampleImage(
        ModelLibrary(dithered_exposures(), on_disk=False), pixmap_stepsize=1, pixmap_order=1, **kwargs
    ).resample_many_to_one()
    inside = inside_frame(jwst_output)

    np.testing.assert_allclose(sparse_output.err[inside], jwst_output.err[inside], rtol=2e-6)


@pytest.mark.parametrize(("plane", "value"), [("var_rnoise", 1.0), ("var_flat", 0.02), ("var_poisson", 0.3)])
def test_exact_variance_on_the_exposure_grid_is_the_exposure_variance(plane, value):
    """SparseResampleImage of the uniform exposure onto its own grid gives each variance plane the exposure's."""
    exposure = uniform_exposure()

    _, output = resample_one(exposure, tan_wcs(SHAPE, CRPIX, CRVAL, SCALE))

    np.testing.assert_allclose(getattr(output, plane), value, rtol=1e-6)


def test_noise_factor_on_the_exposure_grid_is_one():
    """SparseResampleImage of the uniform exposure onto its own grid gives a noise factor of 1 everywhere."""
    exposure = uniform_exposure()

    resamp, _ = resample_one(exposure, tan_wcs(SHAPE, CRPIX, CRVAL, SCALE))

    np.testing.assert_allclose(resamp.noise_factor, 1.0, rtol=1e-6)


def test_noise_factor_half_a_pixel_off_the_exposure_grid_is_two():
    """SparseResampleImage of the uniform exposure onto its grid shifted by half a pixel in x gives a noise factor
    of 2, away from the two outermost columns on each side.

    Each output pixel is the mean of two input pixels, variance 1/2, and shares one of them with each neighbor
    along x, covariance 1/4 each.  The last column is half covered, and its weight raises the factor of the one
    next to it."""
    exposure = uniform_exposure()

    resamp, _ = resample_one(exposure, tan_wcs(SHAPE, (CRPIX[0] - 0.5, CRPIX[1]), CRVAL, SCALE))

    np.testing.assert_allclose(resamp.noise_factor[:, 2:-2], 2.0, rtol=1e-6)


def test_exact_variance_on_a_grid_twice_as_coarse_is_a_quarter():
    """SparseResampleImage of the uniform exposure onto a grid of twice its pixel scale, each output pixel on 2x2
    input pixels, gives VAR_RNOISE 1/4, without stcal's further division by the pixel scale ratio squared."""
    exposure = uniform_exposure()
    out_wcs = tan_wcs((SHAPE[0] // 2, SHAPE[1] // 2), (CRPIX[0] / 2 - 0.25, CRPIX[1] / 2 - 0.25), CRVAL, 2 * SCALE)

    _, output = resample_one(exposure, out_wcs)

    np.testing.assert_allclose(output.var_rnoise, 0.25, rtol=1e-6)


def test_exact_err_is_the_root_of_the_variance_times_the_noise_factor():
    """SparseResampleImage of the dithered exposures gives ERR = sqrt((VAR_RNOISE + VAR_FLAT + VAR_POISSON) * noise
    factor)."""
    resamp = SparseResampleImage(
        ModelLibrary(dithered_exposures(), on_disk=False), enable_var=True, compute_err="from_var", **RESAMPLE_KWARGS
    )

    output = resamp.resample_many_to_one()

    expected = np.sqrt((output.var_rnoise + output.var_flat + output.var_poisson) * resamp.noise_factor)
    np.testing.assert_allclose(output.err, expected, rtol=1e-6)


def test_fill_value_where_there_is_no_weight():
    """SparseResampleImage(fillval="0") of the uniform exposure onto a grid two pixels larger on each side gives
    data 0 in the first row, which no input pixel reaches."""
    exposure = uniform_exposure()
    out_wcs = tan_wcs((SHAPE[0] + 4, SHAPE[1] + 4), (CRPIX[0] + 2, CRPIX[1] + 2), CRVAL, SCALE)

    _, output = resample_one(exposure, out_wcs, fillval="0")

    np.testing.assert_array_equal(output.data[0], 0.0)


def test_variance_is_nan_where_there_is_no_weight():
    """SparseResampleImage(fillval="0") of the uniform exposure onto a grid two pixels larger on each side gives
    VAR_RNOISE NaN in the first row, as stcal does whatever the fill value."""
    exposure = uniform_exposure()
    out_wcs = tan_wcs((SHAPE[0] + 4, SHAPE[1] + 4), (CRPIX[0] + 2, CRPIX[1] + 2), CRVAL, SCALE)

    _, output = resample_one(exposure, out_wcs, fillval="0")

    assert np.all(np.isnan(output.var_rnoise[0]))


def test_no_context_by_default():
    """SparseResampleImage of the uniform exposure without enable_ctx makes no context array."""
    exposure = uniform_exposure()

    _, output = resample_one(exposure, tan_wcs(SHAPE, CRPIX, CRVAL, SCALE))

    assert "con" not in output.instance


def test_other_kernels_are_rejected():
    """SparseResampleImage(kernel="lanczos3") raises ValueError."""
    with pytest.raises(ValueError, match="square kernel"):
        SparseResampleImage(ModelLibrary([uniform_exposure()], on_disk=False), kernel="lanczos3", **RESAMPLE_KWARGS)


def test_propagate_dq_is_rejected():
    """SparseResampleImage(propagate_dq=True) raises ValueError."""
    with pytest.raises(ValueError, match="DQ"):
        SparseResampleImage(ModelLibrary([uniform_exposure()], on_disk=False), propagate_dq=True, **RESAMPLE_KWARGS)


def test_unknown_variance_mode_is_rejected():
    """SparseResampleImage(variance="drizzle") raises ValueError."""
    with pytest.raises(ValueError, match="variance"):
        SparseResampleImage(ModelLibrary([uniform_exposure()], on_disk=False), variance="drizzle", **RESAMPLE_KWARGS)


def test_indef_fill_value_is_rejected():
    """SparseResampleImage(fillval="INDEF") raises ValueError: fill values are numbers."""
    with pytest.raises(ValueError):
        SparseResampleImage(ModelLibrary([uniform_exposure()], on_disk=False), fillval="INDEF", **RESAMPLE_KWARGS)


def test_finalize_without_the_noise_factor_pass_raises():
    """SparseResampleImage.add_model of an exposure and then finalize, with the exact variance, raises RuntimeError."""
    exposure = uniform_exposure()
    resamp = SparseResampleImage(ModelLibrary([exposure], on_disk=False), enable_var=True, **RESAMPLE_KWARGS)
    resamp.add_model(exposure)

    with pytest.raises(RuntimeError, match="noise factor"):
        resamp.finalize()


def test_step_saves_the_noise_factor_file_it_records(tmp_cwd):
    """SparseResampleStep.call of the dithered exposures with save_results saves the noise factor file named in
    meta.snowblind.sparse_resample.noise_factor_file."""
    result = SparseResampleStep.call(dithered_exposures(), save_results=True)

    name = result.meta.instance["snowblind"]["sparse_resample"]["noise_factor_file"]
    assert (tmp_cwd / name).is_file()


def test_step_names_the_noise_factor_file_with_its_suffix(tmp_cwd):
    """SparseResampleStep.call of the dithered exposures with save_results names the noise factor file with the
    suffix nfactor."""
    result = SparseResampleStep.call(dithered_exposures(), save_results=True)

    assert result.meta.instance["snowblind"]["sparse_resample"]["noise_factor_file"].endswith("_nfactor.fits")


def test_step_noise_factor_file_holds_the_noise_factor(tmp_cwd):
    """SparseResampleStep.call of the dithered exposures with save_results saves a noise factor file whose SCI is
    SparseResampleImage's noise factor."""
    result = SparseResampleStep.call(dithered_exposures(), save_results=True)
    resamp = SparseResampleImage(
        ModelLibrary(dithered_exposures(), on_disk=False), enable_var=True, compute_err="from_var", **RESAMPLE_KWARGS
    )
    resamp.resample_many_to_one()

    name = result.meta.instance["snowblind"]["sparse_resample"]["noise_factor_file"]
    with ImageModel(str(tmp_cwd / name)) as noise_factor:
        np.testing.assert_allclose(noise_factor.data, resamp.noise_factor, rtol=1e-6)


def test_step_without_saving_writes_no_file(tmp_cwd):
    """SparseResampleStep.call of the dithered exposures without save_results writes no file."""
    SparseResampleStep.call(dithered_exposures())

    assert list(Path(tmp_cwd).iterdir()) == []


def test_step_without_saving_records_no_noise_factor_file():
    """SparseResampleStep.call of the dithered exposures without save_results records no noise factor file."""
    result = SparseResampleStep.call(dithered_exposures())

    assert "noise_factor_file" not in result.meta.instance["snowblind"]["sparse_resample"]


def test_step_records_the_variance_mode():
    """SparseResampleStep.call(variance="stcal") of the dithered exposures records variance "stcal"."""
    result = SparseResampleStep.call(dithered_exposures(), variance="stcal")

    assert result.meta.instance["snowblind"]["sparse_resample"]["variance"] == "stcal"


def test_step_sets_its_cal_step():
    """SparseResampleStep.call of the dithered exposures sets meta.cal_step.sparse_resample to COMPLETE."""
    result = SparseResampleStep.call(dithered_exposures())

    assert result.meta.cal_step.instance["sparse_resample"] == "COMPLETE"


def test_step_has_no_context_by_default():
    """SparseResampleStep.call of the dithered exposures makes no context array."""
    result = SparseResampleStep.call(dithered_exposures())

    assert "con" not in result.instance


def test_step_rejects_the_kernel_parameter():
    """SparseResampleStep.call(kernel="square") raises ValidationError: the step has no kernel parameter."""
    with pytest.raises(ValidationError, match="kernel"):
        SparseResampleStep.call(dithered_exposures(), kernel="square")


def test_step_single_gives_one_image_per_group():
    """SparseResampleStep.call(single=True) of the four dithered exposures gives a library of four images."""
    result = SparseResampleStep.call(dithered_exposures(), single=True)

    assert len(result) == 4

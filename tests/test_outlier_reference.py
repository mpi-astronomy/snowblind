import numpy as np
import pytest
from astropy import coordinates as coord
from astropy.modeling import models
from gwcs import coordinate_frames as cf
from gwcs import wcs as gwcs_wcs
from scipy import sparse
from stdatamodels.jwst.datamodels import ImageModel

pytest.importorskip("spizzle")

from snowblind import outlier_reference  # noqa: E402


SHAPE = (20, 20)
# the dithered exposures: their shape, pixel scale in degrees, and x and y offsets in pixels and rotations in degrees
EXPOSURE_SHAPE = (40, 50)
SCALE = 0.06 / 3600
CRVAL = (53.1, -27.8)
DITHERS = [(0.0, 0.0, 0.0), (3.3, 1.7, 0.0), (-2.6, 4.1, 0.4), (5.2, -3.4, -0.3)]
# the exposure and the pixel (y, x) that dithered_exposures hits with a cosmic ray
HIT_EXPOSURE, HIT_PIXEL = 1, (20, 25)


def one_row(weights):
    """A sparse array of one row listing input values 0, 1, ... with ``weights``."""
    return sparse.csr_array(np.array([weights], dtype=np.float64))


def flat_image(sky=1.0):
    """An image model of data ``sky`` and error 0.1 everywhere, with an empty dq."""
    model = ImageModel(SHAPE)
    model.data = np.full(SHAPE, sky, dtype=np.float32)
    model.err = np.full(SHAPE, 0.1, dtype=np.float32)
    model.dq = np.zeros(SHAPE, dtype=np.uint32)
    return model


class Predicted:
    """An exposure's prediction as `outlier_reference.predict` gives it: flat at 1, from ``n`` exposures."""

    def __init__(self, model, n):
        self.model = model
        self.prediction = np.ones(SHAPE, dtype=np.float32)
        self.spread = np.zeros(SHAPE, dtype=np.float32)
        self.n_exposures = np.full(SHAPE, n, dtype=np.uint8)


def test_weighted_median_of_equal_weights_is_the_middle_value():
    """weighted_median of one row listing 1, 2 and 10 with equal weights returns 2."""
    result = outlier_reference.weighted_median(one_row([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 10.0]))

    assert result[0] == 2.0


def test_weighted_median_follows_the_weights():
    """weighted_median of one row listing 5, 1 and 2 with weights 3, 1 and 1 returns 5."""
    result = outlier_reference.weighted_median(one_row([3.0, 1.0, 1.0]), np.array([5.0, 1.0, 2.0]))

    assert result[0] == 5.0


def test_weighted_median_leaves_out_nan_values():
    """weighted_median of one row listing NaN and 4 returns 4."""
    result = outlier_reference.weighted_median(one_row([1.0, 1.0]), np.array([np.nan, 4.0]))

    assert result[0] == 4.0


def test_weighted_median_of_an_empty_row_is_nan():
    """weighted_median of a row without entries returns NaN."""
    result = outlier_reference.weighted_median(sparse.csr_array((1, 2)), np.array([1.0, 2.0]))

    assert np.isnan(result[0])


def test_clipped_mean_leaves_out_values_beyond_3_sigma():
    """clipped_mean of one row listing 1, 1.1, 0.9 and 50, each of sigma 0.1, returns 1."""
    values = np.array([1.0, 1.1, 0.9, 50.0])

    result = outlier_reference.clipped_mean(one_row([1.0] * 4), values, np.full(4, 0.1))

    assert result[0] == pytest.approx(1.0)


def test_sky_level_is_the_level_skymatch_measured():
    """sky_level of a model with background level 3, not subtracted, returns 3."""
    model = flat_image()
    model.meta.background.level = 3.0
    model.meta.background.subtracted = False

    assert outlier_reference.sky_level(model) == 3.0


def test_sky_level_is_zero_once_subtracted():
    """sky_level of a model with background level 3, subtracted, returns 0."""
    model = flat_image()
    model.meta.background.level = 3.0
    model.meta.background.subtracted = True

    assert outlier_reference.sky_level(model) == 0.0


def test_sky_level_without_skymatch_is_the_median():
    """sky_level of a model of data 2 without a background level returns 2."""
    assert outlier_reference.sky_level(flat_image(sky=2.0)) == pytest.approx(2.0)


def test_with_two_exposures_a_pixel_above_the_prediction_is_flagged():
    """flag_outliers of a flat image with a spike of +5 at (10, 10), against a flat prediction from two exposures
    everywhere, flags (10, 10)."""
    model = flat_image()
    model.data[10, 10] += 5.0

    flags = outlier_reference.flag_outliers(Predicted(model, 2))

    assert flags[10, 10]


def test_with_two_exposures_a_pixel_below_the_prediction_is_not_flagged():
    """flag_outliers of a flat image with a dip of -5 at (10, 10), against a flat prediction from two exposures
    everywhere, flags nothing: the outlier may be in the other exposure."""
    model = flat_image()
    model.data[10, 10] -= 5.0

    flags = outlier_reference.flag_outliers(Predicted(model, 2))

    assert not flags.any()


def test_with_three_exposures_a_pixel_below_the_prediction_is_flagged():
    """flag_outliers of a flat image with a dip of -5 at (10, 10), against a flat prediction from three exposures
    everywhere, flags (10, 10)."""
    model = flat_image()
    model.data[10, 10] -= 5.0

    flags = outlier_reference.flag_outliers(Predicted(model, 3))

    assert flags[10, 10]


def tan_wcs(shape, crpix, crval, angle=0.0):
    """A gwcs of a TAN projection of ``shape`` with pixel ``crpix`` at ``crval``, SCALE degrees per pixel, rotated by
    ``angle`` degrees."""
    c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    shift = models.Shift(-crpix[0]) & models.Shift(-crpix[1])
    cd = models.AffineTransformation2D(SCALE * np.array([[-c, s], [s, c]]), translation=[0, 0])
    det2sky = shift | cd | models.Pix2Sky_TAN() | models.RotateNative2Celestial(crval[0], crval[1], 180)
    detector = cf.Frame2D(name="detector", axes_names=("x", "y"))
    sky = cf.CelestialFrame(reference_frame=coord.ICRS(), name="world")
    wcs = gwcs_wcs.WCS([(detector, det2sky), (sky, None)])
    wcs.bounding_box = ((-0.5, shape[1] - 0.5), (-0.5, shape[0] - 0.5))
    wcs.array_shape = shape
    return wcs


def dithered_exposures():
    """The four exposures at DITHERS, of data 1 plus noise of 0.1 and error 0.1, HIT_EXPOSURE with a cosmic ray of 5
    at HIT_PIXEL."""
    rng = np.random.default_rng(3)
    exposures = []
    for dx, dy, angle in DITHERS:
        model = ImageModel(EXPOSURE_SHAPE)
        crval = (CRVAL[0] + dx * SCALE / np.cos(np.deg2rad(CRVAL[1])), CRVAL[1] + dy * SCALE)
        model.meta.wcs = tan_wcs(EXPOSURE_SHAPE, (24.5, 19.5), crval, angle)
        model.data = rng.normal(1.0, 0.1, EXPOSURE_SHAPE).astype(np.float32)
        model.err = np.full(EXPOSURE_SHAPE, 0.1, dtype=np.float32)
        model.var_rnoise = np.full(EXPOSURE_SHAPE, 0.01, dtype=np.float32)
        model.dq = np.zeros(EXPOSURE_SHAPE, dtype=np.uint32)
        exposures.append(model)
    exposures[HIT_EXPOSURE].data[HIT_PIXEL] += 5.0
    return exposures


def mosaic_wcs():
    """A TAN grid of 60 x 70 pixels at SCALE around CRVAL, which holds every dithered exposure."""
    return tan_wcs((60, 70), (34.5, 29.5), CRVAL)


def test_prediction_does_not_depend_on_the_tile_size():
    """predict of the dithered exposures with tiles of 16 fine pixels gives the predictions of tiles of 1024."""
    small = outlier_reference.predict(dithered_exposures(), mosaic_wcs(), tile=16)
    whole = outlier_reference.predict(dithered_exposures(), mosaic_wcs(), tile=1024)

    for a, b in zip(small, whole):
        np.testing.assert_allclose(a.prediction, b.prediction, rtol=1e-6, equal_nan=True)


def test_flags_the_cosmic_ray_of_one_exposure():
    """flag_outliers of HIT_EXPOSURE, predicted from the four dithered exposures, flags HIT_PIXEL."""
    exposures = outlier_reference.predict(dithered_exposures(), mosaic_wcs())

    assert outlier_reference.flag_outliers(exposures[HIT_EXPOSURE])[HIT_PIXEL]


def test_flags_nothing_in_an_exposure_without_a_cosmic_ray():
    """flag_outliers of dithered exposure 0, which has no cosmic ray, predicted from all four, flags no pixel."""
    exposures = outlier_reference.predict(dithered_exposures(), mosaic_wcs())

    assert not outlier_reference.flag_outliers(exposures[0]).any()


def test_flags_the_cosmic_ray_with_two_exposures():
    """flag_outliers of HIT_EXPOSURE, predicted from dithered exposures 0 and 1 only, flags HIT_PIXEL: with two
    exposures the reference is their weighted mean, which halves the cosmic ray, rather than a median that takes
    one of them."""
    exposures = outlier_reference.predict(dithered_exposures()[:2], mosaic_wcs())

    assert outlier_reference.flag_outliers(exposures[HIT_EXPOSURE])[HIT_PIXEL]

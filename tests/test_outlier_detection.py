import numpy as np
import pytest
from astropy import coordinates as coord
from astropy.modeling import models
from gwcs import coordinate_frames as cf
from gwcs import wcs as gwcs_wcs
from jwst.datamodels import ModelLibrary
from jwst.outlier_detection import OutlierDetectionStep
from stdatamodels.jwst.datamodels import ImageModel, dqflags
from stpipe.exceptions import ValidationError

pytest.importorskip("spizzle")

from snowblind import SparseOutlierDetectionStep  # noqa: E402


SHAPE = (40, 50)
SCALE = 0.06 / 3600  # degrees per pixel
CRPIX = (24.5, 19.5)
CRVAL = (53.1, -27.8)
# x and y offsets in pixels and rotations in degrees of the dithered exposures
DITHERS = [(0.0, 0.0, 0.0), (3.3, 1.7, 0.0), (-2.6, 4.1, 0.4), (5.2, -3.4, -0.3)]
OUTLIER = dqflags.pixel["OUTLIER"]
# the exposure and the pixel (y, x) that cosmic_ray_exposures hits
HIT_EXPOSURE, HIT_PIXEL = 1, (20, 25)


def tan_wcs(crval, angle):
    """A gwcs of a TAN projection of SHAPE with pixel CRPIX at ``crval``, SCALE degrees per pixel, rotated by
    ``angle`` degrees."""
    c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    shift = models.Shift(-CRPIX[0]) & models.Shift(-CRPIX[1])
    cd = models.AffineTransformation2D(SCALE * np.array([[-c, s], [s, c]]), translation=[0, 0])
    det2sky = shift | cd | models.Pix2Sky_TAN() | models.RotateNative2Celestial(crval[0], crval[1], 180)
    detector = cf.Frame2D(name="detector", axes_names=("x", "y"))
    sky = cf.CelestialFrame(reference_frame=coord.ICRS(), name="world")
    wcs = gwcs_wcs.WCS([(detector, det2sky), (sky, None)])
    wcs.bounding_box = ((-0.5, SHAPE[1] - 0.5), (-0.5, SHAPE[0] - 0.5))
    wcs.array_shape = SHAPE
    return wcs


def cosmic_ray_exposures():
    """The four exposures at DITHERS, of data 1 plus noise of 0.3 and a variance of about 1.3, the exposure
    HIT_EXPOSURE with a cosmic ray of 50 at HIT_PIXEL."""
    rng = np.random.default_rng(42)
    exposures = []
    for i, (dx, dy, angle) in enumerate(DITHERS):
        model = ImageModel(SHAPE)
        crval = (CRVAL[0] + dx * SCALE / np.cos(np.deg2rad(CRVAL[1])), CRVAL[1] + dy * SCALE)
        model.meta.wcs = tan_wcs(crval, angle)
        model.data = rng.normal(1.0, 0.3, SHAPE).astype(np.float32)
        model.var_rnoise = np.full(SHAPE, 1.0, dtype=np.float32)
        model.var_flat = np.full(SHAPE, 0.02, dtype=np.float32)
        model.var_poisson = np.full(SHAPE, 0.3, dtype=np.float32)
        model.err = np.sqrt(model.var_rnoise + model.var_flat + model.var_poisson)
        model.dq = np.zeros(SHAPE, dtype=np.uint32)
        model.meta.filename = f"exposure{i}_cal.fits"
        model.meta.group_id = f"group{i}"
        model.meta.bunit_data = "MJy/sr"
        model.meta.exposure.type = "NRC_IMAGE"
        model.meta.exposure.exposure_time = 100.0
        model.meta.exposure.measurement_time = 90.0
        model.meta.exposure.start_time = 60000.0 + i
        model.meta.exposure.end_time = 60000.0 + i + 0.001
        model.meta.exposure.duration = 110.0
        model.meta.photometry.pixelarea_steradians = np.deg2rad(SCALE) ** 2
        model.meta.photometry.pixelarea_arcsecsq = (SCALE * 3600) ** 2
        model.meta.background.subtracted = False
        model.meta.wcsinfo.ra_ref, model.meta.wcsinfo.dec_ref = model.meta.wcs(*CRPIX)
        model.meta.wcsinfo.roll_ref = 0.0
        model.meta.wcsinfo.v3yangle = 0.0
        model.meta.wcsinfo.vparity = -1
        corners = model.meta.wcs.footprint()
        model.meta.wcsinfo.s_region = "POLYGON ICRS " + " ".join(f"{ra:.9f} {dec:.9f}" for ra, dec in corners)
        exposures.append(model)
    exposures[HIT_EXPOSURE].data[HIT_PIXEL] += 50.0
    return exposures


def outlier_flags(step):
    """The OUTLIER flags that ``step``, in imaging mode, sets in the cosmic-ray exposures: one image per exposure."""
    step.mode = "imaging"
    result = step.run(ModelLibrary(cosmic_ray_exposures(), on_disk=False))
    flags = []
    with result:
        for model in result:
            flags.append((model.dq & OUTLIER) > 0)
            result.shelve(model, modify=False)
    return np.array(flags)


def test_flags_the_cosmic_ray():
    """SparseOutlierDetectionStep of the cosmic-ray exposures flags HIT_PIXEL of HIT_EXPOSURE as OUTLIER."""
    flags = outlier_flags(SparseOutlierDetectionStep())

    assert flags[HIT_EXPOSURE][HIT_PIXEL]


def test_flags_nothing_but_the_cosmic_ray():
    """SparseOutlierDetectionStep of the cosmic-ray exposures flags one pixel in all, the cosmic ray."""
    flags = outlier_flags(SparseOutlierDetectionStep())

    assert flags.sum() == 1


def test_flags_match_jwst():
    """SparseOutlierDetectionStep(pixmap_stepsize=1) of the cosmic-ray exposures flags the pixels that jwst's
    OutlierDetectionStep(pixmap_stepsize=1) flags."""
    flags = outlier_flags(SparseOutlierDetectionStep(pixmap_stepsize=1, pixmap_order=1))
    jwst_flags = outlier_flags(OutlierDetectionStep(pixmap_stepsize=1, pixmap_order=1))

    np.testing.assert_array_equal(flags, jwst_flags)


def test_sets_its_cal_step():
    """SparseOutlierDetectionStep of the cosmic-ray exposures sets meta.cal_step.sparse_outlier_detection to
    COMPLETE in each."""
    step = SparseOutlierDetectionStep()
    step.mode = "imaging"

    result = step.run(ModelLibrary(cosmic_ray_exposures(), on_disk=False))

    statuses = []
    with result:
        for model in result:
            statuses.append(model.meta.cal_step.instance["sparse_outlier_detection"])
            result.shelve(model, modify=False)
    assert statuses == ["COMPLETE"] * len(DITHERS)


def test_rejects_other_modes():
    """SparseOutlierDetectionStep with mode "spec" raises ValueError."""
    step = SparseOutlierDetectionStep()
    step.mode = "spec"

    with pytest.raises(ValueError, match="images only"):
        step.run(ModelLibrary(cosmic_ray_exposures(), on_disk=False))


def test_rejects_the_kernel_parameter():
    """SparseOutlierDetectionStep(kernel="square") raises ValidationError: the step has no kernel parameter."""
    with pytest.raises(ValidationError, match="kernel"):
        SparseOutlierDetectionStep.call(ModelLibrary(cosmic_ray_exposures(), on_disk=False), kernel="square")

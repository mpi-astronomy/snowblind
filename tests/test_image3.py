import os

import numpy as np
import pytest
from astropy import coordinates as coord
from astropy.io import fits
from astropy.modeling import models
from gwcs import coordinate_frames as cf
from gwcs import wcs as gwcs_wcs
from jwst.associations import asn_from_list
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base
from stdatamodels.jwst.datamodels import ImageModel, dqflags
from stpipe import crds_client

pytest.importorskip("spizzle")

from snowblind import Image3Pipeline, SparseOutlierDetectionStep, SparseResampleStep, TiledOutlierStep  # noqa: E402


SHAPE = (40, 50)
SCALE = 0.06 / 3600  # degrees per pixel
CRPIX = (24.5, 19.5)
CRVAL = (53.1, -27.8)
# x and y offsets in pixels and rotations in degrees of the dithered exposures
DITHERS = [(0.0, 0.0, 0.0), (3.3, 1.7, 0.0), (-2.6, 4.1, 0.4), (5.2, -3.4, -0.3)]
# the exposure and the pixel (y, x) that the cosmic ray hits
HIT_EXPOSURE, HIT_PIXEL = 1, (20, 25)
PRODUCT = "mosaic"
SKIPPED = {"tweakreg": {"skip": True}, "skymatch": {"skip": True}, "source_catalog": {"skip": True}}


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


def write_exposures():
    """Save the four exposures at DITHERS, of data 1 plus noise of 0.3, HIT_EXPOSURE with a cosmic ray of 50 at
    HIT_PIXEL, and an association of them for the product PRODUCT; returns the association's file name."""
    rng = np.random.default_rng(42)
    names = []
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
        if i == HIT_EXPOSURE:
            model.data[HIT_PIXEL] += 50.0
        model.meta.filename = f"exposure{i}_cal.fits"
        model.meta.bunit_data = "MJy/sr"
        model.meta.exposure.type = "NRC_IMAGE"
        model.meta.exposure.exposure_time = 100.0
        model.meta.exposure.measurement_time = 90.0
        model.meta.exposure.start_time = 60000.0 + i
        model.meta.exposure.end_time = 60000.0 + i + 0.001
        model.meta.exposure.duration = 110.0
        model.meta.observation.exposure_number = str(i + 1)
        model.meta.photometry.pixelarea_steradians = np.deg2rad(SCALE) ** 2
        model.meta.photometry.pixelarea_arcsecsq = (SCALE * 3600) ** 2
        model.meta.background.subtracted = False
        model.meta.wcsinfo.ra_ref, model.meta.wcsinfo.dec_ref = model.meta.wcs(*CRPIX)
        model.meta.wcsinfo.roll_ref = 0.0
        model.meta.wcsinfo.v3yangle = 0.0
        model.meta.wcsinfo.vparity = -1
        corners = model.meta.wcs.footprint()
        model.meta.wcsinfo.s_region = "POLYGON ICRS " + " ".join(f"{ra:.9f} {dec:.9f}" for ra, dec in corners)
        model.save(model.meta.filename)
        names.append(model.meta.filename)
    _, text = asn_from_list.asn_from_list(names, rule=DMS_Level3_Base, product_name=PRODUCT).dump(format="json")
    with open(f"{PRODUCT}_asn.json", "w") as asn_file:
        asn_file.write(text)
    return f"{PRODUCT}_asn.json"


def run_pipeline(folder, **parameters):
    """Run Image3Pipeline in ``folder`` on the association of the four exposures, with tweakreg, skymatch and
    source_catalog skipped, saving its results, without CRDS step parameters; returns ``folder``."""
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(folder)
        patch.setenv("STPIPE_DISABLE_CRDS_STEPPARS", "True")
        patch.setattr(crds_client, "get_context_used", lambda observatory: "jwst_0000.pmap")
        Image3Pipeline.call(write_exposures(), save_results=True, steps=SKIPPED, **parameters)
    return folder


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    """The folder in which Image3Pipeline ran with its defaults (run_pipeline)."""
    return run_pipeline(tmp_path_factory.mktemp("image3"))


@pytest.fixture(scope="module")
def sparse_outlier_run(tmp_path_factory):
    """The folder in which Image3Pipeline ran with outlier_step "sparse_outlier" (run_pipeline)."""
    return run_pipeline(tmp_path_factory.mktemp("image3_sparse"), outlier_step="sparse_outlier")


def test_tiled_outlier_slot_holds_the_tiled_step():
    """Image3Pipeline().tiled_outlier is a TiledOutlierStep."""
    assert isinstance(Image3Pipeline().tiled_outlier, TiledOutlierStep)


def test_sparse_outlier_slot_holds_the_sparse_step():
    """Image3Pipeline().sparse_outlier is a SparseOutlierDetectionStep."""
    assert isinstance(Image3Pipeline().sparse_outlier, SparseOutlierDetectionStep)


def test_the_tiled_outlier_step_runs_by_default():
    """Image3Pipeline().outlier_step is "tiled_outlier"."""
    assert Image3Pipeline().outlier_step == "tiled_outlier"


def test_sparse_outlier_step_keeps_jwsts_scale():
    """Image3Pipeline().sparse_outlier.scale is jwst's "1.2 0.7", not the tiled step's."""
    assert Image3Pipeline().sparse_outlier.scale == "1.2 0.7"


def test_config_takes_the_sparse_steps_own_parameters():
    """Image3Pipeline.build_config with steps.sparse_outlier.pixfrac 0.8, a parameter only that step has, keeps 0.8."""
    config, _ = Image3Pipeline.build_config(None, outlier_step="sparse_outlier", steps={"sparse_outlier": {"pixfrac": 0.8}})

    assert config["steps"]["sparse_outlier"]["pixfrac"] == 0.8


def test_saved_parameters_build_the_config_again():
    """Image3Pipeline.build_config of the parameters of Image3Pipeline(outlier_step="sparse_outlier") keeps
    outlier_step."""
    parameters = Image3Pipeline(outlier_step="sparse_outlier").get_pars()

    config, _ = Image3Pipeline.build_config(None, **parameters)

    assert config["outlier_step"] == "sparse_outlier"


def test_resample_slot_holds_the_sparse_step():
    """Image3Pipeline().resample is a SparseResampleStep."""
    assert isinstance(Image3Pipeline().resample, SparseResampleStep)


def test_writes_the_mosaic(pipeline_run):
    """Image3Pipeline of the association saves the mosaic as <product>_i2d.fits."""
    assert os.path.isfile(pipeline_run / f"{PRODUCT}_i2d.fits")


def test_writes_the_noise_factor_next_to_the_mosaic(pipeline_run):
    """Image3Pipeline of the association saves the noise factor as <product>_nfactor.fits."""
    assert os.path.isfile(pipeline_run / f"{PRODUCT}_nfactor.fits")


def test_mosaic_names_its_noise_factor_file(pipeline_run):
    """Image3Pipeline of the association writes NFACFILE = <product>_nfactor.fits into the mosaic's header."""
    assert fits.getval(pipeline_run / f"{PRODUCT}_i2d.fits", "NFACFILE") == f"{PRODUCT}_nfactor.fits"


def test_flags_the_cosmic_ray_in_the_crf_file(pipeline_run):
    """Image3Pipeline of the association flags HIT_PIXEL as OUTLIER in the crf file of HIT_EXPOSURE."""
    [crf] = pipeline_run.glob(f"exposure{HIT_EXPOSURE}_*_crf.fits")

    with ImageModel(str(crf)) as model:
        assert model.dq[HIT_PIXEL] & dqflags.pixel["OUTLIER"]


def test_tiled_outlier_step_flagged_the_crf_files(pipeline_run):
    """Image3Pipeline of the association records TiledOutlierStep's status in the crf files."""
    [crf] = pipeline_run.glob(f"exposure{HIT_EXPOSURE}_*_crf.fits")

    assert fits.getval(crf, "S_TILOUT") == "COMPLETE"


def test_sparse_outlier_step_flags_the_cosmic_ray_in_the_crf_file(sparse_outlier_run):
    """Image3Pipeline(outlier_step="sparse_outlier") of the association flags HIT_PIXEL as OUTLIER in the crf file of
    HIT_EXPOSURE."""
    [crf] = sparse_outlier_run.glob(f"exposure{HIT_EXPOSURE}_*_crf.fits")

    with ImageModel(str(crf)) as model:
        assert model.dq[HIT_PIXEL] & dqflags.pixel["OUTLIER"]


def test_sparse_outlier_step_flagged_the_crf_files(sparse_outlier_run):
    """Image3Pipeline(outlier_step="sparse_outlier") of the association records SparseOutlierDetectionStep's status
    in the crf files."""
    [crf] = sparse_outlier_run.glob(f"exposure{HIT_EXPOSURE}_*_crf.fits")

    assert fits.getval(crf, "S_SPOUTL") == "COMPLETE"

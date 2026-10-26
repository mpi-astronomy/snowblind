import numpy as np
from astropy.io import fits
from jwst import datamodels

from snowblind import OpenPixelStep


ADJ_OPEN = datamodels.dqflags.pixel["ADJ_OPEN"]
DO_NOT_USE = datamodels.dqflags.pixel["DO_NOT_USE"]
GOOD = datamodels.dqflags.pixel["GOOD"]


def test_init():
    step = OpenPixelStep(threshold=4.5)

    assert step.threshold == 4.5


def test_call():
    images = datamodels.ModelContainer()
    rng = np.random.default_rng()

    mean = 0.0
    stddev = 0.08
    shape = (10, 10)
    for _ in range(40):
        image = datamodels.ImageModel(rng.normal(loc=mean, scale=stddev, size=shape))
        image.dq = np.zeros(shape, dtype=np.uint32)
        images.append(image)

    for i, image in enumerate(images):
        # Populate detector meta, half A module, half B
        if i < 20:
            image.meta.instrument.detector = "NRCALONG"
            image.meta.filename = f"jw001234_{i}_nrcalong.fits"
        else:
            image.meta.instrument.detector = "NRCBLONG"
            image.meta.filename = f"jw001234_{i}_nrcblong.fits"

        # Drop some hot and dead pixels in
        image.data[2, 2] += 10 * stddev
        image.data[3, 5] += 5 * stddev
        image.data[8, 8] += 3 * stddev

    # Run the step and see if they're recovered
    results = OpenPixelStep.call(images, threshold=3.0)

    for result in results:
        assert result.dq[2, 2] == ADJ_OPEN | DO_NOT_USE
        assert result.dq[3, 5] == ADJ_OPEN | DO_NOT_USE
        assert result.dq[8, 8] == ADJ_OPEN | DO_NOT_USE

        assert result.dq[5, 5] == GOOD


def test_cal_step_complete_in_fits_header(tmp_path):
    """OpenPixelStep.call on three images of one detector: each result saves with S_OPNPIX = COMPLETE."""
    images = datamodels.ModelContainer()
    for i in range(3):
        image = datamodels.ImageModel(np.zeros((10, 10)))
        image.dq = np.zeros((10, 10), dtype=np.uint32)
        image.meta.instrument.detector = "NRCALONG"
        image.meta.filename = f"jw001234_{i}_nrcalong.fits"
        images.append(image)

    results = OpenPixelStep.call(images)
    results[0].save(tmp_path / "result.fits")

    assert results[0].meta.cal_step.open_pixel == "COMPLETE"
    assert fits.getheader(tmp_path / "result.fits")["S_OPNPIX"] == "COMPLETE"

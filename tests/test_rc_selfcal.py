from stdatamodels.jwst import datamodels
import numpy as np
from jwst.datamodels import ModelContainer

from snowblind import RcSelfCalStep


RC = datamodels.dqflags.pixel["RC"]
DO_NOT_USE = datamodels.dqflags.pixel["DO_NOT_USE"]
GOOD = datamodels.dqflags.pixel["GOOD"]


def test_init():
    glob_pattern = "jw*_nrclong_rate.fits"
    step = RcSelfCalStep(glob_pattern=glob_pattern)

    assert step.glob_pattern == glob_pattern


def test_call():
    images = ModelContainer()
    rng = np.random.default_rng()

    mean = 0.02
    stddev = 0.05
    shape = (10, 10)
    for _ in range(10):
        image = rng.normal(loc=mean, scale=stddev, size=shape)
        images.append(datamodels.ImageModel(image))

    for image in images:
        # We need to know the detector
        image.meta.instrument.detector = "NRCALONG"

        # Drop some hot and dead pixels in
        image.data[2, 2] = mean - 10 * stddev
        image.data[3, 5] = mean + 5 * stddev
        image.data[8, 8] = mean + 3 * stddev

    # Run the step and see if they're recovered
    results = RcSelfCalStep.call(images)

    for image in images:
        assert image.dq[2, 2] == RC & DO_NOT_USE
        assert image.dq[3, 5] == RC & DO_NOT_USE
        assert image.dq[8, 8] == RC & DO_NOT_USE

        assert image.dq[5, 5] == GOOD

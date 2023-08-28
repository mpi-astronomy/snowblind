import numpy as np
from jwst import datamodels

from snowblind import RcSelfCalStep


RC = datamodels.dqflags.pixel["RC"]
DO_NOT_USE = datamodels.dqflags.pixel["DO_NOT_USE"]
GOOD = datamodels.dqflags.pixel["GOOD"]


def test_init():
    step = RcSelfCalStep(threshold=4.5)

    assert step.threshold == 4.5


def test_call():
    images = datamodels.ModelContainer()
    rng = np.random.default_rng()

    mean = 0.0
    stddev = 0.08
    shape = (10, 10)
    for _ in range(40):
        image = rng.normal(loc=mean, scale=stddev, size=shape)
        images.append(datamodels.ImageModel(image))

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
    results = RcSelfCalStep.call(images, threshold=3.0)

    for result in results:
        assert result.dq[2, 2] == RC | DO_NOT_USE
        assert result.dq[3, 5] == RC | DO_NOT_USE
        assert result.dq[8, 8] == RC | DO_NOT_USE

        assert result.dq[5, 5] == GOOD

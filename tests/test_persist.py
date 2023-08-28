import numpy as np
from jwst import datamodels
from astropy.time import Time

from snowblind import PersistenceFlagStep


SATURATED = datamodels.dqflags.group["SATURATED"]
PERSISTENCE = datamodels.dqflags.pixel["PERSISTENCE"]
DO_NOT_USE = datamodels.dqflags.pixel["DO_NOT_USE"]
GOOD = datamodels.dqflags.pixel["GOOD"]


def test_init():
    step = PersistenceFlagStep(time=1000)

    assert step.time == 1000


def test_call():
    images = datamodels.ModelContainer()

    time0 = Time(60122.0226664904, format="mjd")
    start_times = []
    for i in range(12):
        images.append(datamodels.ImageModel((10, 10)))

        # Populate start times
        start_time = time0.value + i * 0.0132
        start_times.append(start_time)

    # Make the start times not in order.
    start_times = np.roll(start_times, shift=4)

    for i, image in enumerate(images):
        # Populate detector meta
        image.meta.instrument.detector = "NRCALONG"
        image.meta.filename = f"jw001234_{i}_nrcalong.fits"

        image.meta.exposure.start_time = start_times[i]

    # Drop saturated pixels into the images
    images[0].dq[2, 2] = SATURATED
    images[1].dq[3, 5] = SATURATED
    images[2].dq[8, 8] = SATURATED

    # Run the step and see if they're recovered
    results = PersistenceFlagStep.call(images)

    assert results[1].dq[2, 2] & (PERSISTENCE | DO_NOT_USE)
    assert results[2].dq[3, 5] & (PERSISTENCE | DO_NOT_USE)
    assert results[3].dq[8, 8] & (PERSISTENCE | DO_NOT_USE)

    assert results[8].dq[2, 2] == GOOD
    assert results[1].dq[0, 0] == GOOD

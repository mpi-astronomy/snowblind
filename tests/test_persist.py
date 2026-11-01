import json

import numpy as np
from jwst import datamodels
from astropy.io import fits
from astropy.time import Time

from snowblind import PersistenceFlagStep


SATURATED = datamodels.dqflags.pixel["SATURATED"]
PERSISTENCE = datamodels.dqflags.pixel["PERSISTENCE"]
DO_NOT_USE = datamodels.dqflags.pixel["DO_NOT_USE"]
GOOD = datamodels.dqflags.pixel["GOOD"]


def make_image(i, detector="NRCALONG", start_time=60122.0):
    """A 10x10 image model with an empty dq, named as exposure ``i`` of ``detector``."""
    image = datamodels.ImageModel((10, 10))
    image.dq = np.zeros((10, 10), dtype=np.uint32)
    image.meta.instrument.detector = detector
    image.meta.filename = f"jw01125002001_03101_0000{i}_{detector.lower()}_cal.fits"
    image.meta.exposure.start_time = start_time
    return image


def saturate(image, y, x):
    """Set SATURATED in the 3x3 block centred on (y, x), as the saturation step grows a saturated pixel."""
    image.dq[y - 1:y + 2, x - 1:x + 2] |= SATURATED


def save_images(folder, images):
    """Save each image in ``folder`` under its own file name and return the paths."""
    paths = []
    for image in images:
        path = folder / image.meta.filename
        image.save(path)
        paths.append(str(path))
    return paths


def test_init():
    step = PersistenceFlagStep(time=1000)

    assert step.time == 1000


def test_flags_pixels_saturated_in_earlier_exposures():
    images = datamodels.ModelContainer()

    time0 = Time(60122.0226664904, format="mjd")
    start_times = []
    for i in range(9):
        image = datamodels.ImageModel((10, 10))
        image.dq = np.zeros((10, 10), dtype=np.uint32)
        images.append(image)

        # Populate start times
        start_time = time0.value + i * 0.0132
        start_times.append(start_time)

    # Make the start times not in order.
    start_times = np.roll(start_times, shift=4)

    for i, image in enumerate(images):
        # Populate detector meta
        image.meta.instrument.detector = "NRCALONG"
        image.meta.filename = f"jw01125002001_03101_0000{i}_nrcalong_cal.fits"

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


def test_lone_saturated_pixel_flagged_in_later_exposure():
    """PersistenceFlagStep.call on one lone SATURATED pixel then a later image: the later image is flagged there."""
    first = make_image(1, start_time=60122.0)
    first.dq[5, 5] = SATURATED
    later = make_image(2, start_time=60122.001)

    results = PersistenceFlagStep.call(datamodels.ModelContainer([first, later]))

    assert results[1].dq[5, 5] == PERSISTENCE | DO_NOT_USE


def test_other_detector_not_flagged():
    """PersistenceFlagStep.call on a saturated NRCA1 image and a later NRCA2 image: the NRCA2 dq stays GOOD."""
    first = make_image(1, "NRCA1", 60122.0)
    saturate(first, 2, 2)
    later = make_image(2, "NRCA2", 60122.001)

    results = PersistenceFlagStep.call(datamodels.ModelContainer([first, later]))

    assert results[1].dq[2, 2] == GOOD


def test_input_models_unchanged():
    """PersistenceFlagStep.call on a container: the input models keep their dq; the results are copies."""
    first = make_image(1, start_time=60122.0)
    saturate(first, 2, 2)
    later = make_image(2, start_time=60122.001)
    images = datamodels.ModelContainer([first, later])

    PersistenceFlagStep.call(images)

    assert images[1].dq[2, 2] == GOOD


def test_file_names_saved_with_flags(tmp_path):
    """PersistenceFlagStep.call on a list of file names with save_results: the saved later file has PERSISTENCE set."""
    first = make_image(1, start_time=60122.0)
    saturate(first, 2, 2)
    later = make_image(2, start_time=60122.001)
    paths = save_images(tmp_path, [later, first])
    output_dir = tmp_path / "flagged"

    PersistenceFlagStep.call(paths, save_results=True, suffix="cal", output_dir=str(output_dir))

    assert fits.getdata(output_dir / later.meta.filename, "DQ")[2, 2] == PERSISTENCE | DO_NOT_USE


def test_file_names_return_saved_paths(tmp_path):
    """PersistenceFlagStep.call on two file names with save_results: returns the two saved paths."""
    first = make_image(1, start_time=60122.0)
    later = make_image(2, start_time=60122.001)
    paths = save_images(tmp_path, [first, later])
    output_dir = tmp_path / "flagged"

    written = PersistenceFlagStep.call(paths, save_results=True, suffix="cal", output_dir=str(output_dir))

    assert written == [str(output_dir / first.meta.filename), str(output_dir / later.meta.filename)]


def test_association_members_saved_with_flags(tmp_path):
    """PersistenceFlagStep.call on an association file with save_results: the later member is saved flagged."""
    first = make_image(1, start_time=60122.0)
    saturate(first, 2, 2)
    later = make_image(2, start_time=60122.001)
    save_images(tmp_path, [first, later])
    members = [{"expname": image.meta.filename, "exptype": "science"} for image in (first, later)]
    asn = {"asn_id": "o001", "asn_pool": "pool", "products": [{"name": "product", "members": members}]}
    asn_path = tmp_path / "test_asn.json"
    asn_path.write_text(json.dumps(asn))
    output_dir = tmp_path / "flagged"

    PersistenceFlagStep.call(str(asn_path), save_results=True, suffix="cal", output_dir=str(output_dir))

    assert fits.getdata(output_dir / later.meta.filename, "DQ")[2, 2] == PERSISTENCE | DO_NOT_USE


def test_file_overwritten_in_place(tmp_path):
    """PersistenceFlagStep.call on file names saved back to the same folder and suffix: the file holds the flags."""
    first = make_image(1, start_time=60122.0)
    saturate(first, 2, 2)
    later = make_image(2, start_time=60122.001)
    paths = save_images(tmp_path, [first, later])

    PersistenceFlagStep.call(paths, save_results=True, suffix="cal", output_dir=str(tmp_path))

    assert fits.getdata(paths[1], "DQ")[2, 2] == PERSISTENCE | DO_NOT_USE


def test_save_mask_writes_flagged_pixels(tmp_path):
    """PersistenceFlagStep.call with save_mask on two images: the later image's _persistmask file is 1 at the flag."""
    first = make_image(1, start_time=60122.0)
    saturate(first, 2, 2)
    later = make_image(2, start_time=60122.001)

    PersistenceFlagStep.call(datamodels.ModelContainer([first, later]), save_mask=True, output_dir=str(tmp_path))

    mask = fits.getdata(tmp_path / later.meta.filename.replace("_cal", "_persistmask"))
    assert mask[2, 2] == 1


def test_cal_step_complete_in_fits_header(tmp_path):
    """PersistenceFlagStep.call on two images of one detector: each result saves with S_PERSFL = COMPLETE."""
    images = datamodels.ModelContainer()
    for i in range(2):
        images.append(make_image(i, start_time=60122.0 + i * 0.0132))

    results = PersistenceFlagStep.call(images)
    results[0].save(tmp_path / "result.fits")

    assert results[0].meta.cal_step.persist == "COMPLETE"
    assert fits.getheader(tmp_path / "result.fits")["S_PERSFL"] == "COMPLETE"


def test_cal_step_complete_beside_skipped_jwst_persistence(tmp_path):
    """PersistenceFlagStep.call on an image whose jwst persistence step was skipped: the saved S_PERSFL is COMPLETE."""
    image = make_image(1)
    image.meta.cal_step.persistence = "SKIPPED"

    results = PersistenceFlagStep.call(datamodels.ModelContainer([image]))
    results[0].save(tmp_path / "result.fits")

    assert fits.getheader(tmp_path / "result.fits")["S_PERSFL"] == "COMPLETE"

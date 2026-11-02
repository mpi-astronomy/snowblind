import json
from importlib.resources import files

import asdf
import numpy as np
import pytest
from jwst import datamodels
from astropy.io import fits
from astropy.time import Time

from snowblind import PersistenceFlagStep, persist


SATURATED = datamodels.dqflags.pixel["SATURATED"]
PERSISTENCE = datamodels.dqflags.pixel["PERSISTENCE"]
DO_NOT_USE = datamodels.dqflags.pixel["DO_NOT_USE"]
GOOD = datamodels.dqflags.pixel["GOOD"]
NIRCAM_IMAGING_DETECTORS = [
    "NRCA1", "NRCA2", "NRCA3", "NRCA4", "NRCALONG", "NRCB1", "NRCB2", "NRCB3", "NRCB4", "NRCBLONG"
]


@pytest.fixture(autouse=True)
def packaged_times(tmp_path, monkeypatch):
    """An empty folder of packaged persistence times in place of the package's, so that the step falls back on
    2500 s unless a test writes a time there; returns the folder."""
    folder = tmp_path / "packaged"
    folder.mkdir()
    monkeypatch.setattr(persist, "PACKAGED_DIR", folder)
    return folder


def write_packaged_time(folder, detector, seconds):
    """Write a packaged persistence time of ``seconds`` for ``detector`` into ``folder``."""
    path = folder / persist.PACKAGED_TIME_PATTERN.format(detector=detector.lower())
    asdf.AsdfFile({"detector": detector, "time": seconds}).write_to(path)


def make_image(i, detector="NRCALONG", start_time=60122.0, shape=(10, 10), duration=0.0):
    """An image model of ``shape`` (10x10 by default) with an empty dq, named as exposure ``i`` of ``detector``,
    lasting ``duration`` seconds (0 by default, so that its start, middle and end coincide)."""
    image = datamodels.ImageModel(shape)
    image.dq = np.zeros(shape, dtype=np.uint32)
    image.meta.instrument.detector = detector
    image.meta.filename = f"jw01125002001_03101_0000{i}_{detector.lower()}_cal.fits"
    image.meta.exposure.start_time = start_time
    image.meta.exposure.end_time = start_time + duration / 86400.0
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
    """PersistenceFlagStep.call on nine images 1140 s apart, given out of time order, three of them with a SATURATED
    pixel: the image that starts next is flagged at each of those pixels, and an image that starts earlier is not."""
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
        image.meta.exposure.end_time = start_times[i]

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


def test_exposure_longer_than_time_flags_the_next_one():
    """PersistenceFlagStep.call with time=500 on a 1000 s image saturated at (2, 2) and a 600 s image starting 100 s
    after it ends, so with its middle 400 s after that end: the later image is flagged there."""
    first = make_image(1, start_time=60122.0, duration=1000.0)
    first.dq[2, 2] = SATURATED
    later = make_image(2, start_time=60122.0 + 1100.0 / 86400.0, duration=600.0)

    results = PersistenceFlagStep.call(datamodels.ModelContainer([first, later]), time=500.0)

    assert results[1].dq[2, 2] == PERSISTENCE | DO_NOT_USE


def test_later_exposure_with_its_middle_beyond_time_not_flagged():
    """PersistenceFlagStep.call with time=500 on a 100 s image saturated at (2, 2) and a 1000 s image starting 100 s
    after it ends, so with its middle 600 s after that end: the later image stays GOOD there."""
    first = make_image(1, start_time=60122.0, duration=100.0)
    first.dq[2, 2] = SATURATED
    later = make_image(2, start_time=60122.0 + 200.0 / 86400.0, duration=1000.0)

    results = PersistenceFlagStep.call(datamodels.ModelContainer([first, later]), time=500.0)

    assert results[1].dq[2, 2] == GOOD


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


def test_subarray_flagged_where_earlier_full_frame_saturated():
    """PersistenceFlagStep.call on a 10x10 image saturated at (5, 5) and a later 4x4 subarray starting at detector
    pixel 4, 4 (1-based): the subarray is flagged at (2, 2)."""
    first = make_image(1, start_time=60122.0)
    first.dq[5, 5] = SATURATED
    later = make_image(2, start_time=60122.001, shape=(4, 4))
    later.meta.subarray.ystart = 4
    later.meta.subarray.xstart = 4

    results = PersistenceFlagStep.call(datamodels.ModelContainer([first, later]))

    assert results[1].dq[2, 2] == PERSISTENCE | DO_NOT_USE


def test_full_frame_flagged_where_earlier_subarray_saturated():
    """PersistenceFlagStep.call on a 4x4 subarray starting at detector pixel 4, 4 (1-based) saturated at (0, 0) and a
    later 10x10 image: the 10x10 image is flagged at (3, 3)."""
    first = make_image(1, start_time=60122.0, shape=(4, 4))
    first.meta.subarray.ystart = 4
    first.meta.subarray.xstart = 4
    first.dq[0, 0] = SATURATED
    later = make_image(2, start_time=60122.001)

    results = PersistenceFlagStep.call(datamodels.ModelContainer([first, later]))

    assert results[1].dq[3, 3] == PERSISTENCE | DO_NOT_USE


def test_same_start_time_not_flagged_from_each_other():
    """PersistenceFlagStep.call on two images of one detector with the same start time, the first saturated at
    (2, 2): the second stays GOOD there."""
    first = make_image(1, start_time=60122.0)
    first.dq[2, 2] = SATURATED
    same = make_image(2, start_time=60122.0)

    results = PersistenceFlagStep.call(datamodels.ModelContainer([first, same]))

    assert results[1].dq[2, 2] == GOOD


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


def test_association_background_member_flags_later_science_member(tmp_path):
    """PersistenceFlagStep.call on an association file with save_results, the saturated exposure a background
    member: the later science member is saved flagged."""
    first = make_image(1, start_time=60122.0)
    saturate(first, 2, 2)
    later = make_image(2, start_time=60122.001)
    save_images(tmp_path, [first, later])
    members = [
        {"expname": first.meta.filename, "exptype": "background"},
        {"expname": later.meta.filename, "exptype": "science"},
    ]
    asn = {"asn_id": "o001", "asn_pool": "pool", "products": [{"name": "product", "members": members}]}
    asn_path = tmp_path / "test_asn.json"
    asn_path.write_text(json.dumps(asn))
    output_dir = tmp_path / "flagged"

    PersistenceFlagStep.call(str(asn_path), save_results=True, suffix="cal", output_dir=str(output_dir))

    assert fits.getdata(output_dir / later.meta.filename, "DQ")[2, 2] == PERSISTENCE | DO_NOT_USE


def test_bare_file_names_read_from_input_dir(tmp_path):
    """PersistenceFlagStep.call on bare file names with input_dir set to their folder and save_results: the later
    file is saved flagged."""
    first = make_image(1, start_time=60122.0)
    saturate(first, 2, 2)
    later = make_image(2, start_time=60122.001)
    save_images(tmp_path, [first, later])
    names = [first.meta.filename, later.meta.filename]
    output_dir = tmp_path / "flagged"

    PersistenceFlagStep.call(
        names, input_dir=str(tmp_path), save_results=True, suffix="cal", output_dir=str(output_dir)
    )

    assert fits.getdata(output_dir / later.meta.filename, "DQ")[2, 2] == PERSISTENCE | DO_NOT_USE


def test_file_names_saved_under_their_own_names_without_output_use_model(tmp_path):
    """PersistenceFlagStep.call on two file names with save_results and output_use_model=False: returns the two
    paths named after the exposures."""
    first = make_image(1, start_time=60122.0)
    later = make_image(2, start_time=60122.001)
    paths = save_images(tmp_path, [first, later])
    output_dir = tmp_path / "flagged"

    written = PersistenceFlagStep.call(
        paths, save_results=True, output_use_model=False, suffix="cal", output_dir=str(output_dir)
    )

    assert written == [str(output_dir / first.meta.filename), str(output_dir / later.meta.filename)]


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


def test_packaged_time_reads_the_detectors_file(packaged_times):
    """packaged_time('NRCALONG') with a packaged file holding 1234 s returns 1234.0."""
    write_packaged_time(packaged_times, "NRCALONG", 1234.0)

    assert persist.packaged_time("NRCALONG") == 1234.0


def test_packaged_time_without_a_file_is_none():
    """packaged_time('NRCALONG') with no packaged file for the detector returns None."""
    assert persist.packaged_time("NRCALONG") is None


@pytest.mark.parametrize("detector", NIRCAM_IMAGING_DETECTORS)
def test_package_holds_a_time_for_each_nircam_detector(detector, monkeypatch):
    """packaged_time of each NIRCam imaging detector, from the files packaged with snowblind, returns a time above 0."""
    monkeypatch.setattr(persist, "PACKAGED_DIR", files("snowblind").joinpath("data"))

    assert persist.packaged_time(detector) > 0


def test_packaged_time_flags_a_later_image_within_it(packaged_times):
    """PersistenceFlagStep.call without time, packaged time 500 s, on an image saturated at (2, 2) and one 400 s later:
    the later image is flagged there."""
    write_packaged_time(packaged_times, "NRCALONG", 500.0)
    first = make_image(1, start_time=60122.0)
    first.dq[2, 2] = SATURATED
    later = make_image(2, start_time=60122.0 + 400.0 / 86400.0)

    results = PersistenceFlagStep.call(datamodels.ModelContainer([first, later]))

    assert results[1].dq[2, 2] == PERSISTENCE | DO_NOT_USE


def test_packaged_time_leaves_a_later_image_beyond_it(packaged_times):
    """PersistenceFlagStep.call without time, packaged time 500 s, on an image saturated at (2, 2) and one 600 s later:
    the later image stays GOOD there."""
    write_packaged_time(packaged_times, "NRCALONG", 500.0)
    first = make_image(1, start_time=60122.0)
    first.dq[2, 2] = SATURATED
    later = make_image(2, start_time=60122.0 + 600.0 / 86400.0)

    results = PersistenceFlagStep.call(datamodels.ModelContainer([first, later]))

    assert results[1].dq[2, 2] == GOOD


def test_given_time_overrides_the_packaged_one(packaged_times):
    """PersistenceFlagStep.call with time=1000, packaged time 500 s, on an image saturated at (2, 2) and one 600 s
    later: the later image is flagged there."""
    write_packaged_time(packaged_times, "NRCALONG", 500.0)
    first = make_image(1, start_time=60122.0)
    first.dq[2, 2] = SATURATED
    later = make_image(2, start_time=60122.0 + 600.0 / 86400.0)

    results = PersistenceFlagStep.call(datamodels.ModelContainer([first, later]), time=1000.0)

    assert results[1].dq[2, 2] == PERSISTENCE | DO_NOT_USE


def test_without_a_packaged_time_flags_within_2500_s():
    """PersistenceFlagStep.call without time or a packaged time, on an image saturated at (2, 2) and one 2400 s
    later: the later image is flagged there."""
    first = make_image(1, start_time=60122.0)
    first.dq[2, 2] = SATURATED
    later = make_image(2, start_time=60122.0 + 2400.0 / 86400.0)

    results = PersistenceFlagStep.call(datamodels.ModelContainer([first, later]))

    assert results[1].dq[2, 2] == PERSISTENCE | DO_NOT_USE


def test_without_a_packaged_time_leaves_an_image_beyond_2500_s():
    """PersistenceFlagStep.call without time or a packaged time, on an image saturated at (2, 2) and one 2600 s
    later: the later image stays GOOD there."""
    first = make_image(1, start_time=60122.0)
    first.dq[2, 2] = SATURATED
    later = make_image(2, start_time=60122.0 + 2600.0 / 86400.0)

    results = PersistenceFlagStep.call(datamodels.ModelContainer([first, later]))

    assert results[1].dq[2, 2] == GOOD

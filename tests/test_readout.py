import numpy as np

from snowblind import readout


def detector_pattern(nrows, ncols):
    """Detector-frame image whose value encodes row and column: 1000 * row + col."""
    rows, cols = np.indices((nrows, ncols))
    return (1000 * rows + cols).astype(np.float32)


def test_channel_slices_full_frame():
    """channel_slices(2048) returns four 512-column blocks in order."""
    slices = readout.channel_slices(2048)

    assert slices == [slice(0, 512), slice(512, 1024), slice(1024, 1536), slice(1536, 2048)]


def test_channel_direction_alternates():
    """channel_direction gives +1, -1, +1, -1 for channels 0 to 3 when channel 0 reads forward."""
    directions = [readout.channel_direction(k, first_direction=1) for k in range(4)]

    assert directions == [1, -1, 1, -1]


def test_channel_direction_first_reversed():
    """channel_direction gives -1, +1, -1, +1 when channel 0 reads backward."""
    directions = [readout.channel_direction(k, first_direction=-1) for k in range(4)]

    assert directions == [-1, 1, -1, 1]


def test_clock_index_forward_channel():
    """clock_index for a 3 x 8 forward channel is row * 20 + col."""
    t = readout.clock_index(3, 8, 1)

    rows, cols = np.indices((3, 8))
    np.testing.assert_array_equal(t, rows * 20 + cols)


def test_clock_index_reverse_channel():
    """clock_index for a 3 x 8 reversed channel is row * 20 + (7 - col)."""
    t = readout.clock_index(3, 8, -1)

    rows, cols = np.indices((3, 8))
    np.testing.assert_array_equal(t, rows * 20 + (7 - cols))


def test_stream_length_includes_overhead():
    """stream_length(2048, 512) is 2048 * 524."""
    assert readout.stream_length(2048, 512) == 2048 * 524


def test_to_stream_places_pixels_at_clock_index():
    """to_stream puts each pixel of a reversed channel at its clock_index and zeros elsewhere."""
    image = detector_pattern(3, 8)

    stream = readout.to_stream(image, -1)

    t = readout.clock_index(3, 8, -1)
    np.testing.assert_array_equal(stream[t], image)
    assert stream.size == readout.stream_length(3, 8)
    assert np.count_nonzero(stream) == image.size - 1


def test_stream_round_trip_forward():
    """from_stream(to_stream(image, 1)) returns the image for a forward channel."""
    image = detector_pattern(5, 6)

    back = readout.from_stream(readout.to_stream(image, 1), 5, 6, 1)

    np.testing.assert_array_equal(back, image)


def test_stream_round_trip_reverse():
    """from_stream(to_stream(image, -1)) returns the image for a reversed channel."""
    image = detector_pattern(5, 6)

    back = readout.from_stream(readout.to_stream(image, -1), 5, 6, -1)

    np.testing.assert_array_equal(back, image)


def test_stream_valid_marks_overhead_false():
    """stream_valid(2, 4) is True at the 8 pixel clocks and False at the 24 overhead clocks."""
    valid = readout.stream_valid(2, 4)

    assert valid.sum() == 8
    assert valid.size == 2 * (4 + readout.LINE_OVERHEAD)


def test_frame_round_trip_nircam_a1():
    """to_science_frame(to_detector_frame(x)) is x for fastaxis -1, slowaxis 2."""
    image = detector_pattern(6, 8)

    back = readout.to_science_frame(readout.to_detector_frame(image, -1, 2), -1, 2)

    np.testing.assert_array_equal(back, image)


def test_frame_round_trip_nircam_a2():
    """to_science_frame(to_detector_frame(x)) is x for fastaxis 1, slowaxis -2."""
    image = detector_pattern(6, 8)

    back = readout.to_science_frame(readout.to_detector_frame(image, 1, -2), 1, -2)

    np.testing.assert_array_equal(back, image)


def test_frame_round_trip_niriss():
    """to_science_frame(to_detector_frame(x)) is x for fastaxis -2, slowaxis -1, which transposes."""
    image = detector_pattern(6, 8)

    back = readout.to_science_frame(readout.to_detector_frame(image, -2, -1), -2, -1)

    np.testing.assert_array_equal(back, image)


def test_detector_frame_niriss_has_slow_axis_along_rows():
    """to_detector_frame for fastaxis -2, slowaxis -1 turns a 6 x 8 science image into 8 x 6."""
    image = detector_pattern(6, 8)

    det = readout.to_detector_frame(image, -2, -1)

    assert det.shape == (8, 6)


def stream_of_science_image(science_image, fastaxis, slowaxis, channel, first_direction):
    """Readout stream of one channel of a science-frame image."""
    det = readout.to_detector_frame(science_image, fastaxis, slowaxis)
    columns = readout.channel_slices(det.shape[1])[channel]
    direction = readout.channel_direction(channel, first_direction)
    return readout.to_stream(np.ascontiguousarray(det[:, columns]), direction)


def monotone_within_rows(stream, nrows, width):
    """True if the pixel samples of every row of the stream increase with clock."""
    rows = stream.reshape(nrows, width + readout.LINE_OVERHEAD)[:, :width]
    return bool(np.all(np.diff(rows, axis=1) > 0))


def test_stream_monotone_in_clock_nircam_a1_forward_channel():
    """A detector-frame ramp that increases with clock gives a monotone stream in channel 0 for fastaxis -1, slowaxis 2."""
    det = detector_pattern(8, 16)
    science = readout.to_science_frame(det, -1, 2)

    stream = stream_of_science_image(science, -1, 2, 0, 1)

    assert monotone_within_rows(stream, 8, 4)


def test_stream_monotone_in_clock_nircam_a1_reverse_channel():
    """A detector-frame ramp that decreases with column gives a monotone stream in channel 1 for fastaxis -1, slowaxis 2."""
    det = 1000 * np.indices((8, 16))[0] - np.indices((8, 16))[1]
    science = readout.to_science_frame(det.astype(np.float32), -1, 2)

    stream = stream_of_science_image(science, -1, 2, 1, 1)

    assert monotone_within_rows(stream, 8, 4)


def test_stream_monotone_in_clock_nircam_a2_forward_channel():
    """A detector-frame ramp that increases with clock gives a monotone stream in channel 0 for fastaxis 1, slowaxis -2."""
    det = detector_pattern(8, 16)
    science = readout.to_science_frame(det, 1, -2)

    stream = stream_of_science_image(science, 1, -2, 0, 1)

    assert monotone_within_rows(stream, 8, 4)


def test_stream_monotone_in_clock_niriss_reverse_channel():
    """A detector-frame ramp that decreases with column gives a monotone stream in channel 3 for fastaxis -2, slowaxis -1."""
    det = 1000 * np.indices((8, 16))[0] - np.indices((8, 16))[1]
    science = readout.to_science_frame(det.astype(np.float32), -2, -1)

    stream = stream_of_science_image(science, -2, -1, 3, 1)

    assert monotone_within_rows(stream, 8, 4)


def test_channel_blocks_nircam_a1_reverses_column_order():
    """channel_blocks for fastaxis -1, slowaxis 2 on a 8 x 16 frame puts channel 0 in science columns 12 to 16."""
    blocks = readout.channel_blocks((8, 16), -1, 2)

    assert blocks[0] == (slice(0, 8), slice(12, 16))
    assert blocks[3] == (slice(0, 8), slice(0, 4))


def test_channel_blocks_niriss_are_row_blocks():
    """channel_blocks for fastaxis -2, slowaxis -1 on a 16 x 8 frame are row blocks of 4 rows."""
    blocks = readout.channel_blocks((16, 8), -2, -1)

    assert all(block[1] == slice(0, 8) for block in blocks)
    assert sorted(block[0].start for block in blocks) == [0, 4, 8, 12]


def test_channel_blocks_cover_the_frame_once():
    """channel_blocks for fastaxis 1, slowaxis -2 tile a 8 x 16 frame without overlap."""
    cover = np.zeros((8, 16), dtype=int)

    for rows, cols in readout.channel_blocks((8, 16), 1, -2):
        cover[rows, cols] += 1

    np.testing.assert_array_equal(cover, 1)

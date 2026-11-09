import numpy as np
from jwst.saturation import x_irs2
from stdatamodels.jwst import datamodels

from snowblind.saturation import flag_saturation_neighbors

SATURATED = datamodels.dqflags.group["SATURATED"]
CHARGELOSS = datamodels.dqflags.group["CHARGELOSS"]
DO_NOT_USE = datamodels.dqflags.group["DO_NOT_USE"]


def ramp_with_saturated_pixel():
    """A ramp of 4 groups whose pixel [5, 5] is SATURATED from group 2 on."""
    model = datamodels.RampModel((1, 4, 11, 11))
    model.groupdq[0, 2:, 5, 5] = SATURATED
    return model


def test_flag_saturation_neighbors_flags_the_8_neighbors():
    """flag_saturation_neighbors on a ramp with one saturated pixel sets CHARGELOSS and DO_NOT_USE on its 8 neighbors."""
    result = flag_saturation_neighbors(ramp_with_saturated_pixel())

    neighbors = result.groupdq[0, 2, 4:7, 4:7].ravel().tolist()
    assert neighbors == [CHARGELOSS | DO_NOT_USE] * 4 + [SATURATED] + [CHARGELOSS | DO_NOT_USE] * 4


def test_flag_saturation_neighbors_flags_every_group_from_the_saturation_on():
    """flag_saturation_neighbors on a ramp with a pixel saturated from group 2 flags its neighbor in groups 2 and 3 only."""
    result = flag_saturation_neighbors(ramp_with_saturated_pixel())

    assert result.groupdq[0, :, 5, 6].tolist() == [0, 0, CHARGELOSS | DO_NOT_USE, CHARGELOSS | DO_NOT_USE]


def test_flag_saturation_neighbors_leaves_a_pixel_two_pixels_away():
    """flag_saturation_neighbors on a ramp with one saturated pixel leaves a pixel 2 px from it unflagged."""
    result = flag_saturation_neighbors(ramp_with_saturated_pixel())

    assert (result.groupdq[0, :, 5, 7] == 0).all()


def test_flag_saturation_neighbors_leaves_a_saturated_neighbor_as_saturated():
    """flag_saturation_neighbors on a ramp with two adjacent saturated pixels sets no CHARGELOSS on either."""
    model = ramp_with_saturated_pixel()
    model.groupdq[0, 2:, 5, 6] = SATURATED

    result = flag_saturation_neighbors(model)

    assert result.groupdq[0, 3, 5, 5:7].tolist() == [SATURATED, SATURATED]


def test_flag_saturation_neighbors_reaches_across_interleaved_irs2_reference_rows():
    """flag_saturation_neighbors on an NRS1 IRS2 ramp with normal pixel [7, 100] saturated flags normal pixel [8, 100].

    In IRS2 layout four interleaved reference rows lie between these two rows.
    """
    model = datamodels.RampModel((1, 1, 3200, 2048))
    model.meta.instrument.name = "NIRSPEC"
    model.meta.instrument.detector = "NRS1"
    model.meta.exposure.readpatt = "NRSIRS2RAPID"
    model.meta.exposure.nrs_normal = 16
    model.meta.exposure.nrs_reference = 4
    irs2_mask = x_irs2.make_mask(model)
    saturated = np.zeros((2048, 2048), dtype=np.uint8)
    saturated[7, 100] = SATURATED
    x_irs2.to_irs2(model.groupdq[0, 0], saturated, irs2_mask, "NRS1")

    result = flag_saturation_neighbors(model)

    normal = x_irs2.from_irs2(result.groupdq[0, 0], irs2_mask, "NRS1")
    assert normal[8, 100] == CHARGELOSS | DO_NOT_USE

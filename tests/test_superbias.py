import numpy as np
from stdatamodels.jwst import datamodels

from snowblind.superbias import NOT_ZERO, SuperBiasStep

NY, NX = 4, 4


def set_subarray(model):
    model.meta.subarray.name = "FULL"
    model.meta.subarray.xstart = 1
    model.meta.subarray.ystart = 1
    model.meta.subarray.xsize = NX
    model.meta.subarray.ysize = NY
    model.meta.subarray.fastaxis = -1
    model.meta.subarray.slowaxis = 2
    model.meta.instrument.name = "NIRCAM"
    model.meta.instrument.detector = "NRCA1"


def make_ramp(zeroframe):
    """A ramp of two groups at 1000 DN whose zeroframe holds ``zeroframe`` DN in every pixel."""
    model = datamodels.RampModel((1, 2, NY, NX))
    set_subarray(model)
    model.meta.exposure.zero_frame = True
    model.data[:] = 1000.0
    model.pixeldq = np.zeros((NY, NX), dtype=np.uint32)
    model.groupdq = np.zeros((1, 2, NY, NX), dtype=np.uint8)
    model.zeroframe = np.full((1, NY, NX), zeroframe, dtype=np.float32)
    return model


def save_superbias(tmp_path, value):
    """Save a superbias reference of ``value`` DN in every pixel and return its path."""
    superbias = datamodels.SuperBiasModel((NY, NX))
    set_subarray(superbias)
    superbias.data[:] = value
    superbias.dq = np.zeros((NY, NX), dtype=np.uint32)
    path = tmp_path / "superbias.fits"
    superbias.save(path)
    return str(path)


def test_zeroframe_equal_to_the_superbias_is_not_left_at_zero(tmp_path):
    """SuperBiasStep.call on a zeroframe of 900 DN with a 900 DN superbias returns a zeroframe of NOT_ZERO."""
    result = SuperBiasStep.call(make_ramp(900.0), override_superbias=save_superbias(tmp_path, 900.0))

    assert np.all(result.zeroframe == NOT_ZERO)


def test_zero_zeroframe_stays_zero(tmp_path):
    """SuperBiasStep.call on a zeroframe of 0 DN, the unusable marker, with a 900 DN superbias returns 0 DN."""
    result = SuperBiasStep.call(make_ramp(0.0), override_superbias=save_superbias(tmp_path, 900.0))

    assert np.all(result.zeroframe == 0.0)


def test_zeroframe_has_the_superbias_subtracted(tmp_path):
    """SuperBiasStep.call on a zeroframe of 950 DN with a 900 DN superbias returns 50 DN."""
    result = SuperBiasStep.call(make_ramp(950.0), override_superbias=save_superbias(tmp_path, 900.0))

    assert np.all(result.zeroframe == 50.0)

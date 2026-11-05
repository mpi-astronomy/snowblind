from importlib.metadata import version, PackageNotFoundError

from .snowblind import SnowblindStep
from .jump_plus import JumpPlusStep
from .selfcal import OpenPixelStep
from .persist import PersistenceFlagStep
from .stream_clean import StreamCleanStep
from .reset_rampfit import ResetRampFitStep
from .pipeline import Detector1Pipeline
from .resample import SparseResampleStep
from .outlier_detection import SparseOutlierDetectionStep
from .outlier_reference import TiledOutlierStep
from .image3 import Image3Pipeline


try:
    __version__ = version(__package__ or __name__)
except PackageNotFoundError:
    __version__ = "dev"


__all__ = [
    '__version__',
    'SnowblindStep',
    'JumpPlusStep',
    'OpenPixelStep',
    'PersistenceFlagStep',
    'StreamCleanStep',
    'ResetRampFitStep',
    'Detector1Pipeline',
    'SparseResampleStep',
    'SparseOutlierDetectionStep',
    'TiledOutlierStep',
    'Image3Pipeline',
]


def _get_steps():
    """These steps are provided to stpipe by this package

    Returns
    -------
    list of tuple
        One ``(class_name, class_alias, is_pipeline)`` entry per step and
        pipeline: the fully qualified class name, the ``strun`` alias and
        whether it is a pipeline.
    """
    return [
        ("snowblind.SnowblindStep", 'snowblind', False),
        ("snowblind.JumpPlusStep", 'jump_plus', False),
        ("snowblind.OpenPixelStep", 'open_pixel', False),
        ("snowblind.PersistenceFlagStep", 'persist', False),
        ("snowblind.StreamCleanStep", 'stream_clean', False),
        ("snowblind.ResetRampFitStep", 'reset_rampfit', False),
        ("snowblind.Detector1Pipeline", 'snowblind_detector1', True),
        ("snowblind.SparseResampleStep", 'sparse_resample', False),
        ("snowblind.SparseOutlierDetectionStep", 'sparse_outlier_detection', False),
        ("snowblind.TiledOutlierStep", 'tiled_outlier', False),
        ("snowblind.Image3Pipeline", 'snowblind_image3', True),
    ]

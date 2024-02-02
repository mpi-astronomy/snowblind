from importlib.metadata import version, PackageNotFoundError

from .snowblind import SnowblindStep
from .jump_plus import JumpPlusStep
from .rc_selfcal import RcSelfCalStep
from .persist import PersistenceFlagStep


try:
    __version__ = version(__package__ or __name__)
except PackageNotFoundError:
    __version__ = "dev"


__all__ = [
    '__version__',
    'SnowblindStep',
    'JumpPlusStep',
    'RcSelfCalStep',
    'PersistenceFlagStep',
]


def _get_steps():
    """These steps are provided to stpipe by this package
    """
    return [
        ("snowblind.SnowblindStep", 'snowblind', False),
        ("snowblind.JumpPlusStep", 'jump_plus', False),
        ("snowblind.RcSelfCalStep", 'rc_selfcal', False),
        ("snowblind.PersistenceFlagStep", 'persist', False),
    ]

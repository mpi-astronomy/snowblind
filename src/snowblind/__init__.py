from importlib import metadata

from .snowblind import SnowblindStep
from .jump_plus import JumpPlusStep
from .rc_cross import RcCrossStep


try:
    __version__ = metadata.version(__package__ or __name__)
except Exception:
    __version__ = "dev"


__all__ = ['SnowblindStep', 'JumpPlusStep', 'RcCrossStep', '__version__']


def _get_steps():
    """These steps are provided to stpipe by this package
    """
    return [
        ("snowblind.SnowblindStep", 'snowblind', False),
        ("snowblind.JumpPlusStep", 'jump_plus', False),
        ("snowblind.RcCrossStep", 'rc_cross', False),
    ]

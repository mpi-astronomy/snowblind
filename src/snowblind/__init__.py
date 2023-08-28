from . import _version
from .snowblind import SnowblindStep
from .jump_plus import JumpPlusStep
from .rc_selfcal import RcSelfCalStep
from .persist import PersistenceFlagStep


try:
    __version__ = _version.version
except Exception:
    __version__ = "dev"


__all__ = ['SnowblindStep', 'JumpPlusStep', 'RcSelfCalStep', 'PersistenceFlagStep',
           '__version__']


def _get_steps():
    """These steps are provided to stpipe by this package
    """
    return [
        ("snowblind.SnowblindStep", 'snowblind', False),
        ("snowblind.JumpPlusStep", 'jump_plus', False),
        ("snowblind.RcSelfCalStep", 'rc_selfcal', False),
        ("snowblind.PersistenceFlagStep", 'persist', False),
    ]

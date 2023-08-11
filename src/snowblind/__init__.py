from importlib import metadata

from .snowblind import SnowblindStep
from .jump_plus import JumpPlusStep


try:
    __version__ = metadata.version(__package__ or __name__)
except Exception:
    __version__ = "dev"


__all__ = ['SnowblindStep', 'JumpPlusStep', '__version__']


def _get_steps():
    return [
        ("snowblind.SnowblindStep", 'snowblind', False),
        ("snowblind.JumpPlusStep", 'jump_plus', False),
    ]

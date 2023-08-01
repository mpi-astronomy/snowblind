from importlib import metadata

from .snowblind import SnowblindStep


try:
    __version__ = metadata.version(__package__ or __name__)
except Exception:
    __version__ = "dev"


__all__ = ['SnowblindStep', '__version__']


def _get_steps():
    return [
        ("snowblind.SnowblindStep", 'snowblind', False),
    ]

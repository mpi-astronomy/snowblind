from importlib import metadata

from .snowblind import SnowblindStep


try:
    __version__ = metadata.version(__package__ or __name__)
except:
    __version__ = "dev"



def _get_steps():
    return [
        ("snowblind.SnowblindStep", 'snowblind', False),
    ]
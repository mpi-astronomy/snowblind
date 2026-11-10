"""The per-detector files packaged with snowblind in its data folder.

Each is named ``<kind>_<detector>.asdf``, the detector in lower case.  The kinds are
``persistence`` (`PersistenceFlagStep`), ``snowball_persistence`` (`SnowblindStep`) and
``reset_calib`` (`ResetRampFitStep`); the scripts that write them are named in each step's
documentation.
"""
from importlib.resources import files

import asdf

__all__ = ["packaged_file", "packaged_files", "read_packaged"]

# the data folder; tests point it elsewhere
PACKAGED_DIR = files("snowblind").joinpath("data")


def packaged_file(kind, detector):
    """The packaged file of one kind for a detector.

    Parameters
    ----------
    kind : str
        The kind of file, e.g. ``"persistence"``.
    detector : str
        Detector name as in ``meta.instrument.detector``; case does not matter.

    Returns
    -------
    Traversable or None
        The file; None when the package has none for the detector.
    """
    path = PACKAGED_DIR.joinpath(f"{kind}_{str(detector).lower()}.asdf")
    return path if path.is_file() else None


def packaged_files(kind):
    """Every packaged file of one kind.

    Parameters
    ----------
    kind : str
        The kind of file, e.g. ``"persistence"``.

    Returns
    -------
    dict
        The files, by detector name in upper case.
    """
    prefix, suffix = f"{kind}_", ".asdf"
    return {
        path.name[len(prefix):-len(suffix)].upper(): path
        for path in PACKAGED_DIR.iterdir()
        if path.name.startswith(prefix) and path.name.endswith(suffix)
    }


def read_packaged(path):
    """The tree of a packaged file, read whole so that it outlives the file.

    Parameters
    ----------
    path : Traversable or str
        A packaged file.

    Returns
    -------
    dict
        The tree without asdf's own ``asdf_library`` and ``history`` entries.
    """
    with asdf.open(str(path), lazy_load=False, memmap=False) as handle:
        return {key: value for key, value in handle.tree.items() if key not in ("asdf_library", "history")}

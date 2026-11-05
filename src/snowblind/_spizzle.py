"""spizzle and numba, the optional dependencies of the resampling and outlier steps.

``pip install 'snowblind[spizzle]'`` installs both.  Without them snowblind
still imports: `spizzle` is None, `njit` leaves a function as it is and
`prange` is `range`, and `require` raises with the install hint before
anything would use them.
"""
try:
    import numba
    import spizzle
except ImportError:
    numba = spizzle = None

if numba is None:
    prange = range

    def njit(*args, **kwargs):
        """Leave the function as it is, numba being missing."""
        return lambda function: function
else:
    njit, prange = numba.njit, numba.prange

__all__ = ["spizzle", "njit", "prange", "require"]


def require(user):
    """Raise ImportError unless spizzle and numba are installed.

    Parameters
    ----------
    user : str
        What needs them, for the message.
    """
    if spizzle is None:
        raise ImportError(f"{user} needs spizzle and numba: pip install 'snowblind[spizzle]'")

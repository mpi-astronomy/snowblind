"""Private jwst helpers reused by snowblind.

The underscore-prefixed names, and the ``_fileio`` module, are private to
jwst and may move or change without notice.  They all exist from jwst
2.0.0, where the background helpers moved into their own module, and were
checked against jwst commit e15d3e779 (2026-10-09, "JP-4462: Remove
outdated pipeline names").  If this module fails to import on a newer
jwst, compare it against that commit and update the names here rather than
at the call sites.
"""

JWST_PINNED_COMMIT = "e15d3e779"

try:
    from jwst.clean_flicker_noise.clean_flicker_noise import (  # noqa: F401
        _make_processed_rate_image as make_processed_rate_image,
        _make_scene_mask as make_scene_mask,
        _mask_unusable as mask_unusable,
        make_intermediate_model,
        median_clean,
        read_flat_file,
    )
    from jwst.clean_flicker_noise.background_level import (  # noqa: F401
        background_level,
    )
    from jwst.outlier_detection._fileio import save_blot  # noqa: F401
    from jwst.outlier_detection.utils import (  # noqa: F401
        _flag_resampled_model_crs as flag_resampled_model_crs_with_blot,
    )
except ImportError as err:
    raise ImportError(
        "snowblind reuses private helpers from jwst.clean_flicker_noise and jwst.outlier_detection that "
        f"exist from jwst 2.0.0 and were checked against jwst commit {JWST_PINNED_COMMIT}; "
        f"the installed jwst does not provide them: {err}"
    ) from err

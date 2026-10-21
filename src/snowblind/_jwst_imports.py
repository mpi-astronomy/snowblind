"""Private jwst helpers reused by snowblind.

The underscore-prefixed names are private to jwst and may move or change
without notice.  They were checked against jwst commit e15d3e779
(2026-10-09, "JP-4462: Remove outdated pipeline names").  If this module
fails to import, compare the installed jwst against that commit and update
the names here rather than at the call sites.
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
        clip_to_background,
    )
except ImportError as err:
    raise ImportError(
        "snowblind reuses private helpers from jwst.clean_flicker_noise that "
        f"were checked against jwst commit {JWST_PINNED_COMMIT}; the installed "
        f"jwst no longer provides them: {err}"
    ) from err

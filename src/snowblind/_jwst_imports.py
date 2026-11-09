"""Private jwst helpers reused by snowblind.

The underscore-prefixed names, and the ``_fileio`` module, are private to
jwst and may move or change without notice.  They all exist from jwst
2.0.0 and were checked against jwst commit e15d3e779 (2026-10-09, "JP-4462:
Remove outdated pipeline names").  If this module fails to import on a
newer jwst, compare it against that commit and update the names here rather
than at the call sites.

`snowblind.resample` builds on private parts of stcal's
``stcal.resample.Resample`` too, checked against stcal
`STCAL_CHECKED_VERSION`; the methods it calls or replaces are checked here
at import, and its attributes by the tests that compare it with jwst's
resampling.
"""

JWST_PINNED_COMMIT = "e15d3e779"
STCAL_CHECKED_VERSION = "1.20.0"

try:
    from jwst.clean_flicker_noise.clean_flicker_noise import (  # noqa: F401
        _make_processed_rate_image as make_processed_rate_image,
        _make_scene_mask as make_scene_mask,
        _mask_unusable as mask_unusable,
        make_intermediate_model,
        median_clean,
        read_flat_file,
    )
    from jwst.outlier_detection._fileio import save_blot  # noqa: F401
    from jwst.outlier_detection.outlier_detection_step import OutlierDetectionStep
    from jwst.outlier_detection.utils import (  # noqa: F401
        _flag_resampled_model_crs as flag_resampled_model_crs_with_blot,
    )

    # jwst's outlier step's mode (from the pipeline's ``mode`` attribute or the exposure type) and the
    # association ID of its output names, as functions of a step
    guess_mode = OutlierDetectionStep._guess_mode
    add_asn_id = OutlierDetectionStep._get_asn_id
except (ImportError, AttributeError) as err:
    raise ImportError(
        "snowblind reuses private helpers from jwst.clean_flicker_noise and jwst.outlier_detection that "
        f"exist from jwst 2.0.0 and were checked against jwst commit {JWST_PINNED_COMMIT}; "
        f"the installed jwst does not provide them: {err}"
    ) from err

from stcal.resample import Resample  # noqa: E402

# the methods of stcal's Resample that snowblind.resample calls or replaces
STCAL_RESAMPLE_METHODS = (
    "reset_arrays", "add_model", "finalize", "create_output_model", "is_finalized", "validate_input_model",
    "init_time_counters", "update_time", "finalize_time_info", "_compute_pixel_scale_ratio", "_get_intensity_scale",
    "_check_var_array",
)
_missing = [name for name in STCAL_RESAMPLE_METHODS if not hasattr(Resample, name)]
if _missing:
    raise ImportError(
        f"snowblind.resample builds on stcal's Resample as of stcal {STCAL_CHECKED_VERSION}; the installed stcal "
        f"lacks {', '.join(_missing)}"
    )

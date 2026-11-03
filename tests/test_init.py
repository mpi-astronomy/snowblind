import snowblind


def test_get_steps_lists_the_steps_and_the_pipeline():
    """snowblind._get_steps lists the nine step aliases and the two pipeline aliases in order."""
    aliases = [alias for _, alias, _ in snowblind._get_steps()]

    assert aliases == [
        "snowblind", "jump_plus", "open_pixel", "persist", "stream_clean", "reset_rampfit", "snowblind_detector1",
        "sparse_resample", "sparse_outlier_detection", "tiled_outlier", "snowblind_image3",
    ]

import subprocess
import sys

import snowblind


def test_get_steps_lists_the_steps_and_the_pipeline():
    """snowblind._get_steps lists the nine step aliases and the two pipeline aliases in order."""
    aliases = [alias for _, alias, _ in snowblind._get_steps()]

    assert aliases == [
        "snowblind", "jump_plus", "open_pixel", "persist", "stream_clean", "reset_rampfit", "snowblind_detector1",
        "sparse_resample", "sparse_outlier_detection", "tiled_outlier", "snowblind_image3",
    ]


def test_snowblind_imports_without_spizzle_and_numba():
    """import snowblind in a Python where spizzle and numba cannot be imported succeeds: they are an optional extra."""
    code = "import sys; sys.modules['spizzle'] = None; sys.modules['numba'] = None; import snowblind"

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr

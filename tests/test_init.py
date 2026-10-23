import snowblind


def test_get_steps_lists_all_six_steps():
    """snowblind._get_steps lists the six step aliases in order."""
    aliases = [alias for _, alias, _ in snowblind._get_steps()]

    assert aliases == ["snowblind", "jump_plus", "open_pixel", "persist", "stream_clean", "reset_rampfit"]

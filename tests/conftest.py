import os
from pathlib import Path

import pytest
from stpipe import crds_client
from stpipe.config_parser import ConfigObj

from snowblind.step import Step


@pytest.fixture
def tmp_cwd(tmp_path):
    """Perform test in a pristine temporary working directory."""
    old_dir = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        os.chdir(old_dir)


@pytest.fixture(autouse=True)
def no_crds_server(monkeypatch):
    """Keep the steps from contacting CRDS, so the tests run without a CRDS server.

    Step.call asks CRDS for the step parameter reference, and a jwst step
    that used any reference file (an override counts) asks for the CRDS
    context when it finalizes its result; both are stubbed.  Reference
    files themselves must be given as overrides by the tests.
    """

    def empty_config(cls, dataset, disable=None, crds_observatory=None):
        return ConfigObj()

    for klass in Step.__mro__:
        if "get_config_from_reference" in vars(klass):
            monkeypatch.setattr(klass, "get_config_from_reference", classmethod(empty_config))
    monkeypatch.setattr(crds_client, "get_context_used", lambda observatory: "jwst_0000.pmap")

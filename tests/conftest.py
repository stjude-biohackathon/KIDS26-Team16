"""Fixtures shared by every test."""
import pytest


@pytest.fixture(autouse=True)
def live_results_dir(tmp_path, monkeypatch):
    """-> the folder live evaluations save into during a test.

    Every "Analyze & Grade Note" writes its patient's results file; pointed here,
    a test run never adds to (or overwrites) the real results/live.
    """
    import dashboard.server as server

    target = tmp_path / "live_results"
    monkeypatch.setattr(server, "LIVE_RESULTS_DIR", str(target))
    return target

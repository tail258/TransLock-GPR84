"""核对云端队列拒绝不完整回读，超时只清理自身进程。"""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def batch():
    source = Path(__file__).parents[1] / "scripts/run_prepared_md_batch.py"
    spec = importlib.util.spec_from_file_location("prepared_md_batch_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "changes", [{"trajectory_frames": 499}, {"time_ps": 4990}, {"geometry_failed_frames": 1}]
)
def test_incomplete_readback_is_rejected(batch, changes):
    row = dict(status="passed", trajectory_frames=500, time_ps=5000, geometry_failed_frames=0)
    row.update(changes)
    with pytest.raises(RuntimeError):
        batch.check_readback(row, 5000)


def test_timeout_terminates_only_the_child_and_keeps_log(batch, monkeypatch, tmp_path):
    class Owned:
        terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

    child = Owned()
    monkeypatch.setattr(batch.subprocess, "Popen", lambda *a, **k: child)
    monkeypatch.setattr(batch.time, "monotonic", lambda: 10)
    log = tmp_path / "failure.log"
    with pytest.raises(TimeoutError):
        batch.run_child(["owned"], log, {}, 9, lambda: None)
    assert child.terminated and log.exists()


def test_simulation_failure_does_not_appear_as_success(batch, monkeypatch, tmp_path):
    class Failed:
        returncode = 7

        def poll(self):
            return self.returncode

    monkeypatch.setattr(batch.subprocess, "Popen", lambda *a, **k: Failed())
    with pytest.raises(RuntimeError, match="退出码7"):
        batch.run_child(["owned"], tmp_path / "failure.log", {}, 99, lambda: None)

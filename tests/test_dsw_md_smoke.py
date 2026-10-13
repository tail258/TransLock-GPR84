"""验证云端小样的限时清理、失败留证及回读闸门，不执行真实MD。"""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def smoke(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "scripts/launch_dsw_md_smoke.py"
    spec = importlib.util.spec_from_file_location("dsw_smoke_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "__file__", str(tmp_path / "scripts/launcher.py"))
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: "test GPU")
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "test-version")
    monkeypatch.setattr(module.time, "sleep", lambda duration: None)
    return module, tmp_path


def admission(root):
    return json.loads(next(root.glob("runs/*/cloud_admission.json")).read_text())


def finished_process(module, calls, *, exit_code=0, frames=5):
    def launch(command, **kwargs):
        calls.append(Path(command[1]).name)
        output = Path(command[command.index("--output") + 1])
        output.mkdir(parents=True, exist_ok=True)
        if calls[-1] == "run_md_replicate.py":
            (output / "manifest.json").write_text(json.dumps({"wall_seconds": 60}))
        else:
            (output / "independent_readback.json").write_text(
                json.dumps({"status": "passed", "trajectory_frames": frames})
            )

        class Finished:
            returncode = exit_code

            def poll(self):
                return self.returncode

        return Finished()

    return launch


def test_nonzero_simulation_preserves_failure_and_does_not_review(smoke, monkeypatch):
    module, root = smoke
    calls = []
    monkeypatch.setattr(module.subprocess, "Popen", finished_process(module, calls, exit_code=1))
    with pytest.raises(RuntimeError):
        module.main()
    assert calls == ["run_md_replicate.py"]
    assert admission(root)["status"] == "failed"
    assert next(root.glob("runs/*/simulation.log")).exists()


def test_timeout_terminates_only_owned_child_and_preserves_failure(smoke, monkeypatch):
    module, root = smoke
    ticks = iter([0, 0, 901])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))

    class Hung:
        terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout):
            assert timeout == 10
            return -15

    child = Hung()
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: child)
    with pytest.raises(TimeoutError):
        module.main()
    assert child.terminated
    assert admission(root)["status"] == "failed"
    assert admission(root)["error_type"] == "TimeoutError"


def test_incomplete_readback_cannot_be_marked_passed(smoke, monkeypatch):
    module, root = smoke
    monkeypatch.setattr(module.subprocess, "Popen", finished_process(module, [], frames=4))
    with pytest.raises(AssertionError):
        module.main()
    assert admission(root)["status"] == "failed"


def test_complete_smoke_keeps_formal_replica_scope_false(smoke, monkeypatch):
    module, root = smoke
    monkeypatch.setattr(module.subprocess, "Popen", finished_process(module, []))
    module.main()
    result = admission(root)
    assert result["status"] == "passed_cloud_50ps_and_readback"
    assert result["independent_formal_replica"] is False
    assert result["estimated_ns_per_day"] == pytest.approx(72)

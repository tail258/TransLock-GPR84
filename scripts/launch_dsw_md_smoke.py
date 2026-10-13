"""在已参数化体系上执行有进度的GLPG50ps云端小样，严格限时并独立回读。"""

import importlib.metadata
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def main():
    bundle = Path(__file__).resolve().parents[1]
    os.chdir(bundle)
    directory = (
        bundle / "runs" / ("GLPG1205_cloud50ps_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    )
    directory.mkdir(parents=True, exist_ok=False)
    output = directory / "simulation"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(bundle / "src")
    env["OPENMM_CPU_THREADS"] = "6"
    started = time.monotonic()
    admission = dict(
        status="starting",
        target_time_ps=50,
        base_seed=20261998,
        started_utc=datetime.now(UTC).isoformat(),
        independent_formal_replica=False,
        maximum_each_process_seconds=900,
        cu_per_instance_hour=6.991,
    )

    def save():
        temporary = directory / "cloud_admission.json.tmp"
        temporary.write_text(
            json.dumps(admission, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(directory / "cloud_admission.json")

    save()

    def run(script, arguments, label):
        began = time.monotonic()
        with (directory / (label + ".log")).open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, str(bundle / "scripts" / script), *arguments],
                cwd=bundle,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                previous = None
                while process.poll() is None:
                    if time.monotonic() - began > 900:
                        raise TimeoutError(f"{label}超出15分钟小样上限")
                    row = None
                    try:
                        row = json.loads((output / "manifest.json").read_text())
                    except (FileNotFoundError, json.JSONDecodeError):
                        pass
                    progress = row.get("time_ps", 0) if row else 0
                    if label == "simulation" and progress != previous:
                        percentage = min(100, 100 * progress / 50)
                        fill = int(percentage / 5)
                        bar = "#" * fill + "-" * (20 - fill)
                        print(
                            f"GLPG1205 [{bar}] {percentage:5.1f}%  {progress:.0f}/50 ps", flush=True
                        )
                        previous = progress
                    time.sleep(2)
                if process.returncode != 0:
                    raise RuntimeError(f"{label}失败，退出码{process.returncode}；见{label}.log")
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()

    try:
        admission["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
            timeout=20,
        ).strip()
        admission["software"] = {
            p: importlib.metadata.version(p)
            for p in (
                "openmm",
                "openmm-cuda-12",
                "numpy",
                "rdkit",
                "nvidia-cuda-nvrtc-cu12",
                "nvidia-cuda-runtime-cu12",
                "nvidia-cufft-cu12",
            )
        }
        admission["status"] = "running_cloud_smoke"
        save()
        print(admission["gpu"], flush=True)
        warmup = "inputs/GLPG1205/warmup"
        run(
            "run_md_replicate.py",
            [
                "--root",
                str(bundle),
                "--warmup",
                warmup,
                "--restart",
                warmup + "/restart_v1",
                "--output",
                str(output),
                "--seed",
                "20261998",
                "--time-ps",
                "50",
            ],
            "simulation",
        )
        print("GLPG1205 [####################] 100.0%  50/50 ps；正在回读全部帧", flush=True)
        run(
            "review_md_replicate.py",
            ["--run", str(output), "--output", str(output / "analysis_v1")],
            "readback",
        )
        result = json.loads((output / "manifest.json").read_text())
        readback = json.loads((output / "analysis_v1/independent_readback.json").read_text())
        assert readback["status"] == "passed" and readback["trajectory_frames"] == 5
        admission.update(
            status="passed_cloud_50ps_and_readback",
            readback=readback,
            estimated_ns_per_day=0.05 / result["wall_seconds"] * 86400,
            wall_seconds=time.monotonic() - started,
            completed_utc=datetime.now(UTC).isoformat(),
        )
        print(
            json.dumps(
                {k: admission[k] for k in ("status", "estimated_ns_per_day", "wall_seconds")}
            ),
            flush=True,
        )
    except Exception as error:
        admission.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            finished_utc=datetime.now(UTC).isoformat(),
        )
        raise
    finally:
        save()
        print("小样进程已结束；DSW实例仍运行，请在控制台停止以免继续消耗额度。", flush=True)


if __name__ == "__main__":
    main()

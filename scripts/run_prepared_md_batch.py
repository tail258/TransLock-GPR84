"""复用已验收末态和云端短测，运行固定重复队列并保存实时进度。"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def check_readback(row, time_ps):
    if (
        row["status"] != "passed"
        or row["trajectory_frames"] != time_ps // 10
        or abs(row["time_ps"] - time_ps) > 1e-4
        or row["geometry_failed_frames"] != 0
    ):
        raise RuntimeError("重复轨迹的完整帧、时间或几何回读未通过")


def run_child(command, log, env, deadline, progress):
    with log.open("x", encoding="utf-8") as handle:
        child = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError("队列或当前步骤超过固定运行时间上限")
                progress()
                time.sleep(10)
            if child.returncode:
                raise RuntimeError(f"步骤退出码{child.returncode}，见{log}")
            progress()
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "environment", "admission", "plan", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    os.chdir(bundle)
    admission = read(args.admission)
    if admission["status"] not in {
        "passed_cloud_50ps_and_readback",
        "passed_local_50ps_and_readback",
    }:
        raise RuntimeError("云端50ps与独立回读闸门未通过")
    check_readback(admission["readback"], 50)
    plan = read(args.plan)
    duration = plan["time_ps"]
    if duration <= 0 or duration % 10 or not plan["jobs"]:
        raise ValueError("固定队列长度或任务为空")
    pairs = [(job["ligand_id"], job["seed"]) for job in plan["jobs"]]
    if len(pairs) != len(set(pairs)):
        raise ValueError("队列包含重复对象和种子")
    for job in plan["jobs"]:
        warmup = bundle / job["warmup"]
        if read(warmup / "manifest.json")["status"] != "complete_1ns_technical_sample":
            raise RuntimeError("准备末态未完成")
        if read(warmup / "analysis_v1/independent_readback.json")["status"] != "passed":
            raise RuntimeError("准备末态回读未通过")
        if read(warmup / "restart_v1/manifest.json")["status"] != "restart_checkpoint_passed":
            raise RuntimeError("迁移末态闸门未通过")
    args.output.mkdir(parents=True, exist_ok=False)
    evidence = args.output / "runtime_evidence"
    evidence.mkdir()
    for name in ("pyproject.toml", "uv.lock", "original_pins.json"):
        shutil.copyfile(args.environment / name, evidence / name)
    shutil.copyfile(__file__, evidence / "batch_source.py")
    shutil.copyfile(args.admission, evidence / "cloud_admission.json")
    shutil.copyfile(args.plan, args.output / "batch_plan.json")
    env = os.environ.copy()
    env.update(
        PYTHONUTF8="1",
        PYTHONPATH=str(bundle / "src"),
        OPENMM_CPU_THREADS=str(plan.get("cpu_threads_per_worker", 6)),
    )
    started = time.monotonic()
    deadline = started + plan["maximum_batch_seconds"]
    state = dict(
        status="running",
        started_utc=datetime.now(UTC).isoformat(),
        pid=os.getpid(),
        plan=plan,
        completed=[],
        verified_ns=0,
        integrated_ns=0,
        scope="固定重复探索；共享准备末态；不证明活性或收敛",
        instance_stop_requested=False,
    )
    state_path = args.output / "batch_manifest.json"
    save(state_path, state)
    try:
        for job in plan["jobs"]:
            target = args.output / "replicates" / job["ligand_id"] / str(job["seed"])
            target.parent.mkdir(parents=True, exist_ok=True)
            warmup = bundle / job["warmup"]
            state.update(current_job=job, phase="simulation", current_time_ps=0)
            last_printed = -1

            def progress(target=target, job=job):
                nonlocal last_printed
                try:
                    row = read(target / "manifest.json")
                    elapsed = float(row.get("time_ps", 0))
                except (FileNotFoundError, json.JSONDecodeError):
                    elapsed = 0
                state.update(
                    current_time_ps=elapsed,
                    integrated_ns=state["verified_ns"] + elapsed / 1000,
                    updated_utc=datetime.now(UTC).isoformat(),
                    wall_seconds=time.monotonic() - started,
                )
                save(state_path, state)
                block = int(elapsed / 100)
                if block != last_printed:
                    last_printed = block
                    print(
                        f"{job['ligand_id']} {job['seed']} | {elapsed:.0f}/{duration} ps | "
                        f"累计积分{state['integrated_ns']:.2f}ns；已验收{len(state['completed'])}条",
                        flush=True,
                    )

            command = [sys.executable, "-u", str(bundle / "scripts/run_md_replicate.py")]
            values = dict(
                root=bundle,
                warmup=warmup,
                restart=warmup / "restart_v1",
                output=target,
                seed=job["seed"],
                **{"time-ps": duration},
            )
            for key, value in values.items():
                command.extend(["--" + key, str(value)])
            run_child(
                command,
                target.parent / f"{job['seed']}.log",
                env,
                min(deadline, time.monotonic() + plan["maximum_replica_seconds"]),
                progress,
            )
            state["phase"] = "readback"
            run_child(
                [
                    sys.executable,
                    "-u",
                    str(bundle / "scripts/review_md_replicate.py"),
                    "--run",
                    str(target),
                    "--output",
                    str(target / "analysis_v1"),
                ],
                target.parent / f"{job['seed']}_review.log",
                env,
                min(deadline, time.monotonic() + 900),
                progress,
            )
            check_readback(read(target / "analysis_v1/independent_readback.json"), duration)
            state["completed"].append(dict(**job, output=str(target)))
            state["verified_ns"] = len(state["completed"]) * duration / 1000
            save(state_path, state)
        state.update(status="complete_fixed_queue", finished_utc=datetime.now(UTC).isoformat())
        print(
            f"QUEUE_COMPLETE | {len(state['completed'])}条 | {state['verified_ns']}ns", flush=True
        )
    except Exception as error:
        state.update(
            status="stopped_on_failure",
            error_type=type(error).__name__,
            error=str(error),
            failed_utc=datetime.now(UTC).isoformat(),
        )
        raise
    finally:
        save(state_path, state)


if __name__ == "__main__":
    main()

"""串行推进两候选小样、CUDA短验收和三对象固定三重复，失败保留并停止。"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def save(path, data):
    # 先写完整临时文件再替换本任务自己的状态文件，便于并行只读查询。
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "repo", "output", "control", "review-environment"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    state_path = args.output / "batch_manifest.json"
    assert not state_path.exists()
    scripts = args.repo / "scripts"
    candidates = ["CHEMBL120691", "CHEMBL1086046"]
    objects = [{"id": "GLPG1205", "warmup": str(args.control)}] + [
        {"id": c, "warmup": str(args.output / c / "warmup")} for c in candidates
    ]
    plan = dict(
        objects=objects,
        seeds=[20261011, 20261012, 20261013],
        time_ps=5000,
        job_order=[
            f"{obj['id']}_{seed}" for seed in (20261011, 20261012, 20261013) for obj in objects
        ],
        frozen_utc=datetime.now(UTC).isoformat(),
        total_exploration_ns=45,
        minimum_free_GiB=20,
        maximum_gpu_temperature_C=85,
        warmup_time_ps=1000,
        scope="每对象共享小样末态；目录中性图为条件；失败不换种子掩盖",
    )
    save(args.output / "batch_plan.json", plan)
    shutil.copyfile(__file__, args.output / "batch_source_snapshot.py")
    state = dict(
        status="running",
        plan=plan,
        started_utc=datetime.now(UTC).isoformat(),
        completed=[],
        current="candidate_warmup_gates",
    )
    save(state_path, state)

    def run(script, values, log, review=False):
        state["current"] = f"{script}: {log.parent.name}/{log.name}"
        state["updated_utc"] = datetime.now(UTC).isoformat()
        save(state_path, state)
        env = os.environ.copy()
        if review:
            env["UV_PROJECT_ENVIRONMENT"] = str(args.review_environment)
            command = [
                "uv",
                "run",
                "--project",
                str(args.repo),
                "--frozen",
                "--no-sync",
                "python",
                "-u",
                str(scripts / script),
            ]
        else:
            command = [sys.executable, "-u", str(scripts / script)]
        command += [str(v) for v in values]
        with log.open("x", encoding="utf-8") as output:
            subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
        state["completed"].append(
            dict(script=script, log=str(log), completed_utc=datetime.now(UTC).isoformat())
        )
        save(state_path, state)

    try:
        for candidate in candidates:
            base = args.output / candidate
            warmup = base / "warmup"
            assert (
                json.loads((base / "parameters/manifest.json").read_text())["status"]
                == "ligand_parameters_passed"
            )
            if warmup.exists():
                # 等待首项已启动的进程结束，随后继续队列。
                start = time.monotonic()
                while True:
                    try:
                        wm = json.loads((warmup / "manifest.json").read_text())
                    except json.JSONDecodeError:
                        time.sleep(1)
                        continue
                    if wm["status"] in {"complete_1ns_technical_sample", "failed_complete_system"}:
                        break
                    if time.monotonic() - start > 5400:
                        raise TimeoutError("已有小样超过90分钟未完成，停止队列")
                    time.sleep(10)
                assert wm["status"] == "complete_1ns_technical_sample", wm
            else:
                run(
                    "run_membrane_control.py",
                    [
                        "--root",
                        args.root,
                        "--construct",
                        base / "construct",
                        "--parameters",
                        base / "parameters",
                        "--config",
                        base / "config.yaml",
                        "--output",
                        warmup,
                    ],
                    base / "warmup.log",
                )
            run(
                "review_membrane_control.py",
                ["--run", warmup, "--output", warmup / "analysis_v1"],
                base / "review.log",
                review=True,
            )
            run(
                "export_md_restart.py",
                ["--run", warmup, "--output", warmup / "restart_v1"],
                base / "restart.log",
            )
        smoke = args.output / "cuda_replica_smoke_10ps"
        run(
            "run_md_replicate.py",
            [
                "--root",
                args.root,
                "--warmup",
                args.control,
                "--restart",
                args.control / "restart_v1",
                "--seed",
                20261999,
                "--time-ps",
                10,
                "--output",
                smoke,
            ],
            args.output / "smoke.log",
        )
        run(
            "review_md_replicate.py",
            ["--run", smoke, "--output", smoke / "analysis_v1"],
            args.output / "smoke_review.log",
            review=True,
        )
        result_rows = []
        for seed in plan["seeds"]:
            for obj in objects:
                warmup = Path(obj["warmup"])
                target = args.output / "replicates" / obj["id"] / str(seed)
                target.parent.mkdir(parents=True, exist_ok=True)
                run(
                    "run_md_replicate.py",
                    [
                        "--root",
                        args.root,
                        "--warmup",
                        warmup,
                        "--restart",
                        warmup / "restart_v1",
                        "--seed",
                        seed,
                        "--time-ps",
                        5000,
                        "--output",
                        target,
                    ],
                    target.parent / f"{seed}.log",
                )
                run(
                    "review_md_replicate.py",
                    ["--run", target, "--output", target / "analysis_v1"],
                    target.parent / f"{seed}_review.log",
                    review=True,
                )
                result_rows.append(
                    json.loads((target / "analysis_v1/independent_readback.json").read_text())
                )
                save(args.output / "replicate_results.json", result_rows)
        state.update(
            status="complete_9_replicates",
            completed_exploration_ns=45,
            finished_utc=datetime.now(UTC).isoformat(),
        )
        save(state_path, state)
        print(json.dumps({"status": state["status"], "ns": 45}), flush=True)
    except Exception as error:
        state.update(
            status="stopped_on_failure",
            error_type=type(error).__name__,
            error=str(error),
            failed_utc=datetime.now(UTC).isoformat(),
        )
        save(state_path, state)
        raise


if __name__ == "__main__":
    main()

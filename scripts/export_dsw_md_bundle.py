"""导出已验收膜体系的可迁移运行包；不复制历史轨迹、权重或虚拟环境。"""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    plan = read(args.batch / "batch_plan.json")
    args.output.mkdir(parents=True, exist_ok=False)
    inventory = []

    def copy(source, destination):
        target = args.output / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        inventory.append(
            {"source": str(source), "destination": str(destination), "bytes": target.stat().st_size}
        )

    for name in (
        "run_md_replicate.py",
        "run_membrane_control.py",
        "review_md_replicate.py",
        "review_membrane_control.py",
        "launch_dsw_md_smoke.py",
        "export_dsw_md_bundle.py",
    ):
        copy(repo / "scripts" / name, Path("scripts") / name)
    for name in ("md_pilot.py", "md_diagnostics.py", "md_restart.py", "dcd_reader.py"):
        copy(repo / "src/translock_screening" / name, Path("src/translock_screening") / name)
    package = args.output / "src/translock_screening/__init__.py"
    package.write_text('"""可迁移MD运行时的选取副本。"""\n', encoding="utf-8")
    for name in ("pyproject.toml", "uv.lock"):
        copy(repo / "environments/membrane_md" / name, Path("evidence/original_environment") / name)
    objects = []
    for obj in plan["objects"]:
        source = Path(obj["warmup"])
        wm = read(source / "manifest.json")
        assert wm["status"] == "complete_1ns_technical_sample"
        assert read(source / "analysis_v1/independent_readback.json")["status"] == "passed"
        assert read(source / "restart_v1/manifest.json")["status"] == "restart_checkpoint_passed"
        base = Path("inputs") / obj["id"]
        for name in ("manifest.json", "atom_map.json", "analysis_v1/independent_readback.json"):
            copy(source / name, base / "warmup" / name)
        for name in ("manifest.json", "state.xml", "system.xml", "system.pdb"):
            copy(source / "restart_v1" / name, base / "warmup/restart_v1" / name)
        for name in ("residue_map.json", "retained_oriented_heavy.pdb"):
            copy(Path(wm["construct"]) / name, base / "construct" / name)
        for name in ("manifest.json", "ligand.sdf", "atom_parameters.json"):
            copy(Path(wm["parameters"]) / name, base / "parameters" / name)
        # 仅迁移副本的运行指针；原始清单独立保留作为来源证据。
        copy(source / "manifest.json", base / "evidence/source_warmup_manifest.json")
        wm["source_paths"] = {"construct": wm["construct"], "parameters": wm["parameters"]}
        wm["construct"] = (base / "construct").as_posix()
        wm["parameters"] = (base / "parameters").as_posix()
        save(args.output / base / "warmup/manifest.json", wm)
        objects.append({"id": obj["id"], "warmup": (base / "warmup").as_posix()})
    env = args.output / "environments/membrane_md"
    env.mkdir(parents=True)
    # R550支持CUDA12.4；限制NVRTC版本，避免新PTX要求更高驱动。只安装运行所需子集。
    (env / "pyproject.toml").write_text(
        '[project]\nname = "translock-dsw-md-runtime"\nversion = "0.1.0"\n'
        'requires-python = ">=3.11,<3.12"\ndependencies = [\n'
        ' "numpy==2.2.6", "pandas==2.3.3", "pyyaml==6.0.3",\n'
        ' "openmm[cuda12]==8.6.0", "rdkit==2026.3.5", "matplotlib==3.11.2",\n'
        ' "nvidia-cuda-nvrtc-cu12==12.4.127", "nvidia-cuda-runtime-cu12==12.4.127",\n'
        ' "nvidia-cuda-nvcc-cu12==12.4.131", "nvidia-cuda-cupti-cu12==12.4.127",\n'
        ' "nvidia-nvjitlink-cu12==12.4.127", "nvidia-cufft-cu12==11.2.1.3",\n]\n'
        "\n[tool.uv]\npackage = false\nenvironments = [\"sys_platform == 'linux'\"]\n",
        encoding="utf-8",
    )
    save(
        args.output / "bundle_manifest.json",
        dict(
            status="exported_pending_local_readback_and_cloud_cuda",
            created_utc=datetime.now(UTC).isoformat(),
            objects=objects,
            original_plan=plan,
            files=inventory,
            source_batch=str(args.batch),
            scope="只迁移已验收System/State；不重新参数化；云端CUDA尚未验收；不自动启动45ns队列",
        ),
    )
    (args.output / "README.md").write_text(
        "# TransLock-GPR84 DSW最小运行包\n\n"
        "本包含GLPG1205与两候选已验收末态、序列映射、参数证据及选取代码。"
        "不含原始数据库、模型权重、历史轨迹、账号或认证信息。\n\n"
        "在此目录运行，使用Python3.11与uv；项目路径为`environments/membrane_md`。"
        "首次只执行GLPG1205独立50ps小样与全部帧回读，不计入9条正式重复。\n\n"
        "```bash\nuv sync --project environments/membrane_md --frozen --python 3.11\n"
        "uv run --project environments/membrane_md --frozen --no-sync "
        "python scripts/launch_dsw_md_smoke.py\n```\n\n"
        "终端显示当前50ps进度；`runs/*/cloud_admission.json`记录真实GPU、软件版本和回读结果。"
        "最长小样及回读各15分钟，超时停止子进程并保留日志。"
        "计算结束后实例仍消耗额度，须在控制台停止；系统盘不扩容。\n\n"
        "本包保持OpenMM8.6与既有科学参数，CUDA依赖限定12.4以匹配云端R550驱动。"
        "云端运行前仍需真实CUDA检查；本地Blackwell的CUDA12.9环境保持。"
        "相同协议不保证不同GPU的轨迹逐位一致。原始完整环境锁定见`evidence/original_environment`。\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"status": "exported", "output": str(args.output), "objects": len(objects)},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

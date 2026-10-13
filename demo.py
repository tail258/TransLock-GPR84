"""用正式标准化和主模型对少量分子评分；不生成全库排名。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import submission
from src.translock_screening.screening_application import prepare_catalogue


ROOT = Path(__file__).resolve().parent
PRETRAINED = ROOT / "models/pretrained/chemeleon_mp.pt"


def run(input_file: Path, output_dir: Path, model_dir: Path, device: str) -> Path:
    """先验证输入和资源，再创建全新的输出目录。"""
    source = input_file.resolve()
    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"输出目录已存在: {output}")
    raw = pd.read_csv(source)
    required = {"molecule_chembl_id", "canonical_smiles"}
    if raw.empty or not required.issubset(raw.columns):
        raise ValueError("输入应有molecule_chembl_id、canonical_smiles两列和至少一条记录")
    ids = raw["molecule_chembl_id"]
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError("分子ID不能缺失或重复")

    directory, archived = submission.model_source(model_dir)
    if not PRETRAINED.is_file():
        raise FileNotFoundError(f"缺少预训练架构资源: {PRETRAINED}")

    known_ids = set(pd.read_csv(ROOT / "data/registry/excluded_known_ids.csv")["known_id"])
    known_connections = set(
        pd.read_csv(ROOT / "data/registry/excluded_connections.csv")["connection_id"]
    )
    prepared = prepare_catalogue(
        raw,
        known_ids,
        known_connections,
        submission.settings()["scope"],
    )
    ready = prepared.loc[prepared.status.eq("ready_for_scoring")].copy()
    prepared["prediction_raw_logit"] = np.nan
    if not ready.empty:
        model = submission.load_model(20261001, device, directory)
        graphs = submission.GraphCache(ready.smiles.tolist(), device)
        values = submission.predict(model, graphs, np.arange(len(ready)), 16)
        prepared.loc[ready.index, "prediction_raw_logit"] = values

    prepared["prediction_unit"] = "dimensionless_raw_logit"
    prepared["score_context"] = "small_input_no_catalogue_rank"
    prepared["model_seed"] = 20261001
    output.mkdir(parents=True, exist_ok=False)
    prepared.to_csv(output / "demo_results.csv", index=False)
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "stage": "small_input_inference",
                "input_records": len(raw),
                "scored_graphs": len(ready),
                "status_counts": prepared.status.value_counts().to_dict(),
                "model_seed": 20261001,
                "archived_model_directory": archived,
                "device": device,
                "interpretation": "Raw logits are not activity probabilities or full-catalogue ranks.",
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "examples/demo_molecules.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    destination = run(args.input, args.output, args.model_dir, args.device)
    print(f"完成：{destination / 'demo_results.csv'}")

"""回读归档排名、指标和结构，并以固定26图推理核验正式权重。"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from rdkit import Chem
from translock_screening.external_validation import collapse_connections, fixed_metrics
from translock_screening.training_factor import GraphCache, make_splits, predict
from translock_screening.screening_application import add_rankings, nominate
from submission import load_model, results_table, save_json, settings


def verify(device, output, inference=True):
    cfg = settings()
    output = Path(output)
    if not output.is_absolute():
        output = ROOT / output
    if output.exists():
        raise FileExistsError("核验输出目录已存在，不能覆盖")
    checks = []
    units = pd.read_csv(ROOT / "data/training/training_units.csv")
    admitted = pd.read_csv(ROOT / "data/training/admitted_measurements.csv")
    assert (len(admitted), len(units), units.smiles.nunique(), units.document_id.nunique(),
            units.assay_id.nunique()) == (285, 277, 248, 6, 17)
    _, members = make_splits(units)
    pd.testing.assert_frame_equal(members,
        pd.read_csv(ROOT / "data/training/fold_members.csv"), check_dtype=False)
    checks.append("数据规模及六篇文献留出成员精确回读")
    scored = pd.read_csv(ROOT / "results/screening/scored_records.csv")
    all_records = pd.read_csv(ROOT / "results/screening/all_records.csv")
    catalogue = pd.read_csv(ROOT / "data/catalogue/structures.csv")
    assert len(catalogue) == len(all_records) == 100000
    assert len(scored) == 82883 and scored.candidate_id.is_unique and scored.smiles.is_unique
    assert all_records.status.value_counts().to_dict() == {
        "scored_candidate_hypothesis_only": 82883, "structure_or_scope_excluded": 14420,
        "duplicate_excluded": 2685, "known_target_excluded": 12}
    assert set(scored.candidate_id) == set(all_records.loc[
        all_records.status.eq("scored_candidate_hypothesis_only"), "candidate_id"])
    assert all_records.candidate_id.tolist() == catalogue.molecule_chembl_id.tolist()
    assert not set(scored.connection_id).intersection(units.connection_id)
    ids = set(pd.read_csv(ROOT / "data/registry/excluded_known_ids.csv").known_id)
    connections = set(pd.read_csv(ROOT / "data/registry/excluded_connections.csv").connection_id)
    assert not set(scored.candidate_id).intersection(ids)
    assert not set(scored.connection_id).intersection(connections)
    checks.append("100000条去向、82883评分图及训练/已知身份排除")
    ranked = add_rankings(scored.copy(), cfg["training"]["seeds"])
    for column in [f"rank_{s}" for s in cfg["training"]["seeds"]] + ["similarity_reference_rank"]:
        np.testing.assert_array_equal(ranked[column], scored[column])
    nominated, pool = nominate(ranked, cfg["ranking"])
    archived = pd.read_csv(ROOT / "results/screening/candidate_hypotheses.csv")
    assert pool == 829 and len(nominated) == 10
    pd.testing.assert_frame_equal(nominated.reset_index(drop=True), archived, check_dtype=False,
                                  rtol=0, atol=1e-12)
    sdf = list(Chem.SDMolSupplier(str(ROOT / "results/screening/candidate_hypotheses_2d.sdf")))
    assert len(sdf) == 10 and all(m is not None for m in sdf)
    for mol, row in zip(sdf, archived.itertuples(), strict=True):
        assert mol.GetProp("_Name") == row.candidate_id
        assert Chem.MolToSmiles(mol) == row.smiles
    checks.append("完整三种子排名、829项池、10项提名及2D身份")
    external = pd.read_csv(ROOT / "results/external/predictions.csv")
    expected = pd.read_csv(ROOT / "results/external/metrics.csv")
    assert len(external) == 61 and external.label.value_counts().to_dict() == {1: 53, 0: 8}
    metric_rows = []
    for view, frame in [("primary", external), ("connection_collapsed", collapse_connections(external))]:
        for method in [f"score_{s}" for s in cfg["training"]["seeds"]] + ["nearest_strong_similarity"]:
            actual = fixed_metrics(frame.label, frame[method])
            target = expected[expected.view.eq(view) & expected.method.eq(method)].iloc[0]
            for key, value in actual.items():
                np.testing.assert_allclose(value, target[key], rtol=0, atol=1e-12)
            metric_rows.append({"view": view, "method": method, **actual})
    checks.append("61项外部数据及60连接图口径的八组指标回读")
    structures = json.loads((ROOT / "results/structure_evidence/exported_complexes.json").read_text())
    assert len(structures) == 30
    assert sum(r["role"] == "directory_candidate_hypothesis" for r in structures) == 28
    for record in structures:
        mol = Chem.SDMolSupplier(str(ROOT / "results/structures" / Path(record["sdf"]).name),
                                removeHs=False)[0]
        assert mol is not None
        assert Chem.MolToSmiles(Chem.RemoveHs(mol)) == Chem.MolToSmiles(Chem.MolFromSmiles(record["smiles"]))
        lines = (ROOT / "results/structures" / Path(record["pdb"]).name).read_text().splitlines()
        ligand = [l for l in lines if l.startswith("HETATM") and l[21] == record["ligand_chain"]]
        # 原导出保留SDF显式氢，不能把PDB原子数与仅重原子数直接比较。
        assert len(ligand) == mol.GetNumAtoms()
        xyz = np.array([[float(l[a:b]) for a, b in [(30, 38), (38, 46), (46, 54)]] for l in ligand])
        np.testing.assert_allclose(xyz, mol.GetConformer().GetPositions(), rtol=0, atol=0.00051)
    checks.append("28份候选及2份对照PDB/SDF化学身份和配体坐标")
    recovery = []
    if inference:
        saved_fit = pd.read_csv(ROOT / "logs/full_fit/training_fit_predictions.csv")
        graphs = GraphCache(units.smiles.iloc[:16].tolist(), device)
        candidates = GraphCache(archived.smiles.tolist(), device)
        for seed in cfg["training"]["seeds"]:
            model = load_model(seed, device)
            values = predict(model, graphs, np.arange(16), 16)
            saved = saved_fit[saved_fit.seed.eq(seed)].set_index("unit_id")\
                .loc[units.unit_id.iloc[:16], "training_logit"].to_numpy()
            error_fit = float(np.max(np.abs(values - saved)))
            np.testing.assert_allclose(values, saved, rtol=0, atol=1e-5)
            values = predict(model, candidates, np.arange(10), 16)
            error_candidates = float(np.max(np.abs(values - archived[f"score_{seed}"].to_numpy())))
            np.testing.assert_allclose(values, archived[f"score_{seed}"], rtol=0, atol=1e-5)
            recovery.append({"seed": seed, "train_16_max_error": error_fit,
                             "candidate_10_max_error": error_candidates})
            del model
        checks.append("三个最终权重加载后16个训练样本和10个候选推理回读")
    result = results_table(archived, "screening_application_2026-10-01_v1",
                           "正式提名；单专利外部回顾结果不等于候选实测验证")
    for column in ["structure_pdb_files", "ligand_sdf_files"]:
        assert result[column].str.len().gt(0).all()
        for files in result[column]:
            assert all((ROOT / f).is_file() for f in files.split(";"))
    output.mkdir(parents=True)
    result.to_csv(output / "results.csv", index=False, encoding="utf-8")
    pd.DataFrame(metric_rows).to_csv(output / "external_metrics_readback.csv", index=False)
    summary = {"status": "passed", "checks": checks, "model_recovery": recovery,
               "inference_replayed": inference, "full_library_rescored": False,
               "retrained": False, "hashes_computed": False,
               "final_results": 10, "structures": 30}
    save_json(output / "verification.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--no-inference", action="store_true")
    args = parser.parse_args()
    verify(args.device, args.output, not args.no_inference)

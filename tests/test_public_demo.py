"""公开小样入口的身份准入和缺资源行为。"""

import numpy as np
import pandas as pd
import pytest

import demo
from src import submission


def test_demo_keeps_exclusions_and_scores_only_admitted_graphs(tmp_path, monkeypatch):
    source = demo.ROOT / "examples/demo_molecules.csv"
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    pretrained = tmp_path / "chemeleon_mp.pt"
    pretrained.touch()
    monkeypatch.setattr(demo, "PRETRAINED", pretrained)
    monkeypatch.setattr(demo.submission, "model_source", lambda _: (model_dir, False))
    monkeypatch.setattr(demo.submission, "load_model", lambda *args: object())
    monkeypatch.setattr(demo.submission, "GraphCache", lambda smiles, device: smiles)
    monkeypatch.setattr(
        demo.submission,
        "predict",
        lambda model, graphs, indices, batch_size: np.array([1.25, 2.5]),
    )

    output = demo.run(source, tmp_path / "output", model_dir, "cpu")
    result = pd.read_csv(output / "demo_results.csv")
    assert result.candidate_id.tolist() == [
        "CHEMBL120691", "CHEMBL1086046", "CHEMBL3717853", "DEMO_OUT_OF_SCOPE"
    ]
    assert result.prediction_raw_logit.notna().sum() == 2
    assert result.loc[2, "status"] == "known_target_excluded"
    assert result.loc[3, "status"] == "structure_or_scope_excluded"
    assert result.score_context.eq("small_input_no_catalogue_rank").all()


def test_demo_rejects_missing_weights_before_writing_output(tmp_path):
    with pytest.raises(FileNotFoundError):
        demo.run(demo.ROOT / "examples/demo_molecules.csv", tmp_path / "output", tmp_path, "cpu")
    assert not (tmp_path / "output").exists()


def test_public_result_table_omits_unavailable_pose_links():
    frame = pd.read_csv(demo.ROOT / "results/screening/candidate_hypotheses.csv").head(1)
    table = submission.results_table(frame, "public_fixture", "small check")
    assert table.structure_pdb_files.tolist() == [""]
    assert table.ligand_sdf_files.tolist() == [""]

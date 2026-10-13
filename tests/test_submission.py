"""提交入口的防误用与文件对应关系，不重训或重复全库评分。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import submission


def test_full_catalogue_rejects_truncated_input(tmp_path, monkeypatch):
    source = tmp_path / "structures.csv"
    pd.DataFrame({"molecule_chembl_id": ["CHEMBL1"],
                  "canonical_smiles": ["CCO"]}).to_csv(source, index=False)
    (tmp_path / "manifest.json").write_text(
        (submission.ROOT / "data/catalogue/manifest.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    monkeypatch.setattr("sys.argv", ["screen.py", "--input", str(source),
                                     "--output", str(output)])
    with pytest.raises(ValueError):
        submission.screen_main()
    assert not output.exists()


def test_real_candidate_table_corresponds_to_28_structures():
    frame = pd.read_csv(submission.ROOT / "results/screening/candidate_hypotheses.csv")
    table = submission.results_table(frame, "test_readback", "test")
    files = [f for value in table.structure_pdb_files for f in value.split(";")]
    assert len(files) == len(set(files)) == 28
    assert all((submission.ROOT / f).is_file() for f in files)
    assert table.candidate_id.tolist() == frame.candidate_id.tolist()
    assert table.loc[table.md_completed_replicates.eq(3), "candidate_id"].tolist() == [
        "CHEMBL120691", "CHEMBL1086046"]


@pytest.fixture
def fitted_directory(tmp_path):
    """接口测试只创建权重占位文件；真实权重加载由正式核验入口执行。"""
    folder = tmp_path / "trained"
    folder.mkdir()
    for seed in submission.settings()["training"]["seeds"]:
        (folder / f"model_{seed}.pt").touch()
    submission.save_json(folder / "training_config.json", submission.settings()["training"])
    submission.save_json(folder / "run_manifest.json", {
        "status": "complete", "stage": "train_full", "units": 277, "graphs": 248,
        "epochs": 30, "package_version": submission.settings()["package_version"],
        "started_utc": "2026-10-03T00:00:00+00:00",
    })
    (folder / "training_members.csv").write_bytes(
        (submission.ROOT / "data/training/training_units.csv").read_bytes())
    return folder


def test_completed_fixed_fit_is_accepted(fitted_directory):
    directory, archived = submission.model_source(fitted_directory)
    assert directory == fitted_directory.resolve()
    assert not archived


@pytest.mark.parametrize("fault", ["missing_weight", "unfinished", "epochs", "members"])
def test_invalid_refit_never_falls_back_to_archive(fitted_directory, fault):
    if fault == "missing_weight":
        (fitted_directory / "model_20261003.pt").rename(fitted_directory / "not_a_model")
    elif fault == "members":
        members = pd.read_csv(fitted_directory / "training_members.csv")
        members.loc[0, "smiles"] = "CCO"
        members.to_csv(fitted_directory / "training_members.csv", index=False)
    else:
        target = fitted_directory / "run_manifest.json"
        record = json.loads(target.read_text(encoding="utf-8"))
        record["status" if fault == "unfinished" else "epochs"] = "running" if fault == "unfinished" else 1
        submission.save_json(target, record)
    with pytest.raises((FileNotFoundError, ValueError, AssertionError)):
        submission.model_source(fitted_directory)


def test_pipeline_passes_new_weights_and_preserves_result(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(submission, "catalogue_input", lambda p: events.append("validate"))

    def train(argv):
        folder = Path(argv[argv.index("--output") + 1])
        folder.mkdir()
        events.append(("train", folder))
        return folder

    def screen(argv):
        assert Path(argv[argv.index("--model-dir") + 1]) == events[1][1]
        folder = Path(argv[argv.index("--output") + 1])
        folder.mkdir()
        (folder / "results.csv").write_text("candidate_id\nfixture\n", encoding="utf-8")
        events.append("screen")
        return folder

    monkeypatch.setattr(submission, "train_main", train)
    monkeypatch.setattr(submission, "screen_main", screen)
    output = submission.pipeline("data/catalogue/structures.csv", tmp_path / "flow", "cpu")
    assert events[0] == "validate" and events[2] == "screen"
    assert (output / "results.csv").read_bytes() == (output / "screening/results.csv").read_bytes()
    assert json.loads((output / "pipeline_manifest.json").read_text())["status"] == "complete"


def test_pipeline_training_failure_does_not_start_screen(tmp_path, monkeypatch):
    monkeypatch.setattr(submission, "catalogue_input", lambda p: None)

    def fail(argv):
        raise RuntimeError("training_failed_fixture")

    monkeypatch.setattr(submission, "train_main", fail)
    monkeypatch.setattr(submission, "screen_main", lambda **kw: pytest.fail("不应启动筛选"))
    with pytest.raises(RuntimeError, match="training_failed_fixture"):
        submission.pipeline("unused.csv", tmp_path / "flow", "cpu")
    record = json.loads((tmp_path / "flow/pipeline_manifest.json").read_text())
    assert record["status"] == "failed"
    assert not (tmp_path / "flow/results.csv").exists()


def test_invalid_pipeline_input_does_not_start_training(tmp_path, monkeypatch):
    source = tmp_path / "invalid.csv"
    pd.DataFrame({"molecule_chembl_id": ["X"]}).to_csv(source, index=False)
    monkeypatch.setattr(submission, "train_main", lambda **kw: pytest.fail("不应启动训练"))
    with pytest.raises(ValueError, match="缺少"):
        submission.pipeline(source, tmp_path / "flow", "cpu")
    assert not (tmp_path / "flow").exists()


def test_refit_screen_uses_selected_weights_and_new_provenance(fitted_directory, tmp_path, monkeypatch):
    # 仅对衔接接口使用三个真实分子图，推理替身不产生科研结果。
    frame = pd.read_csv(submission.ROOT / "results/screening/candidate_hypotheses.csv").head(3)
    frame = frame.drop(columns=[c for c in frame if c.startswith(("rank_", "score_")) or c == "nomination_order"])
    frame["status"] = "ready_for_scoring"
    monkeypatch.setattr(submission, "catalogue_input", lambda p: pd.DataFrame(index=range(3)))
    monkeypatch.setattr(submission, "prepare_catalogue", lambda *a: frame.copy())
    monkeypatch.setattr(submission.importlib.metadata, "version", lambda n: "2.2.0")
    monkeypatch.setattr(submission, "validate_training_scope", lambda *a: None)
    monkeypatch.setattr(submission, "GraphCache", lambda *a: None)
    loaded = []

    def load(seed, device, folder):
        loaded.append((seed, folder))
        return seed

    monkeypatch.setattr(submission, "load_model", load)
    monkeypatch.setattr(submission, "predict", lambda *a: np.array([1., 2., 3.]))
    monkeypatch.setattr(submission, "nearest_support", lambda *a: pd.DataFrame())
    monkeypatch.setattr(submission, "metadata", lambda *a, **kw: {"status": "complete", **kw})
    output = submission.screen_main(argv=["--output", str(tmp_path / "screen"),
        "--model-dir", str(fitted_directory), "--device", "cpu"])
    assert loaded == [(s, fitted_directory.resolve()) for s in submission.settings()["training"]["seeds"]]
    result = pd.read_csv(output / "results.csv")
    assert result.candidate_id.tolist() == [frame.candidate_id.iloc[2]]
    assert result.model_version.str.startswith("CheMeleon_GPR84_refit_").all()
    assert result.structure_path_base.eq("package_root").all()
    assert not json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))["archived_weights"]


def test_changed_structure_does_not_inherit_candidate_md():
    frame = pd.read_csv(submission.ROOT / "results/screening/candidate_hypotheses.csv").head(1)
    frame["candidate_id"] = "CHEMBL120691"
    frame["smiles"] = "CCO"
    result = submission.results_table(frame, "fixture", "fixture")
    assert result.structure_pdb_files.iloc[0] == ""
    assert result.md_completed_replicates.iloc[0] == 0


def training_directory(tmp_path):
    """仅建立目录契约fixture；权重软链接指向已有文件，不训练新模型。"""
    cfg = submission.settings()
    for seed in cfg["training"]["seeds"]:
        (tmp_path / f"model_{seed}.pt").symlink_to(submission.ROOT / "models" / f"model_{seed}.pt")
    submission.save_json(tmp_path / "training_config.json", cfg["training"])
    (tmp_path / "training_members.csv").write_bytes(
        (submission.ROOT / "data/training/training_units.csv").read_bytes())
    submission.save_json(tmp_path / "run_manifest.json", {
        "status": "complete", "stage": "train_full", "units": 277,
        "graphs": 248, "epochs": 30, "package_version": cfg["package_version"],
        "started_utc": "2026-10-03T00:00:00+00:00",
    })
    return tmp_path


def test_model_source_rejects_missing_seed_without_archive_fallback(tmp_path):
    with pytest.raises(FileNotFoundError, match="model_20261001"):
        submission.model_source(tmp_path)


def test_refit_directory_requires_exact_training_contract(tmp_path):
    training_directory(tmp_path)
    assert submission.model_source(tmp_path) == (tmp_path.resolve(), False)
    cfg = submission.settings()["training"]
    cfg["epochs"] = 1
    submission.save_json(tmp_path / "training_config.json", cfg)
    with pytest.raises(ValueError, match="固定配置"):
        submission.model_source(tmp_path)


def test_refit_directory_rejects_changed_training_member(tmp_path):
    training_directory(tmp_path)
    units = pd.read_csv(tmp_path / "training_members.csv")
    units.loc[0, "label"] = 1 - units.loc[0, "label"]
    units.to_csv(tmp_path / "training_members.csv", index=False)
    with pytest.raises(AssertionError):
        submission.model_source(tmp_path)


def test_selected_weight_directory_real_graph_inference(tmp_path):
    """两个真实候选加载显式目录的已有主权重并回读分数，不拟合或评分全库。"""
    training_directory(tmp_path)
    directory, _ = submission.model_source(tmp_path)
    frame = pd.read_csv(submission.ROOT / "results/screening/candidate_hypotheses.csv").head(2)
    model = submission.load_model(20261001, "cpu", directory)
    graphs = submission.GraphCache(frame.smiles.tolist(), "cpu")
    values = submission.predict(model, graphs, np.arange(2), 16)
    np.testing.assert_allclose(values, frame.score_20261001, rtol=0, atol=1e-5)


def test_new_result_model_version_and_structure_identity():
    frame = pd.read_csv(submission.ROOT / "results/screening/candidate_hypotheses.csv").head(1)
    frame.loc[0, "smiles"] = "CCO"
    table = submission.results_table(frame, "new_run", "正式重拟合", model_version="new_model")
    assert table.model_version.tolist() == ["new_model"]
    assert table.structure_pdb_files.tolist() == [""]
    assert table.md_completed_replicates.tolist() == [0]


def test_pipeline_passes_training_output_to_screen_and_copies_results(tmp_path, monkeypatch):
    calls = []
    def fit(argv):
        directory = submission.path(argv[argv.index("--output") + 1])
        directory.mkdir()
        calls.append(("train", directory))
        return directory
    def screen(argv):
        directory = submission.path(argv[argv.index("--output") + 1])
        assert submission.path(argv[argv.index("--model-dir") + 1]) == calls[0][1]
        directory.mkdir()
        (directory / "results.csv").write_text("candidate_id\nTEST\n", encoding="utf-8")
        calls.append(("screen", directory))
        return directory
    monkeypatch.setattr(submission, "train_main", fit)
    monkeypatch.setattr(submission, "screen_main", screen)
    output = submission.pipeline("data/catalogue/structures.csv", tmp_path / "run", "cpu")
    assert [stage for stage, _ in calls] == ["train", "screen"]
    assert (output / "results.csv").read_bytes() == (output / "screening/results.csv").read_bytes()
    assert json.loads((output / "pipeline_manifest.json").read_text())["status"] == "complete"
    with pytest.raises(FileExistsError):
        submission.pipeline("data/catalogue/structures.csv", output, "cpu")


def test_pipeline_stops_and_records_failure_before_screening(tmp_path, monkeypatch):
    def fail(argv):
        raise RuntimeError("训练未完成")
    def unexpected(argv):
        pytest.fail("训练失败后不应筛选")
    monkeypatch.setattr(submission, "train_main", fail)
    monkeypatch.setattr(submission, "screen_main", unexpected)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="训练未完成"):
        submission.pipeline("data/catalogue/structures.csv", output, "cpu")
    assert json.loads((output / "pipeline_manifest.json").read_text())["status"] == "failed"


def test_screen_uses_refit_directory_and_preserves_new_nomination(tmp_path, monkeypatch):
    directory = tmp_path / "training"
    directory.mkdir()
    training_directory(directory)
    frame = pd.read_csv(submission.ROOT / "results/screening/candidate_hypotheses.csv").head(2)
    prepared = frame[["candidate_id", "smiles", "scaffold"]].copy()
    prepared["status"] = "ready_for_scoring"
    selected = []
    monkeypatch.setattr(submission, "prepare_catalogue", lambda *a: prepared)
    monkeypatch.setattr(submission, "GraphCache", lambda *a: None)
    def load(seed, device, model_dir):
        selected.append((seed, model_dir))
        return seed
    monkeypatch.setattr(submission, "load_model", load)
    monkeypatch.setattr(submission, "predict", lambda seed, *a: frame[f"score_{seed}"].to_numpy())
    output = tmp_path / "screening"
    submission.screen_main(argv=["--output", str(output), "--model-dir", str(directory),
                                 "--device", "cpu"])
    assert selected == [(seed, directory.resolve()) for seed in submission.settings()["training"]["seeds"]]
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["archived_weights"] is False and manifest["status"] == "complete"
    table = pd.read_csv(output / "results.csv")
    assert table.model_version.str.contains("refit_seed20261001").all()
    assert len(table) == 1  # 小型流程fixture，不要求与十项历史提名相同。

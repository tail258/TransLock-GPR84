"""从新表、原回读和导出坐标独立核对解释产物，不重新运行积分。"""

import argparse
import json
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
from rdkit import Chem


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in {"href", "src"} and value:
                self.paths.append(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--representatives", type=Path)
    args = parser.parse_args()
    summaries = load(args.analysis / "summaries.json")
    contacts = pd.read_csv(args.analysis / "contact_details.csv")
    assert len(summaries) == 9
    sdf_count, frames = 0, 0
    for summary in summaries:
        cid, seed = summary["ligand_id"], summary["seed"]
        folder = args.analysis / cid / str(seed)
        source = Path(summary["source_run"])
        definition = load(args.analysis / cid / "ligand_definition.json")
        table = pd.read_csv(folder / "displacements.csv")
        original = pd.read_csv(source / "analysis_v1/frames.csv")
        assert len(table) == len(original) == 500
        assert np.array_equal(table.time_ps, np.arange(1, 501) * 10)
        assert np.allclose(
            table.original_total_A, original.ligand_RMSD_to_original_pose_A, atol=2e-4
        )
        assert np.allclose(
            table.replica_start_total_A, original.ligand_RMSD_to_replica_start_A, atol=2e-4
        )
        for ref in ["original", "replica_start"]:
            assert np.allclose(
                table[ref + "_total_A"] ** 2,
                table[ref + "_centroid_shift_A"] ** 2 + table[ref + "_centered_A"] ** 2,
                atol=1e-10,
            )
        data = np.load(folder / "aligned_ligand_frames.npz")
        xyz = data["xyz_A"]
        reference = np.load(args.analysis / cid / "reference_geometry.npz")["ligand"]
        assert xyz.shape == (500, definition["heavy_atom_count"], 3)
        direct = np.sqrt(((xyz - reference) ** 2).sum(axis=2).mean(axis=1))
        assert np.allclose(direct, table.original_total_A, atol=1e-10)
        # 直接做两两坐标差验证代表帧，不复用生产中的Gram矩阵距离。
        mean_distances = np.array(
            [np.sqrt(((xyz - point) ** 2).sum(axis=2).mean(axis=1)).mean() for point in xyz]
        )
        medoid = int(np.argmin(mean_distances))
        assert medoid + 1 == summary["representative_frame"]
        for label, frame in [
            ("representative", medoid),
            ("late_representative", summary["late_representative_frame"] - 1),
        ]:
            representative_folder = (
                args.representatives / cid / str(seed) if args.representatives else folder
            )
            mol = Chem.SDMolSupplier(str(representative_folder / f"{label}.sdf"), removeHs=True)[0]
            assert mol is not None and Chem.MolToInchiKey(mol) == definition["inchikey"]
            if args.representatives:
                assert not mol.HasProp("atom.dprop.PartialCharge")
            assert np.allclose(mol.GetConformer().GetPositions(), xyz[frame], atol=6e-5)
            sdf_count += 1
        for key, stats in summary["metrics"].items():
            values = table[key].to_numpy()
            actual = dict(
                median=np.median(values),
                minimum=values.min(),
                maximum=values.max(),
                first_100_mean=values[:100].mean(),
                last_100_mean=values[-100:].mean(),
                final=values[-1],
            )
            assert all(np.isclose(stats[k], v, atol=1e-10) for k, v in actual.items())
        c = contacts[(contacts.ligand_id == cid) & (contacts.seed == seed)].set_index("residue")
        old = load(source / "analysis_v1/independent_readback.json")
        for row in old["contact_occupancy"]:
            assert np.isclose(c.loc[row["residue"], "occupancy"], row["frames"] / 500)
        frames += 500
    source_manifest = load(args.evidence / "source_manifest.json")
    table = pd.read_csv(args.evidence / "candidate_evidence.csv")
    old = pd.read_csv(Path(source_manifest["sources"]["candidates"]["path"]))
    assert table.candidate_id.tolist() == old.candidate_id.tolist()
    for key in [
        "nomination_order",
        "rank_20261001",
        "rank_20261002",
        "rank_20261003",
        "unspecified_tetrahedral_centers",
    ]:
        assert np.array_equal(table[key], old[key])
    assert table.MD_replicates.tolist() == [3, 0, 0, 0, 0, 0, 0, 3, 0, 0]
    assert len(pd.read_csv(args.evidence / "all_candidate_pose_states.csv")) == 14
    assert len(pd.read_csv(args.evidence / "public_measurements_source.csv")) == 41
    assert len(pd.read_csv(args.evidence / "training_neighbor_measurements.csv")) == 30
    refs_count = 0
    for page in [args.analysis / "figures/index.html", args.evidence / "index.html"]:
        refs = References()
        refs.feed(page.read_text(encoding="utf-8"))
        for value in refs.paths:
            parsed = urlparse(value)
            if parsed.scheme or not parsed.path:
                continue
            assert (page.parent / unquote(parsed.path)).resolve().exists(), value
            refs_count += 1
    result = dict(
        status="passed",
        checked_at=datetime.now(UTC).isoformat(),
        frames=frames,
        original_RMSDs_match=True,
        displacement_identity_passed=True,
        independent_medoid_check=True,
        representative_SDFs=sdf_count,
        candidate_order_preserved=True,
        candidate_pose_states=14,
        local_HTML_references=refs_count,
        hashing=False,
        simulation_started=False,
    )
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

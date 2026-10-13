"""固定目录与已知靶点清单的关键身份/分页保护。"""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_screening_catalogue.py"
spec = importlib.util.spec_from_file_location("fetch_screening_catalogue", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def page(key, items, offset=0, limit=1000, total=None, order_by="molecule_chembl_id"):
    total = total if total is not None else len(items)
    next_link = None
    if offset + len(items) < total:
        next_link = f"/chembl/api/data/{key}.json?offset={offset + len(items)}&order_by={order_by}"
    return {
        key: items,
        "page_meta": {"offset": offset, "limit": limit, "total_count": total, "next": next_link},
    }


def test_check_page_rejects_offset_total_and_sort_link():
    good = page("molecules", [{"molecule_chembl_id": "CHEMBL1"}], total=2)
    assert module.check_page(good, "molecules", 0, 1000, None, "molecule_chembl_id")[1] == 2
    bad = json.loads(json.dumps(good))
    bad["page_meta"]["offset"] = 1
    with pytest.raises(ValueError):
        module.check_page(bad, "molecules", 0, 1000, None, "molecule_chembl_id")
    bad = json.loads(json.dumps(good))
    bad["page_meta"]["total_count"] = 3
    with pytest.raises(ValueError):
        module.check_page(bad, "molecules", 0, 1000, 2, "molecule_chembl_id")
    bad = json.loads(json.dumps(good))
    bad["page_meta"]["next"] = "/chembl/api/data/molecule.json?offset=1"
    with pytest.raises(ValueError):
        module.check_page(bad, "molecules", 0, 1000, None, "molecule_chembl_id")


def test_catalogue_rejects_duplicate_and_wrong_type(tmp_path, monkeypatch):
    manifest = {"requests": [], "pages": [], "statistics": {}}
    items = [
        {
            "molecule_chembl_id": f"CHEMBL{i:04d}",
            "molecule_type": "Small molecule",
            "molecule_structures": {"canonical_smiles": "CC"},
        }
        for i in range(1000)
    ]
    monkeypatch.setattr(module, "request_json", lambda *args: page("molecules", items, total=1000))
    module.catalogue(tmp_path, manifest, 1000)
    assert manifest["statistics"]["downloaded_records"] == 1000
    assert (tmp_path / "structures.csv").read_text(encoding="utf-8").count("\n") == 1001

    other = tmp_path / "other"
    other.mkdir()
    items[1]["molecule_type"] = "Protein"
    with pytest.raises(ValueError, match="Small molecule"):
        module.catalogue(other, {"requests": [], "pages": [], "statistics": {}}, 1000)

    another = tmp_path / "another"
    another.mkdir()
    items[1]["molecule_type"] = "Small molecule"
    items[1]["molecule_chembl_id"] = items[0]["molecule_chembl_id"]
    with pytest.raises(ValueError):
        module.catalogue(another, {"requests": [], "pages": [], "statistics": {}}, 1000)


def test_existing_output_is_rejected_before_writing(tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        module.run(tmp_path, probe=True)
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_known_target_collects_all_activity_ids_and_batch_molecules(tmp_path, monkeypatch):
    manifest = {"requests": [], "pages": [], "statistics": {}}

    def fake_request(root, state, url, filename):
        if "activity.json" in url:
            return page(
                "activities",
                [
                    {"activity_id": 1, "molecule_chembl_id": "CHEMBL1"},
                    {"activity_id": 2, "molecule_chembl_id": "CHEMBL1"},
                    {"activity_id": 3, "molecule_chembl_id": "CHEMBL2"},
                ],
                limit=1000,
                total=3,
            )
        return page(
            "molecules",
            [
                {
                    "molecule_chembl_id": "CHEMBL1",
                    "molecule_type": "Small molecule",
                    "molecule_structures": {"canonical_smiles": "CC"},
                },
                {
                    "molecule_chembl_id": "CHEMBL2",
                    "molecule_type": "Small molecule",
                    "molecule_structures": {"canonical_smiles": "CO"},
                },
            ],
            limit=100,
            total=2,
        )

    monkeypatch.setattr(module, "request_json", fake_request)
    module.known_target(tmp_path, manifest)
    assert manifest["statistics"]["known_target_activity_count"] == 3
    assert manifest["statistics"]["known_target_unique_molecule_ids"] == 2
    assert manifest["statistics"]["known_target_unresolved_molecule_ids"] == []
    assert (tmp_path / "known_target_activities.csv").read_text(encoding="utf-8").count("\n") == 4

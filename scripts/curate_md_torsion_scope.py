"""排除sp近线性四原子角，保留原候选角度表；只重绘受影响的对照图。"""

import argparse
import json
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem

from translock_screening.md_interactions import torsion_definitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("analysis", "existing", "bundle", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    table = pd.read_csv(args.analysis / "torsion_summary.csv")
    keep_rows, exclusions, selected = [], [], {}
    for cid in table.ligand_id.unique():
        source = args.bundle / "inputs" / cid / "parameters/ligand.sdf"
        mol = Chem.RemoveHs(Chem.SDMolSupplier(str(source), removeHs=False)[0])
        core = json.loads((args.existing / cid / "ligand_definition.json").read_text())[
            "core_atom_indices"
        ]
        current = torsion_definitions(mol, core)
        valid = {t["bond_index"] for t in current}
        old = json.loads((args.analysis / cid / "chemical_definitions.json").read_text())[
            "torsions"
        ]
        selected[cid] = [t for t in old if t["bond_index"] in valid]
        for t in old:
            if t["bond_index"] not in valid:
                assert any(
                    mol.GetAtomWithIdx(i).GetHybridization() == Chem.HybridizationType.SP
                    for i in t["atoms"][1:3]
                )
                exclusions.append(
                    {"ligand_id": cid, **t, "reason": "sp-adjacent near-linear angle"}
                )
        keep_rows.append(table[(table.ligand_id == cid) & table.bond_index.isin(valid)])
    assert [(e["ligand_id"], e["bond_index"]) for e in exclusions] == [
        ("GLPG1205", 25),
        ("GLPG1205", 28),
    ]
    pd.concat(keep_rows).to_csv(args.output / "torsion_summary.csv", index=False)
    cid = "GLPG1205"
    fig, axes = plt.subplots(len(selected[cid]), 1, figsize=(12, 6), sharex=True)
    for seed, color in zip(
        (20261011, 20261012, 20261013), ("#2878b5", "#d77a2b", "#239474"), strict=True
    ):
        data = pd.read_csv(args.analysis / cid / str(seed) / "torsion_angles.csv")
        for ax, t in zip(axes, selected[cid], strict=True):
            ax.scatter(
                data.time_ps / 1000,
                data[f"bond_{t['bond_index']}"],
                s=2,
                color=color,
                alpha=0.55,
                label=str(seed),
            )
            ax.set(ylim=(-180, 180), yticks=(-180, 0, 180), ylabel="deg")
            ax.set_title(
                f"Bond {t['bond_index']}: {' - '.join(t['atom_names'])}", fontsize=9, loc="left"
            )
    axes[0].legend(ncol=3, fontsize=8, markerscale=3)
    axes[-1].set_xlabel("Time (ns)")
    fig.suptitle(
        "GLPG1205 | interpretable dihedrals\n"
        "Two sp-adjacent near-linear quadruplets excluded; original values preserved"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "svg", "pdf"):
        fig.savefig(args.output / f"torsions_GLPG1205.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    page = args.analysis / "figures/index.html"
    original = page.read_text(encoding="utf-8")
    (args.output / "index_before_curation.html").write_text(original, encoding="utf-8")
    updated = original.replace(
        "torsions_GLPG1205.png", "../../torsion_review_v2/torsions_GLPG1205.png"
    )
    updated = updated.replace(
        "torsions_GLPG1205.pdf", "../../torsion_review_v2/torsions_GLPG1205.pdf"
    )
    updated = updated.replace(
        "../torsion_summary.csv", "../../torsion_review_v2/torsion_summary.csv"
    )
    note = (
        "<p>对照GLPG1205的两条邻接炔碳的近线性四原子角不适合普通二面角解释，"
        "已从当前图和汇总中排除，原始表及旧图保留。其余数值保持。</p>"
    )
    updated = updated.replace(
        "<h2>GLPG1205：关键二面角</h2>", "<h2>GLPG1205：关键二面角</h2>" + note
    )
    page.write_text(updated, encoding="utf-8")
    manifest = {
        "status": "passed",
        "excluded": exclusions,
        "selected": selected,
        "new_numeric_calculation": False,
        "old_files_preserved": True,
        "wall_seconds": time.monotonic() - start,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copyfile(__file__, args.output / "source_snapshot.py")
    print(
        json.dumps(
            {"status": "passed", "excluded": len(exclusions), "seconds": time.monotonic() - start}
        )
    )


if __name__ == "__main__":
    main()

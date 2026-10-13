"""读取存档9WIJ和当前受体，定位完整膜MD前的构建体、缺区与取向问题。"""

import argparse
import importlib.util
import json
import platform
import shutil
from pathlib import Path

import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.SeqUtils import seq1


def ranges(values):
    result = []
    for value in sorted(set(values)):
        if result and value == result[-1][1] + 1:
            result[-1][1] = value
        else:
            result.append([value, value])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cif", "receptor", "uniprot", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    cif = MMCIF2Dict(str(args.cif))
    uniprot = json.loads(args.uniprot.read_text())
    sequence = uniprot["sequence"]["value"]
    ca, residues, mutations = {}, {}, []
    for line in args.receptor.read_text().splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        number, chain, name = int(line[22:26]), line[21], line[17:20]
        if number in ca:
            raise ValueError("相同原生编号出现多个CA；需显式链映射")
        xyz = np.array([float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54))])
        ca[number] = xyz
        residues.setdefault(chain, []).append(number)
        if not 1 <= number <= len(sequence):
            raise ValueError("受体编号超出原生序列，不能直接比较")
        letter = seq1(name, custom_map={"HIE": "H", "HID": "H", "HIP": "H"})
        if letter != sequence[number - 1]:
            mutations.append(
                {
                    "chain": chain,
                    "position": number,
                    "prepared": letter,
                    "uniprot": sequence[number - 1],
                }
            )
    features = [f for f in uniprot["features"] if f["type"] == "Transmembrane"]
    tm_summary, axes = [], []
    for f in features:
        begin, end = f["location"]["start"]["value"], f["location"]["end"]["value"]
        present = [p for p in range(begin, end + 1) if p in ca]
        xyz = np.array([ca[p] for p in present])
        if len(xyz) < 3:
            raise ValueError("跨膜段坐标不足")
        axis = np.linalg.svd(xyz - xyz.mean(axis=0))[2][0]
        axes.append(axis)
        tm_summary.append(
            {
                "start": begin,
                "end": end,
                "present_CA": len(present),
                "missing": [p for p in range(begin, end + 1) if p not in ca],
                "axis_abs_dot_z": float(abs(axis[2])),
            }
        )
    # 用几何近似估计共同跨膜轴，膜插入中心需进一步检查。
    eigenvalues, eigenvectors = np.linalg.eigh(sum(np.outer(a, a) for a in axes))
    axis = eigenvectors[:, int(eigenvalues.argmax())]
    axis_angle_z = float(np.degrees(np.arccos(np.clip(abs(axis[2]), 0, 1))))
    missing = ranges(set(range(1, len(sequence) + 1)) - set(ca))
    missing_categories = []
    for begin, end in missing:
        overlapping = [
            f["type"]
            for f in uniprot["features"]
            if "start" in f["location"]
            and "end" in f["location"]
            and f["location"]["start"].get("value", 100000) <= end
            and f["location"]["end"].get("value", -1) >= begin
        ]
        missing_categories.append(
            {"range": [begin, end], "feature_types": sorted(set(overlapping))}
        )
    unobserved = []
    prefix = "_pdbx_unobs_or_zero_occ_residues."
    if prefix + "auth_seq_id" in cif:
        for chain in sorted(set(cif[prefix + "auth_asym_id"])):
            nums = [
                int(n)
                for n, c in zip(
                    cif[prefix + "auth_seq_id"], cif[prefix + "auth_asym_id"], strict=True
                )
                if c == chain
            ]
            unobserved.append({"auth_chain": chain, "ranges": ranges(nums)})
    diffs = []
    prefix = "_struct_ref_seq_dif."
    if prefix + "details" in cif:
        for i in range(len(cif[prefix + "details"])):
            diffs.append(
                {
                    k: cif[prefix + k][i]
                    for k in (
                        "mon_id",
                        "db_mon_id",
                        "seq_num",
                        "pdbx_seq_db_seq_num",
                        "details",
                        "pdbx_seq_db_accession_code",
                    )
                }
            )
    alignments = []
    prefix = "_struct_ref_seq."
    if prefix + "align_id" in cif:
        for i in range(len(cif[prefix + "align_id"])):
            alignments.append(
                {
                    k: cif[prefix + k][i]
                    for k in (
                        "align_id",
                        "pdbx_db_accession",
                        "seq_align_beg",
                        "seq_align_end",
                        "db_align_beg",
                        "db_align_end",
                        "pdbx_auth_seq_align_beg",
                        "pdbx_auth_seq_align_end",
                    )
                }
            )
    chain_ends = []
    ordered_chains = list(residues)
    for left, right in zip(ordered_chains[:-1], ordered_chains[1:], strict=True):
        a, b = residues[left][-1], residues[right][0]
        chain_ends.append(
            {
                "from": f"{left}:{a}",
                "to": f"{right}:{b}",
                "CA_distance_A": float(np.linalg.norm(ca[a] - ca[b])),
                "native_missing_between": max(0, b - a - 1),
            }
        )
    internal_gaps = []
    for chain, numbers in residues.items():
        for a, b in zip(numbers[:-1], numbers[1:], strict=True):
            if b - a > 1:
                internal_gaps.append(
                    {
                        "chain": chain,
                        "from": a,
                        "to": b,
                        "native_missing_between": b - a - 1,
                        "CA_distance_A": float(np.linalg.norm(ca[a] - ca[b])),
                    }
                )
    result = {
        "status": "structure_preflight_only_not_md_acceptance",
        "python": platform.python_version(),
        "input_cif": str(args.cif),
        "input_receptor": str(args.receptor),
        "input_uniprot": str(args.uniprot),
        "uniprot_length": len(sequence),
        "prepared_CA_count": len(ca),
        "prepared_chains": {k: ranges(v) for k, v in residues.items()},
        "prepared_sequence_mismatches": mutations,
        "missing_native_ranges": missing_categories,
        "deposited_unobserved_residues": unobserved,
        "deposited_sequence_differences": diffs,
        "deposited_reference_alignments": alignments,
        "deposited_entity_sequences": {
            key: cif.get(key)
            for key in ("_entity_poly.pdbx_seq_one_letter_code_can", "_entity_poly.pdbx_strand_id")
        },
        "chain_end_separations": chain_ends,
        "same_chain_numbering_gaps": internal_gaps,
        "input_records": [
            {"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in (args.cif, args.receptor, args.uniprot)
        ],
        "transmembrane_coverage": tm_summary,
        "estimated_TM_axis_angle_to_current_z_degrees": axis_angle_z,
        "tm_axis_scope": "七段CA主轴共同方向的初步几何诊断，非膜插入位置验收",
        "parameter_packages_available": {
            name: importlib.util.find_spec(name) is not None
            for name in ("openmmforcefields", "openff")
        },
        "ambertools_executables_available": {
            name: shutil.which(name) for name in ("antechamber", "parmchk2", "tleap")
        },
        "limits": "不建膜、不积分；需决定构建体/缺环端点模型和膜取向，补齐配体参数后另验收",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), args.output / "source_snapshot.py")
    (args.output / "structure_preflight.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "prepared_CA_count",
                    "prepared_chains",
                    "prepared_sequence_mismatches",
                    "chain_end_separations",
                    "estimated_TM_axis_angle_to_current_z_degrees",
                    "parameter_packages_available",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

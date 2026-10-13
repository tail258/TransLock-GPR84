"""固定重复MD的独立随机初始化与周期边界诊断。"""

import json
from pathlib import Path

import numpy as np
import openmm
from openmm import unit
from rdkit import Chem

from translock_screening.md_diagnostics import fit_to_reference, minimum_image, pair_distances


def fresh_replica(system, state, base_seed):
    """从坐标创建新随机流；调用方抽取新速度，不加载checkpoint。"""
    if dict(state.getParameters()).get("k", 0) != 0:
        raise ValueError("重复起点仍有位置约束")
    system.setDefaultPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    seeds = {"thermostat": base_seed, "barostat": base_seed + 100, "velocities": base_seed + 200}
    for force in system.getForces():
        if isinstance(force, openmm.MonteCarloMembraneBarostat):
            force.setRandomNumberSeed(seeds["barostat"])
            force.setFrequency(25)
        if isinstance(force, openmm.CustomExternalForce):
            for i in range(force.getNumGlobalParameters()):
                if force.getGlobalParameterName(i) == "k":
                    force.setGlobalParameterDefaultValue(i, 0)
    integrator = openmm.LangevinMiddleIntegrator(
        310 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picosecond
    )
    integrator.setRandomNumberSeed(seeds["thermostat"])
    integrator.setConstraintTolerance(1e-6)
    return integrator, seeds


def nearest_molecular_image(points, center, box):
    """整体平移完整分子到参照中心最近周期像，保留内部几何。"""
    points = np.asarray(points)
    delta = points.mean(axis=0) - center
    return points + minimum_image(delta, box) - delta


class PilotDiagnostics:
    def __init__(self, warmup, starting_xyz):
        self.warmup = Path(warmup)
        self.manifest = json.loads((self.warmup / "manifest.json").read_text())
        self.atoms = json.loads((self.warmup / "atom_map.json").read_text())
        parameters = Path(self.manifest["parameters"])
        construct = Path(self.manifest["construct"])
        self.ligand = Chem.SDMolSupplier(str(parameters / "ligand.sdf"), removeHs=False)[0]
        self.key = json.loads((parameters / "manifest.json").read_text())["inchikey"]
        protein = [a for a in self.atoms if a["chain"] == "R"]
        self.protein_n = len(protein)
        self.lig_all = np.arange(self.protein_n, self.protein_n + self.ligand.GetNumAtoms())
        params = json.loads((parameters / "atom_parameters.json").read_text())
        self.lig_heavy = (
            np.array([i for i, a in enumerate(params) if a["atomic_number"] != 1]) + self.protein_n
        )
        self.original_ligand = (
            np.array([a["position_A"] for a in params if a["atomic_number"] != 1]) / 10
        )
        by_name = {(int(a["residue_id"]), a["name"]): a["index"] for a in protein}
        self.mapping = json.loads((construct / "residue_map.json").read_text())
        native = {r["native_id"]: r["model_id"] for r in self.mapping if r["native_id"] is not None}
        self.cn = [(by_name[(i, "C")], by_name[(i + 1, "N")]) for i in range(1, len(self.mapping))]
        self.ss = [
            (by_name[(native[a], "SG")], by_name[(native[b], "SG")])
            for a, b in [(11, 166), (93, 168)]
        ]
        self.alpha_ids = [i for i in range(1, len(self.mapping) + 1) if (i, "CB") in by_name]
        self.alpha = np.array(
            [[by_name[(i, n)] for n in ("N", "CA", "C", "CB")] for i in self.alpha_ids]
        )
        self.ha = np.array([by_name[(i, "HA")] for i in self.alpha_ids])
        ca_reference = {}
        for line in (construct / "retained_oriented_heavy.pdb").read_text().splitlines():
            if line.startswith("ATOM  ") and line[12:16].strip() == "CA":
                ca_reference[int(line[22:26])] = [
                    float(line[a:b]) / 10 for a, b in ((30, 38), (38, 46), (46, 54))
                ]
        ids = sorted(ca_reference)
        self.ca = np.array([by_name[(i, "CA")] for i in ids])
        self.original_ca = np.array([ca_reference[i] for i in ids])
        self.protein_heavy = np.array([a["index"] for a in protein if a["element"] != "H"])
        model_map = {r["model_id"]: r["native_id"] for r in self.mapping}
        self.heavy_residues = []
        for i in self.protein_heavy:
            a = self.atoms[i]
            native_id = model_map[int(a["residue_id"])]
            self.heavy_residues.append(
                f"{a['residue']}{native_id}"
                if native_id is not None
                else f"synthetic_ALA{a['residue_id']}"
            )
        self.phosphorus = np.array(
            [
                a["index"]
                for a in self.atoms
                if a["residue"] in {"POP", "POPC"} and a["element"] == "P"
            ]
        )
        assert len(self.phosphorus) > 50
        self.start = np.array(starting_xyz)
        self.initial_contacts = None
        self.pocket_ca = np.array(
            [
                by_name[(native[i], "CA")]
                for i in (100, 101, 169, 170, 172, 183, 187, 332, 335, 336, 339, 340)
            ]
        )

    def calculate(self, xyz, vectors):
        xyz, vectors = np.asarray(xyz), np.asarray(vectors)
        assert len(xyz) == len(self.atoms) and np.isfinite(xyz).all()
        assert np.allclose(vectors, np.diag(np.diag(vectors)), atol=1e-8)
        box = np.diag(vectors)
        assert np.isfinite(box).all() and (box > 0).all()
        cn = pair_distances(xyz, self.cn, box) * 10
        ss = pair_distances(xyz, self.ss, box) * 10
        q = self.alpha
        v = [minimum_image(xyz[q[:, i]] - xyz[q[:, 1]], box) * 10 for i in (0, 2, 3)]
        vh = minimum_image(xyz[self.ha] - xyz[q[:, 1]], box) * 10
        volume = np.einsum("ij,ij->i", np.cross(v[0], v[1]), v[2])
        hvolume = np.einsum("ij,ij->i", np.cross(v[0], v[1]), vh)
        angle = np.rad2deg(
            np.arccos(
                np.clip(
                    np.einsum("ij,ij->i", v[2], vh)
                    / (np.linalg.norm(v[2], axis=1) * np.linalg.norm(vh, axis=1)),
                    -1,
                    1,
                )
            )
        )
        bad = np.where((volume <= 1) | (hvolume >= 0) | (angle <= 80) | (angle >= 140))[0]
        mol = Chem.Mol(self.ligand)
        for i, point in enumerate(xyz[self.lig_all] * 10):
            mol.GetConformer().SetAtomPosition(i, point)
        Chem.RemoveStereochemistry(mol)
        Chem.AssignStereochemistryFrom3D(mol, replaceExistingTags=True)
        key = Chem.MolToInchiKey(Chem.RemoveHs(mol))
        pocket = xyz[self.pocket_ca].mean(axis=0)
        lig_xyz = nearest_molecular_image(xyz[self.lig_heavy], pocket, box)
        rotation, a, b, ca_rmsd = fit_to_reference(xyz[self.ca], self.original_ca)
        rmsd = (
            np.sqrt(
                np.mean(np.sum(((lig_xyz - a) @ rotation + b - self.original_ligand) ** 2, axis=1))
            )
            * 10
        )
        rotation, a, b, _ = fit_to_reference(xyz[self.ca], self.start[self.ca])
        initial_lig = nearest_molecular_image(
            self.start[self.lig_heavy], self.start[self.pocket_ca].mean(axis=0), box
        )
        initial_rmsd = (
            np.sqrt(np.mean(np.sum(((lig_xyz - a) @ rotation + b - initial_lig) ** 2, axis=1))) * 10
        )
        distances = np.linalg.norm(
            minimum_image(xyz[self.protein_heavy][:, None] - xyz[self.lig_heavy][None, :], box),
            axis=2,
        )
        contacts = sorted(
            {self.heavy_residues[i] for i in np.where(distances.min(axis=1) <= 0.4)[0]}
        )
        if self.initial_contacts is None:
            self.initial_contacts = set(contacts)
        # 围绕受体中心展开膜头基；两层检查不受整个体系平移影响。
        pz = minimum_image(xyz[self.phosphorus] - xyz[self.ca].mean(axis=0), box)[:, 2]
        upper, lower = pz[pz > 0], pz[pz <= 0]
        gap = float(upper.mean() - lower.mean()) if len(upper) and len(lower) else 0.0
        geometry = bool(
            cn.min() > 1.1
            and cn.max() < 1.6
            and ss.min() > 1.8
            and ss.max() < 2.3
            and not len(bad)
            and key == self.key
            and gap > 2
            and min(len(upper), len(lower)) > 0.3 * len(pz)
        )
        return dict(
            geometry_passed=geometry,
            peptide_CN_min_A=float(cn.min()),
            peptide_CN_max_A=float(cn.max()),
            disulfide_min_A=float(ss.min()),
            disulfide_max_A=float(ss.max()),
            alpha_volume_min_A3=float(volume.min()),
            alpha_hydrogen_volume_max_A3=float(hvolume.max()),
            alpha_angle_min_deg=float(angle.min()),
            alpha_angle_max_deg=float(angle.max()),
            failed_alpha_model_ids=[self.alpha_ids[i] for i in bad],
            ligand_coordinate_inchikey=key,
            ligand_coordinate_identity_matches=key == self.key,
            native_CA_RMSD_to_original_A=ca_rmsd * 10,
            ligand_RMSD_to_original_pose_A=float(rmsd),
            ligand_RMSD_to_replica_start_A=float(initial_rmsd),
            pocket_center_distance_A=float(np.linalg.norm(lig_xyz.mean(axis=0) - pocket) * 10),
            closest_receptor_heavy_A=float(distances.min() * 10),
            contacts_4A=contacts,
            initial_contact_recall=len(set(contacts) & self.initial_contacts)
            / max(1, len(self.initial_contacts)),
            area_nm2=float(box[0] * box[1]),
            volume_nm3=float(np.prod(box)),
            phosphate_layer_separation_nm=gap,
            leaflet_counts=[len(upper), len(lower)],
            box_nm=box.tolist(),
        )

"""在隔离uv环境中为指定立体配体生成Sage/NAGL参数并保存原子映射。"""

import argparse
import importlib.metadata
import json
import shutil
from pathlib import Path

import numpy as np
import yaml


def save(path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("construct", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reuse-charged-molecule", type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    source = args.construct / "ligand_oriented.sdf"
    assert (
        json.loads((args.construct / "manifest.json").read_text())["status"]
        == "construct_geometry_passed"
    )
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), args.output / "source_snapshot.py")
    shutil.copyfile(args.config, args.output / "config.yaml")
    shutil.copyfile(source, args.output / "input_ligand.sdf")
    result = {
        "status": "parameterizing",
        "source": str(source),
        "source_bytes": source.stat().st_size,
        "source_mtime_ns": source.stat().st_mtime_ns,
    }
    save(args.output / "manifest.json", result)
    try:
        import openmm
        from openff.toolkit import ForceField as SMIRNOFFForceField
        from openff.toolkit import Molecule
        from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper
        from openmm import unit
        from openmm.app import ForceField, HBonds, NoCutoff, PDBFile
        from openmmforcefields.generators import SMIRNOFFTemplateGenerator
        from rdkit import Chem

        rd_mol = Chem.SDMolSupplier(str(source), removeHs=False)[0]
        expected_key = Chem.MolToInchiKey(Chem.MolFromSmiles(cfg["ligand_smiles"]))
        assert rd_mol is not None and Chem.MolToInchiKey(Chem.RemoveHs(rd_mol)) == expected_key
        rd_mol = Chem.AddHs(rd_mol, addCoords=True)
        coordinate_copy = Chem.Mol(rd_mol)
        Chem.RemoveStereochemistry(coordinate_copy)
        Chem.AssignStereochemistryFrom3D(coordinate_copy, replaceExistingTags=True)
        assert Chem.MolToInchiKey(Chem.RemoveHs(coordinate_copy)) == expected_key
        mol = Molecule.from_rdkit(rd_mol, allow_undefined_stereo=False, hydrogens_are_explicit=True)
        mol.name = "LIG"
        for i, atom in enumerate(mol.atoms):
            atom.name = f"{atom.symbol}{i + 1}"
        smirnoff = SMIRNOFFForceField(cfg["ligand_forcefield"])
        charge_model = smirnoff["NAGLCharges"].model_file
        if args.reuse_charged_molecule:
            charged = Molecule.from_json(args.reuse_charged_molecule.read_text())
            assert charged.to_smiles(mapped=True) == mol.to_smiles(mapped=True)
            assert [a.name for a in charged.atoms] == [a.name for a in mol.atoms]
            mol.partial_charges = charged.partial_charges
            result["reused_charged_molecule"] = str(args.reuse_charged_molecule)
        else:
            mol.assign_partial_charges(charge_model, toolkit_registry=NAGLToolkitWrapper())
        charges = np.array(mol.partial_charges.m)
        assert len(charges) == mol.n_atoms and np.isfinite(charges).all()
        assert abs(charges.sum() - float(mol.total_charge.m)) < 1e-5
        assert mol.n_atoms == rd_mol.GetNumAtoms()
        atom_records = [
            {
                "index": i,
                "name": a.name,
                "atomic_number": a.atomic_number,
                "charge_e": float(charges[i]),
                "position_A": rd_mol.GetConformer().GetPositions()[i].tolist(),
            }
            for i, a in enumerate(mol.atoms)
        ]
        save(args.output / "atom_parameters.json", atom_records)
        (args.output / "openff_molecule.json").write_text(mol.to_json(), encoding="utf-8")
        generator = SMIRNOFFTemplateGenerator(
            molecules=mol,
            forcefield=cfg["ligand_forcefield"],
            cache=str(args.output / "template_cache.json"),
        )
        ff = ForceField(*cfg["forcefields"])
        ff.registerTemplateGenerator(generator.generator)
        topology = mol.to_topology().to_openmm()
        system = ff.createSystem(topology, nonbondedMethod=NoCutoff, constraints=HBonds)
        assert system.getNumParticles() == mol.n_atoms
        nb = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
        actual_q = np.array(
            [
                nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                for i in range(mol.n_atoms)
            ]
        )
        assert np.allclose(actual_q, charges, atol=1e-7)
        positions = unit.Quantity(rd_mol.GetConformer().GetPositions() / 10, unit.nanometer)
        integrator = openmm.VerletIntegrator(0.001)
        context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
        context.setPositions(positions)
        state = context.getState(getEnergy=True, getForces=True)
        energy = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        assert np.isfinite(energy)
        assert np.isfinite(
            state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        ).all()
        with (args.output / "ligand.pdb").open("w", encoding="utf-8") as h:
            PDBFile.writeFile(topology, positions, h)
        (args.output / "ligand_system.xml").write_text(
            openmm.XmlSerializer.serialize(system), encoding="utf-8"
        )
        mol.to_file(str(args.output / "ligand.sdf"), file_format="sdf")
        versions = {
            n: importlib.metadata.version(n)
            for n in (
                "openmm",
                "rdkit",
                "openff-toolkit",
                "openff-nagl",
                "openff-nagl-models",
                "openforcefields",
                "openmmforcefields",
                "openff-interchange",
                "torch",
            )
        }
        result.update(
            status="ligand_parameters_passed",
            inchikey=expected_key,
            atoms=mol.n_atoms,
            formal_charge_e=float(mol.total_charge.m),
            partial_charge_sum_e=float(charges.sum()),
            charge_range_e=[float(charges.min()), float(charges.max())],
            ligand_energy_kJ_mol=energy,
            forcefield=cfg["ligand_forcefield"],
            charge_model=charge_model,
            packages=versions,
            ligand_id=cfg.get("ligand_id", "GLPG1205"),
            scope="指定目录图的Sage/NAGL参数；中性状态为计算条件，非优势微观状态或活性验证",
        )
        save(args.output / "manifest.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as error:
        result.update(
            status="failed_parameterization", error_type=type(error).__name__, error=str(error)
        )
        save(args.output / "manifest.json", result)
        raise


if __name__ == "__main__":
    main()

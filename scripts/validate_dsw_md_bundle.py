"""在迁移目录中回读三个真实System/State；检查原子、盒、立体和全部运行指针。"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import openmm
from openmm import unit

from translock_screening.md_pilot import PilotDiagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    os.chdir(bundle)
    manifest = json.loads((bundle / "bundle_manifest.json").read_text())
    rows = []
    for obj in manifest["objects"]:
        warmup = bundle / obj["warmup"]
        restart = warmup / "restart_v1"
        state = openmm.XmlSerializer.deserialize((restart / "state.xml").read_text())
        system = openmm.XmlSerializer.deserialize((restart / "system.xml").read_text())
        xyz = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        diagnostic = PilotDiagnostics(warmup, xyz).calculate(xyz, box)
        assert diagnostic["geometry_passed"] and dict(state.getParameters())["k"] == 0
        assert system.getNumParticles() == len(xyz)
        defaults = np.array(
            [v.value_in_unit(unit.nanometer) for v in system.getDefaultPeriodicBoxVectors()]
        )
        assert np.allclose(defaults, box)
        rows.append(
            dict(
                id=obj["id"],
                atoms=len(xyz),
                geometry_passed=True,
                ligand_identity=diagnostic["ligand_coordinate_identity_matches"],
                source_time_ps=state.getTime().value_in_unit(unit.picosecond),
                restraint_k=0,
            )
        )
    result = dict(
        status="passed_three_relocated_system_states_without_integration",
        objects=rows,
        cloud_cuda_tested=False,
        steps_executed=0,
    )
    (bundle / "local_readback.json").write_text(json.dumps(result, indent=2) + "\n")
    manifest["status"] = "local_relocation_readback_passed_cloud_cuda_pending"
    manifest["local_readback"] = "local_readback.json"
    (bundle / "bundle_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

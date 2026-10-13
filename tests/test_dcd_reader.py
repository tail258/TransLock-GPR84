"""重点核查DCD截断与变步长时间，避免静默缩短轨迹。"""

import struct

import numpy as np
import pytest

from translock_screening.dcd_reader import read_openmm_dcd, time_for_step


def test_time_uses_changed_timestep_instead_of_header_dt():
    stages = [{"time_ps": 10, "timestep_fs": 1}, {"time_ps": 20, "timestep_fs": 2}]
    assert time_for_step(5000, stages) == 5
    assert time_for_step(15000, stages) == 20
    assert time_for_step(20000, stages) == 30
    with pytest.raises(ValueError, match="超过"):
        time_for_step(20001, stages)


def test_reader_detects_partial_frame_and_maps_angstroms_to_nm(tmp_path):
    def record(data):
        return struct.pack("<i", len(data)) + data + struct.pack("<i", len(data))

    header = bytearray(84)
    header[:4] = b"CORD"
    struct.pack_into("<3i", header, 4, 1, 5000, 5000)
    struct.pack_into("<i", header, 44, 1)
    data = record(header) + record(struct.pack("<i", 0)) + record(struct.pack("<i", 2))
    data += record(struct.pack("<6d", 100, 0, 110, 0, 0, 120))
    for values in ([1, 2], [3, 4], [5, 6]):
        data += record(np.array(values, dtype="<f4").tobytes())
    path = tmp_path / "frame.dcd"
    path.write_bytes(data)
    frame = list(read_openmm_dcd(path))[0]
    assert frame.step == 5000
    assert np.allclose(frame.positions_nm, [[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    assert np.allclose(frame.box_vectors_nm, np.diag([10, 11, 12]))
    path.write_bytes(data[:-1])
    with pytest.raises(ValueError, match="截断"):
        list(read_openmm_dcd(path))


def test_actual_openmm_writer_first_frame_is_first_step(tmp_path):
    import openmm
    from openmm import unit
    from openmm.app import DCDFile, Topology, element

    topology = Topology()
    residue = topology.addResidue("ALA", topology.addChain())
    topology.addAtom("CA", element.carbon, residue)
    topology.addAtom("CB", element.carbon, residue)
    topology.setPeriodicBoxVectors(
        [openmm.Vec3(10, 0, 0), openmm.Vec3(0, 11, 0), openmm.Vec3(0, 0, 12)] * unit.nanometer
    )
    path = tmp_path / "actual_writer.dcd"
    coordinates = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]) * unit.nanometer
    with path.open("wb") as handle:
        writer = DCDFile(handle, topology, 0.001 * unit.picosecond, firstStep=5000, interval=5000)
        writer.writeModel(coordinates)
        writer.writeModel(coordinates)
    frames = list(read_openmm_dcd(path))
    assert [f.step for f in frames] == [5000, 10000]
    assert np.allclose(frames[-1].positions_nm, coordinates.value_in_unit(unit.nanometer))

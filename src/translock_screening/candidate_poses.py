"""目录提名的条件性立体输入；只补结构，不修改提名与效力标签。"""

from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions


def conditional_stereoisomers(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or Chem.GetFormalCharge(mol) != 0 or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("要求登记的单片段中性结构")
    if len([s for s in Chem.FindPotentialStereo(mol) if str(s.specified) == "Unspecified"]) > 5:
        raise ValueError("未指定立体超出本次小样枚举预算")
    original = Chem.MolToSmiles(mol, isomericSmiles=False)
    states = []
    options = StereoEnumerationOptions(onlyUnassigned=True, unique=True, maxIsomers=0)
    for state in EnumerateStereoisomers(mol, options=options):
        if Chem.MolToSmiles(state, isomericSmiles=False) != original:
            raise ValueError("枚举改变连接图")
        states.append(Chem.MolToSmiles(state, isomericSmiles=True))
    if not states or len(states) > 32:
        raise ValueError("条件性立体枚举数量异常")
    return sorted(set(states))


def first_acceptable_pose(audits, minimum_distance):
    for row in audits:
        if row["valid"] and row["minimum_receptor_distance_A"] >= minimum_distance:
            return row["rank"]
    return None

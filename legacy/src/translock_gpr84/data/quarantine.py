"""依据来源身份生成隔离视图，不覆盖、修正或删除原数据。"""

import pandas as pd


def quarantine_reasons(frame: pd.DataFrame, policy: dict) -> pd.Series:
    routes = {
        "activity_id": "activity_id",
        "record_id": "record_id",
        "molecule_chembl_id": "chembl_id",
        "parent_molecule_chembl_id": "chembl_id",
        "standardized_inchikey": "historical_key",
    }
    if not (set(routes) | {"source_records"}) & set(frame.columns):
        raise ValueError("表格缺少可审计的身份字段")
    reasons = []
    for row in frame.to_dict("records"):
        hits = []
        for item in policy["records"]:
            for source in [row, *(row.get("source_records") or [])]:
                for field, policy_field in routes.items():
                    targets = {str(item[policy_field])}
                    if field == "standardized_inchikey":
                        targets.add(str(item["author_structure_key"]))
                    if str(source.get(field)) in targets:
                        hits.append(f"{item['compound']}:{field}")
        reasons.append(";".join(sorted(set(hits))))
    return pd.Series(reasons, index=frame.index, dtype=str)


def quarantine_mask(frame: pd.DataFrame, policy: dict) -> pd.Series:
    return quarantine_reasons(frame, policy).ne("")


def partition_frame(frame: pd.DataFrame, policy: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """保留原索引以冻结划分；只在隔离副本添加原因。"""
    reasons = quarantine_reasons(frame, policy)
    retained = frame.loc[reasons == ""].copy()
    held = frame.loc[reasons != ""].copy()
    held["quarantine_reason"] = reasons.loc[reasons != ""]
    return retained, held

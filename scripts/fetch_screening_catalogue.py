#!/usr/bin/env python3
"""下载 ChEMBL 37 固定前 100000 条小分子及 GPR84 已知测量身份。"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import Request, urlopen

BASE = "https://www.ebi.ac.uk/chembl/api/data/"
VERSION = "ChEMBL_37"
TARGET = "CHEMBL3714079"
PAGE_SIZE = 1000
FULL_LIMIT = 100_000
TIMEOUT = 90
ATTEMPTS = 5
STRUCTURE_FIELDS = [
    "molecule_chembl_id",
    "molecule_type",
    "canonical_smiles",
    "standard_inchi",
    "standard_inchi_key",
    "gpr84_label",
]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def save_manifest(root: Path, manifest: dict) -> None:
    # 清单更新先写临时文件，再原子替换；原始 API 页永不覆盖。
    temp = root / "manifest.json.tmp"
    temp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    temp.replace(root / "manifest.json")


def request_json(root: Path, manifest: dict, url: str, filename: str) -> dict:
    path = root / filename
    if path.exists():
        raise FileExistsError(f"原始响应已存在：{path}")
    for attempt in range(1, ATTEMPTS + 1):
        event = {"url": url, "raw_file": filename, "attempt": attempt, "started_at": utc_now()}
        manifest["requests"].append(event)
        save_manifest(root, manifest)
        try:
            req = Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "TransLock-GPR84-screening-catalogue/1.0",
                },
            )
            with urlopen(req, timeout=TIMEOUT) as response:
                raw = response.read()
                event.update(
                    http_status=response.status, response_url=response.geturl(), bytes=len(raw)
                )
            with path.open("xb") as stream:
                stream.write(raw)
            event.update(status="saved", finished_at=utc_now())
            save_manifest(root, manifest)
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("API JSON 顶层不是对象")
            return payload
        except (HTTPError, URLError, TimeoutError) as exc:
            event.update(
                status="request_failed",
                finished_at=utc_now(),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            if isinstance(exc, HTTPError):
                event["http_status"] = exc.code
            save_manifest(root, manifest)
            if (
                attempt == ATTEMPTS
                or isinstance(exc, HTTPError)
                and exc.code != 429
                and exc.code < 500
            ):
                raise
            time.sleep(min(2 ** (attempt - 1), 20))
    raise RuntimeError("请求未完成")


def url_for(resource: str, **params: object) -> str:
    return BASE + resource + ".json?" + urlencode(params)


def check_page(
    payload: dict, key: str, offset: int, limit: int, prior_total: int | None, order_by: str | None
) -> tuple[list[dict], int]:
    records = payload.get(key)
    meta = payload.get("page_meta")
    if (
        not isinstance(records, list)
        or not isinstance(meta, dict)
        or not records
        or len(records) > limit
    ):
        raise ValueError("API 页缺少记录、元数据或超出请求页大小")
    if meta.get("offset") != offset or meta.get("limit") != limit:
        raise ValueError("API offset/limit 与请求不一致")
    total = meta.get("total_count")
    if (
        not isinstance(total, int)
        or total < offset + len(records)
        or prior_total is not None
        and total != prior_total
    ):
        raise ValueError("API total_count 缺失、无效或分页期间改变")
    next_link = meta.get("next")
    if offset + len(records) < total:
        if not isinstance(next_link, str):
            raise ValueError("API 提前结束")
        query = parse_qs(urlsplit(next_link).query)
        if query.get("offset") != [str(offset + len(records))]:
            raise ValueError("API 下一页 offset 不连续")
        if order_by and query.get("order_by") != [order_by]:
            raise ValueError("API 下一页未保留排序")
    elif next_link is not None:
        raise ValueError("API 末页仍有下一页")
    return records, total


def verify_status(status: dict) -> None:
    if status.get("status") != "UP" or status.get("chembl_db_version") != VERSION:
        raise ValueError(
            f"数据库状态/版本异常：{status.get('status')} {status.get('chembl_db_version')}"
        )


def catalogue(root: Path, manifest: dict, limit: int) -> None:
    total = None
    last_id = ""
    seen: set[str] = set()
    missing = 0
    with (root / "structures.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=STRUCTURE_FIELDS, lineterminator="\n")
        writer.writeheader()
        for offset in range(0, limit, PAGE_SIZE):
            count = min(PAGE_SIZE, limit - offset)
            url = url_for(
                "molecule",
                molecule_type="Small molecule",
                order_by="molecule_chembl_id",
                limit=count,
                offset=offset,
                only="molecule_chembl_id,molecule_type,molecule_structures",
            )
            filename = f"molecules_page_{offset // PAGE_SIZE + 1:03d}.json"
            payload = request_json(root, manifest, url, filename)
            items, total = check_page(
                payload, "molecules", offset, count, total, "molecule_chembl_id"
            )
            if len(items) != count:
                raise ValueError("目录页少于请求数量")
            ids = [item.get("molecule_chembl_id") for item in items]
            if (
                any(not isinstance(x, str) or not x for x in ids)
                or ids != sorted(ids)
                or ids[0] <= last_id
            ):
                raise ValueError("API 排序/分子 ID 无效")
            if any(item.get("molecule_type") != "Small molecule" for item in items):
                raise ValueError("API 返回非 Small molecule")
            if any(x in seen for x in ids):
                raise ValueError("目录跨页重复 ID")
            seen.update(ids)
            last_id = ids[-1]
            for item in items:
                structures = item.get("molecule_structures") or {}
                if not isinstance(structures, dict):
                    raise ValueError("结构字段不是对象")
                if not structures.get("canonical_smiles"):
                    missing += 1
                writer.writerow(
                    {
                        "molecule_chembl_id": item["molecule_chembl_id"],
                        "molecule_type": item["molecule_type"],
                        "canonical_smiles": structures.get("canonical_smiles") or "",
                        "standard_inchi": structures.get("standard_inchi") or "",
                        "standard_inchi_key": structures.get("standard_inchi_key") or "",
                        "gpr84_label": "unassessed",
                    }
                )
            manifest["pages"].append(
                {
                    "kind": "catalogue",
                    "raw_file": filename,
                    "url": url,
                    "offset": offset,
                    "count": len(items),
                    "first_id": ids[0],
                    "last_id": ids[-1],
                    "total_count": total,
                }
            )
            manifest["statistics"].update(
                downloaded_records=len(seen),
                missing_canonical_smiles=missing,
                first_id=manifest["pages"][0]["first_id"],
                last_id=last_id,
                catalogue_total_count=total,
            )
            save_manifest(root, manifest)
            print(f"catalogue {len(seen)}/{limit}", flush=True)
    if len(seen) != limit:
        raise ValueError("目录条数与冻结目标不一致")


def known_target(root: Path, manifest: dict) -> None:
    total = None
    offset = 0
    last_activity = -1
    activity_ids: set[int] = set()
    molecule_ids: set[str] = set()
    with (root / "known_target_activities.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["activity_id", "molecule_chembl_id"], lineterminator="\n"
        )
        writer.writeheader()
        while total is None or offset < total:
            count = PAGE_SIZE
            url = url_for(
                "activity",
                target_chembl_id=TARGET,
                order_by="activity_id",
                limit=count,
                offset=offset,
                only="activity_id,molecule_chembl_id",
            )
            filename = f"known_target_activities_page_{offset // PAGE_SIZE + 1:03d}.json"
            payload = request_json(root, manifest, url, filename)
            items, total = check_page(payload, "activities", offset, count, total, "activity_id")
            ids = [item.get("activity_id") for item in items]
            if (
                any(type(x) is not int for x in ids)
                or ids != sorted(ids)
                or ids[0] <= last_activity
                or any(x in activity_ids for x in ids)
            ):
                raise ValueError("activity_id 排序/唯一性错误")
            if any(
                not isinstance(item.get("molecule_chembl_id"), str)
                or not item["molecule_chembl_id"]
                for item in items
            ):
                raise ValueError("活动缺少分子 ID")
            last_activity = ids[-1]
            activity_ids.update(ids)
            for item in items:
                molecule_ids.add(item["molecule_chembl_id"])
                writer.writerow(
                    {
                        "activity_id": item["activity_id"],
                        "molecule_chembl_id": item["molecule_chembl_id"],
                    }
                )
            manifest["pages"].append(
                {
                    "kind": "known_target_activity",
                    "raw_file": filename,
                    "url": url,
                    "offset": offset,
                    "count": len(items),
                    "total_count": total,
                }
            )
            offset += len(items)
            manifest["statistics"].update(
                known_target_activity_count=offset,
                known_target_unique_molecule_ids=len(molecule_ids),
                known_target_total_count=total,
            )
            save_manifest(root, manifest)
            print(f"known target activity {offset}/{total}", flush=True)
    if len(activity_ids) != total:
        raise ValueError("已知靶点 activity 清单不完整")
    # API __in 批查每批 80 个 ID，保留原响应；可能存在不再可取结构的分子。
    found: set[str] = set()
    ordered = sorted(molecule_ids)
    with (root / "known_target_molecules.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=STRUCTURE_FIELDS, lineterminator="\n")
        writer.writeheader()
        for index, start in enumerate(range(0, len(ordered), 80), 1):
            chunk = ordered[start : start + 80]
            url = url_for(
                "molecule",
                molecule_chembl_id__in=",".join(chunk),
                limit=100,
                offset=0,
                only="molecule_chembl_id,molecule_type,molecule_structures",
            )
            filename = f"known_target_molecules_batch_{index:03d}.json"
            payload = request_json(root, manifest, url, filename)
            items, batch_total = check_page(payload, "molecules", 0, 100, None, None)
            if batch_total != len(items) or batch_total > len(chunk):
                raise ValueError("已知靶点分子批次未完整返回")
            for item in items:
                mol_id = item.get("molecule_chembl_id")
                if mol_id not in chunk or mol_id in found:
                    raise ValueError("已知靶点分子批次 ID 非法或重复")
                found.add(mol_id)
                structures = item.get("molecule_structures") or {}
                if not isinstance(structures, dict):
                    raise ValueError("已知靶点结构字段不是对象")
                writer.writerow(
                    {
                        "molecule_chembl_id": mol_id,
                        "molecule_type": item.get("molecule_type") or "",
                        "canonical_smiles": structures.get("canonical_smiles") or "",
                        "standard_inchi": structures.get("standard_inchi") or "",
                        "standard_inchi_key": structures.get("standard_inchi_key") or "",
                        "gpr84_label": "known_target_activity_any_endpoint",
                    }
                )
            manifest["pages"].append(
                {
                    "kind": "known_target_molecule",
                    "raw_file": filename,
                    "url": url,
                    "requested_ids": chunk,
                    "returned_count": len(items),
                    "total_count": batch_total,
                }
            )
            manifest["statistics"]["known_target_structure_records"] = len(found)
            save_manifest(root, manifest)
            print(f"known target molecule {len(found)}/{len(ordered)}", flush=True)
    missing_ids = sorted(molecule_ids - found)
    manifest["statistics"]["known_target_unresolved_molecule_ids"] = missing_ids
    if missing_ids:
        raise ValueError("已知靶点分子批查未返回全部 ID；保留完整 activity 清单供后续排除")


def run(output: Path, probe: bool = False) -> int:
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    limit = 1000 if probe else FULL_LIMIT
    manifest = {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_now(),
        "scope": (
            "固定 molecule_chembl_id 升序的 ChEMBL Small molecule 前 N 条；"
            "非随机抽样；未知活性不作阴性"
        ),
        "selection": {
            "molecule_type": "Small molecule",
            "order_by": "molecule_chembl_id",
            "offset_start": 0,
            "requested_limit": limit,
            "page_size": PAGE_SIZE,
            "local_resorting": False,
            "probe": probe,
        },
        "known_target_identity": {
            "target_chembl_id": TARGET,
            "policy": "全部 target activity ID 及分子图身份排除，不限效力、机制或端点",
            "probe_includes_known_target": False if probe else True,
        },
        "licence": {
            "name": "Creative Commons Attribution-ShareAlike 3.0 Unported",
            "spdx": "CC-BY-SA-3.0",
            "url": "https://creativecommons.org/licenses/by-sa/3.0/",
            "source": "https://www.ebi.ac.uk/chembldb/",
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
            "argv": sys.argv,
            "timeout_seconds": TIMEOUT,
            "max_attempts": ATTEMPTS,
        },
        "requests": [],
        "pages": [],
        "statistics": {},
    }
    save_manifest(output, manifest)
    try:
        shutil.copyfile(Path(__file__), output / "source_snapshot.py")
        before = request_json(output, manifest, BASE + "status.json", "api_status_before.json")
        verify_status(before)
        manifest["database_version"] = before["chembl_db_version"]
        manifest["database_release_date"] = before.get("chembl_release_date")
        catalogue(output, manifest, limit)
        if not probe:
            known_target(output, manifest)
        after = request_json(output, manifest, BASE + "status.json", "api_status_after.json")
        verify_status(after)
        if before.get("chembl_db_version") != after.get("chembl_db_version") or before.get(
            "chembl_release_date"
        ) != after.get("chembl_release_date"):
            raise ValueError("前后 API 数据库版本不一致")
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        print(f"下载失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    finally:
        manifest.update(finished_at=utc_now(), elapsed_seconds=round(time.monotonic() - started, 3))
        save_manifest(output, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output": str(output),
                "statistics": manifest["statistics"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if manifest["status"] == "complete" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="尚不存在的新目录")
    parser.add_argument(
        "--probe", action="store_true", help="仅取前 1000 条且不取全靶点活动，用于真实 API 探针"
    )
    args = parser.parse_args()
    return run(args.output.expanduser().resolve(), args.probe)


if __name__ == "__main__":
    raise SystemExit(main())

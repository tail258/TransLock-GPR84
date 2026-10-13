"""获取已核准的固定官方权重，保存来源与文件大小，不计算哈希。"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    url = "https://zenodo.org/api/records/15460715/files/chemeleon_mp.pt/content"
    record = {
        "url": url,
        "doi": "10.5281/zenodo.15460715",
        "license": "MIT",
        "expected_bytes": 34859448,
        "started_utc": datetime.now(UTC).isoformat(),
        "hashes_computed": False,
        "status": "running",
    }
    try:
        with urlopen(
            Request(url, headers={"User-Agent": "TransLock-GPR84/1.0"}), timeout=60
        ) as response:
            payload = response.read()
            record["http_status"] = response.status
            record["resolved_url"] = response.url
        record["actual_bytes"] = len(payload)
        if len(payload) != record["expected_bytes"]:
            raise ValueError("文件大小与官方固定记录不符")
        with (args.output / "chemeleon_mp.pt").open("xb") as stream:
            stream.write(payload)
        record["status"] = "download_complete_not_yet_inference_verified"
    except Exception as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["completed_utc"] = datetime.now(UTC).isoformat()
        (args.output / "provenance.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()

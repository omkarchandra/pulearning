#!/usr/bin/env python3
"""Record the Slurm execution identity and hashes for a completed fingerprint report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fingerprint-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    assert report["phase"] == "section 30 sorted-content fingerprints"
    assert {Path(path).name for path in report["fingerprints"]} == {
        "view.parquet", "aggregation_qc.parquet", "dependencies.parquet"
    }
    receipt = {
        "status": "complete",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "fingerprint_report": str(args.report.resolve()),
        "fingerprint_report_sha256": sha256(args.report),
        "fingerprint_script": str(args.fingerprint_script.resolve()),
        "fingerprint_script_sha256": sha256(args.fingerprint_script),
        "receipt_script": str(Path(__file__).resolve()),
        "receipt_script_sha256": sha256(Path(__file__).resolve()),
    }
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

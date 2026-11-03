#!/usr/bin/env python3
"""Verify the unmatched rE2G target-ID audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


RE2G_SHA256 = "074820992b59f46273e650aabe6d8c96156e43b12ee5213f56a56f5815298d12"
TSS_SHA256 = "24c2a0180f6ab7feb2e6d9e6008a8555694cd9574d16abaa30f70f06c7cc328b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    assert report["status"] == "complete"
    assert report["contract"]["gene_id_column"] == "column1"
    assert report["contract"]["source_biotype_column"] == "column3"
    assert report["inputs"]["re2g"]["sha256"] == RE2G_SHA256
    assert report["inputs"]["tss"]["sha256"] == TSS_SHA256
    assert report["integrity"]["target_ids_with_multiple_source_biotypes"] == 0
    rows = report["by_source_biotype"]
    for row in rows:
        assert row["matched_gencode_v50"] + row["unmatched_gencode_v50"] == row["target_ids"]
        assert row["unmatched_fraction"] == round(
            row["unmatched_gencode_v50"] / row["target_ids"], 8
        )
    summary = report["summary"]
    assert sum(row["target_ids"] for row in rows) == summary["distinct_target_ids"]
    assert sum(row["matched_gencode_v50"] for row in rows) == summary["matched_gencode_v50"]
    assert sum(row["unmatched_gencode_v50"] for row in rows) == summary["unmatched_gencode_v50"]
    assert report["integrity"]["distinct_target_ids"] == summary["distinct_target_ids"]

    verification = {
        "status": "verified",
        "report": str(args.report.resolve()),
        "report_sha256": sha256(args.report),
        "verifier_sha256": sha256(Path(__file__).resolve()),
        "checks": {
            "canonical_column_binding": True,
            "pinned_inputs": True,
            "one_source_biotype_per_target": True,
            "biotype_partitions_summary": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

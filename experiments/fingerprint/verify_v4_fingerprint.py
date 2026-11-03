#!/usr/bin/env python3
"""Verify the schema-bound v4 fingerprint report and parallel-build fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fixture-report", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")

    report = json.loads(args.report.read_text(encoding="utf-8"))
    fixture = json.loads(args.fixture_report.read_text(encoding="utf-8"))
    manifest = json.loads(args.bundle_manifest.read_text(encoding="utf-8"))
    assert report["phase"] == "section 30 sorted-content fingerprints"
    assert report["hash_format"].startswith("store logical hash: schema header")
    assert report["settings"]["threads"] == 4
    assert report["settings"]["memory_limit"] == "12GB"
    assert report["settings"]["batch_size"] > 0
    assert len(report["fingerprints"]) == 1
    artifact_path, artifact = next(iter(report["fingerprints"].items()))
    assert Path(artifact_path).name == "dependencies.parquet"
    expected = manifest["outputs"]["dependencies"]
    assert artifact["rows"] == manifest["dependency_rows"] == expected["rows"]
    assert artifact["artifact_sha256"] == expected["sha256"]
    assert artifact["original_file_order_logical_sha256"] == expected["logical_sha256"]
    expected_schema = [
        [field["name"], field["type"], field["nullable"]]
        for field in expected["schema"]
    ]
    assert artifact["schema"] == expected_schema
    assert set(artifact["key_columns"]) == set(expected["keys"])
    assert len(artifact["sorted_content_sha256"]) == 64
    int(artifact["sorted_content_sha256"], 16)

    grouped = {}
    for path, record in fixture["fingerprints"].items():
        seed = "seed0" if "bundle_seed0" in path else "seed1" if "bundle_seed1" in path else None
        assert seed is not None
        grouped.setdefault(Path(path).name, {})[seed] = record
    for name in ("view.parquet", "aggregation_qc.parquet", "dependencies.parquet"):
        assert set(grouped[name]) == {"seed0", "seed1"}
        left, right = grouped[name]["seed0"], grouped[name]["seed1"]
        assert left["sorted_content_sha256"] == right["sorted_content_sha256"]
        assert left["rows"] == right["rows"]
        assert left["schema"] == right["schema"]
        assert left["key_columns"] == right["key_columns"]

    verification = {
        "status": "verified",
        "report": str(args.report.resolve()),
        "report_sha256": sha256(args.report),
        "fixture_report_sha256": sha256(args.fixture_report),
        "bundle_manifest_sha256": sha256(args.bundle_manifest),
        "verifier_sha256": sha256(Path(__file__).resolve()),
        "dependencies": {
            "rows": artifact["rows"],
            "sorted_content_sha256": artifact["sorted_content_sha256"],
            "original_file_order_logical_sha256": artifact[
                "original_file_order_logical_sha256"
            ],
            "artifact_sha256": artifact["artifact_sha256"],
        },
        "parallel_fixture_pairs_equal": 3,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Verify the weighted bundle, replay, infinity fixture, and sorted fingerprints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


FROZEN_LAMBDA_BP = 68_152.45706
V4_MANIFEST_SHA256 = "21cccb8082d9c5217602507601209098d30eec71ce99eadf24e2ec24aef69d2f"
V4_LINKS_SHA256 = "b313dbcf01bdb4c4d04d5a83cc0e82df7c5dc3877988a80007aa7143361aeb5a"
V4_DEPENDENCIES_SORTED_SHA256 = "9cff3122dd69ce86eb75e5e790c39ed50d12526a89ddcf359d93090dce2447a4"
HELD_INPUTS = {
    "catalog": "92762255f4e9d1e665a58a2e1bca7f5420d29e3bffa883a528707c10e8ab350d",
    "genes": "4b4e35f858a4845d4d339573b22608405e0ffab0f20bb1d2aac8847a517c21fe",
    "measurements": "e4de76ef18ff1ee8921fb3de4a8dbf2682a557641de1e6367690ad21035212b3",
    "partitions": "0248ed02c299ee9a999a5388c5e947de008be3df0a144dfcf0732f10192562ef",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-report", type=Path, required=True)
    parser.add_argument("--fingerprint-report", type=Path, required=True)
    parser.add_argument("--fingerprint-receipt", type=Path, required=True)
    parser.add_argument("--source-fingerprint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    build = json.loads(args.build_report.read_text(encoding="utf-8"))
    fingerprints = json.loads(args.fingerprint_report.read_text(encoding="utf-8"))
    fingerprint_receipt = json.loads(args.fingerprint_receipt.read_text(encoding="utf-8"))
    source_fingerprint = json.loads(args.source_fingerprint.read_text(encoding="utf-8"))

    assert build["phase"] == "section 39.2 preregistered weighted CRE bundle"
    assert build["status"] == "complete"
    assert build["weight"] == {
        "amplitude_used": False,
        "distance_definition": "signed cCRE-midpoint-minus-TSS distance from v4 link table",
        "formula": "score_v4 * exp(-abs(distance_to_tss_bp) / lambda_bp)",
        "lambda_bp": FROZEN_LAMBDA_BP,
        "w_at_zero": 1.0,
    }
    source_manifest = Path(build["source_bundle"]["path"]) / "manifest.json"
    assert sha256(source_manifest) == build["source_bundle"]["manifest_sha256"] == V4_MANIFEST_SHA256
    assert build["inputs"]["candidate_links"]["sha256"] == V4_LINKS_SHA256
    assert build["inputs"]["retained_links"]["sha256"] == V4_LINKS_SHA256
    for name, digest in HELD_INPUTS.items():
        assert build["inputs"][name]["sha256"] == digest
        assert sha256(Path(build["inputs"][name]["path"])) == digest

    weighted_links = Path(build["weighted_links"]["path"])
    assert sha256(weighted_links) == build["weighted_links"]["sha256"]
    assert build["weighted_links"]["sha256"] != V4_LINKS_SHA256
    assert build["weighted_links"]["rows"] == 11_289_221
    assert all(build["weighted_links"][key] == 0 for key in (
        "duplicate_pairs", "wrong_method_rows", "fallback_rows", "bad_score_rows",
        "source_only_pairs", "weighted_only_pairs", "score_formula_mismatches",
    ))
    assert build["weighted_links"]["lambda_bp"] == FROZEN_LAMBDA_BP

    bundle = Path(build["weighted_bundle"]["path"])
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert sha256(manifest_path) == build["weighted_bundle"]["manifest_sha256"]
    assert manifest["inputs"]["candidate_links"]["sha256"] == sha256(weighted_links)
    assert manifest["inputs"]["retained_links"]["sha256"] == sha256(weighted_links)
    for name, digest in HELD_INPUTS.items():
        assert manifest["inputs"][name]["sha256"] == digest
    for key in ("view_rows", "qc_rows", "dependency_rows"):
        assert build["weighted_bundle"][key] == build["source_bundle"][key]
    assert build["weighted_bundle"]["dependency_rows"] == 225_784_420

    fixture_path = Path(build["lambda_infinity_fixture"]["report"])
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert sha256(fixture_path) == build["lambda_infinity_fixture"]["report_sha256"]
    assert fixture["status"] == "passed"
    assert fixture["score_float32_bits_equal"] is True
    assert set(fixture["artifacts"]) == {
        "view.parquet", "aggregation_qc.parquet", "dependencies.parquet"
    }
    for record in fixture["artifacts"].values():
        assert record["logical_values_equal"] is True
        assert record["arrow_tables_equal"] is True
        assert record["floating_value_bit_hashes_equal"] is True

    replay_path = Path(build["full_replay"]["report"])
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    assert sha256(replay_path) == build["full_replay"]["report_sha256"]
    assert replay["status"] == "valid" and replay["replay_mode"] == "full"
    assert set(replay["artifacts"]) == {"view", "aggregation_qc", "dependencies"}
    for name, record in replay["artifacts"].items():
        assert record["rows"] == manifest["outputs"][name]["rows"]
        assert record["logical_sha256"] == manifest["outputs"][name]["logical_sha256"]

    assert fingerprints["phase"] == "section 30 sorted-content fingerprints"
    assert fingerprints["settings"]["threads"] == 4
    assert fingerprints["settings"]["memory_limit"] == "12GB"
    assert fingerprints["settings"]["batch_size"] == 10_000
    assert sha256(args.fingerprint_report) == fingerprint_receipt["fingerprint_report_sha256"]
    assert fingerprint_receipt["status"] == "complete"
    assert fingerprint_receipt["slurm_job_id"]
    by_name = {Path(path).name: record for path, record in fingerprints["fingerprints"].items()}
    assert set(by_name) == {"view.parquet", "aggregation_qc.parquet", "dependencies.parquet"}
    for filename, record in by_name.items():
        output_name = Path(filename).stem
        expected = manifest["outputs"][output_name]
        assert record["rows"] == expected["rows"]
        assert record["artifact_sha256"] == expected["sha256"]
        assert record["original_file_order_logical_sha256"] == expected["logical_sha256"]
        assert record["original_manifest_sha256"] == sha256(manifest_path)
        assert record["unique_full_key_verified"] is True
        assert len(record["sorted_content_sha256"]) == 64
        int(record["sorted_content_sha256"], 16)

    source_records = {
        Path(path).name: record for path, record in source_fingerprint["fingerprints"].items()
    }
    assert source_records["dependencies.parquet"]["sorted_content_sha256"] == V4_DEPENDENCIES_SORTED_SHA256
    assert by_name["dependencies.parquet"]["sorted_content_sha256"] == V4_DEPENDENCIES_SORTED_SHA256
    assert manifest["outputs"]["view"]["sha256"] != json.loads(source_manifest.read_text())["outputs"]["view"]["sha256"]
    assert manifest["outputs"]["aggregation_qc"]["sha256"] != json.loads(source_manifest.read_text())["outputs"]["aggregation_qc"]["sha256"]

    build_script = Path(build["provenance"]["build_script"])
    design = Path(build["provenance"]["design"])
    assert sha256(build_script) == build["provenance"]["build_script_sha256"]
    assert sha256(design) == build["provenance"]["design_sha256"]
    assert build["provenance"]["slurm_job_id"]

    receipt = {
        "status": "verified",
        "build_job_id": build["provenance"]["slurm_job_id"],
        "fingerprint_job_id": fingerprint_receipt["slurm_job_id"],
        "verification_job_id": os.environ.get("SLURM_JOB_ID"),
        "build_report_sha256": sha256(args.build_report),
        "weighted_manifest_sha256": sha256(manifest_path),
        "full_replay_sha256": sha256(replay_path),
        "lambda_infinity_fixture_sha256": sha256(fixture_path),
        "fingerprint_report_sha256": sha256(args.fingerprint_report),
        "verifier_sha256": sha256(Path(__file__).resolve()),
        "fingerprints": {
            name: record["sorted_content_sha256"] for name, record in sorted(by_name.items())
        },
        "checks": {
            "frozen_weight": True,
            "same_admitted_link_pairs": True,
            "held_v4_inputs": True,
            "lambda_infinity_bitwise_fixture": True,
            "independent_full_replay": True,
            "schema_bound_sorted_fingerprints": True,
            "dependency_multiset_unchanged_from_v4": True,
            "weighted_values_differ_from_v4": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

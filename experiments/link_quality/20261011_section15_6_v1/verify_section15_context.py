#!/usr/bin/env python3
"""Verify the canonical-contract Section 15 context revalidation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


RE2G_SHA256 = "074820992b59f46273e650aabe6d8c96156e43b12ee5213f56a56f5815298d12"
TSS_SHA256 = "24c2a0180f6ab7feb2e6d9e6008a8555694cd9574d16abaa30f70f06c7cc328b"
LINKS_SHA256 = "b313dbcf01bdb4c4d04d5a83cc0e82df7c5dc3877988a80007aa7143361aeb5a"
EXPECTED_BINS = {"0-10kb", "10-50kb", "50-100kb", ">100kb"}


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
    assert report["bundle"]["status"] == "complete"
    assert report["contract"]["gene_id_column"] == "column1"
    assert report["contract"]["region_column"] == "column5"
    assert report["inputs"]["re2g"]["sha256"] == RE2G_SHA256
    assert report["inputs"]["tss"]["sha256"] == TSS_SHA256
    assert report["inputs"]["retained_links"]["sha256"] == LINKS_SHA256
    assert report["integrity"]["pairs_with_multiple_regions"] == 0
    assert report["integrity"]["malformed_region_or_distance"] == 0
    assert report["integrity"]["chromosome_mismatches"] == 0

    distance = report["protein_coding_pair_distance"]
    bins = distance["bins"]
    assert set(bins) == EXPECTED_BINS
    assert sum(row["pairs"] for row in bins.values()) == distance["total_pairs"]
    for row in bins.values():
        assert row["fraction"] == round(row["pairs"] / distance["total_pairs"], 8)
    assert abs(sum(row["fraction"] for row in bins.values()) - 1.0) <= 2e-8

    assert report["old_reference_coverage"]["lncRNA_proximity_coverage"] == {
        "linked": 34774,
        "total": 34866,
    }
    assert report["old_reference_coverage"]["documented_lncRNA_rE2G_coverage"] == {
        "linked": 1434,
        "total": 34866,
    }
    semantics = report["contract"]["bundle_tier_semantics"]
    assert semantics["admitted_method"] == "tier1_distance_window_unconstrained"
    assert "100,000" in semantics["primary_rule"]
    assert "is_lncRNA_fallback false" in semantics["nonfallback"]
    assert "rejects every is_lncRNA_fallback=true row" in semantics["excluded_fallback"]
    coverage = (
        report["lncRNA_proximity_coverage"],
        report["lncRNA_rE2G_coverage"],
        report["protein_coding_rE2G_coverage"],
    )
    for row in coverage:
        assert 0 <= row["linked"] <= row["total"]
        assert row["fraction"] == round(row["linked"] / row["total"], 8)

    verification = {
        "status": "verified",
        "report": str(args.report.resolve()),
        "report_sha256": sha256(args.report),
        "verifier_sha256": sha256(Path(__file__).resolve()),
        "checks": {
            "canonical_column_binding": True,
            "pinned_inputs": True,
            "pair_integrity": True,
            "distance_bins_partition_pairs": True,
            "coverage_arithmetic": True,
            "bundle_tier_semantics": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

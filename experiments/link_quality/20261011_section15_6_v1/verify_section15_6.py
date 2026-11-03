#!/usr/bin/env python3
"""Independently check Section 15.6 report partitions and arithmetic."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


EXPECTED_BINS = {"0-10kb", "10-50kb", "50-100kb"}
EXPECTED_CLASSES = {
    "PLS", "pELS", "dELS", "CA-CTCF", "CA-H3K4me3", "CA-TF", "CA", "TF"
}
EXPECTED_RE2G_SHA256 = "074820992b59f46273e650aabe6d8c96156e43b12ee5213f56a56f5815298d12"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def close_at_six_decimals(observed, numerator: int, denominator: int) -> bool:
    return observed == round(numerator / denominator, 6)


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
    assert report["inputs"]["re2g"]["support_column"] == "column7 (biosample)"
    assert report["inputs"]["re2g"]["sha256"] == EXPECTED_RE2G_SHA256
    assert report["integrity"]["missing_ccre_class"] == 0
    assert report["integrity"]["invalid_distance_bin"] == 0
    assert report["integrity"]["duplicate_proximity_pairs"] == 0

    thresholds = {int(k): value for k, value in report["thresholds"].items()}
    expected_k = set(thresholds)
    tables = {
        "distance": report["precision_by_distance_bin"],
        "class": report["precision_by_ccre_class"],
        "distance_and_class": report["precision_by_distance_and_ccre_class"],
    }
    checks = {}
    for name, rows in tables.items():
        grouped = defaultdict(list)
        for row in rows:
            grouped[int(row["k"])].append(row)
            assert 0 <= row["supported_pairs"] <= row["proximity_pairs"]
            assert close_at_six_decimals(
                row["micro_precision"], row["supported_pairs"], row["proximity_pairs"]
            )
            assert 0.0 <= row["macro_precision_median"] <= 1.0
            assert 0.0 <= row["macro_precision_iqr"][0] <= row["macro_precision_iqr"][1] <= 1.0
        assert set(grouped) == expected_k
        for k, group in grouped.items():
            assert sum(row["proximity_pairs"] for row in group) == thresholds[k]["proximity_pairs"]
            assert sum(row["supported_pairs"] for row in group) == thresholds[k]["supported_pairs"]
            if name == "distance":
                assert {row["distance_bin"] for row in group} == EXPECTED_BINS
            elif name == "class":
                assert {row["ccre_class"] for row in group} == EXPECTED_CLASSES
            else:
                observed = {(row["distance_bin"], row["ccre_class"]) for row in group}
                assert len(observed) == len(group)
                assert {distance for distance, _ in observed} == EXPECTED_BINS
                assert {ccre for _, ccre in observed} == EXPECTED_CLASSES
                assert observed <= {
                    (distance, ccre)
                    for distance in EXPECTED_BINS
                    for ccre in EXPECTED_CLASSES
                }
        checks[f"{name}_partitions_overall"] = True

    for k, row in thresholds.items():
        assert row["proximity_pairs"] == report["integrity"]["analysis_proximity_pairs"]
        assert 0 <= row["supported_pairs"] <= row["proximity_pairs"]
        assert close_at_six_decimals(
            row["micro_precision"], row["supported_pairs"], row["proximity_pairs"]
        )
        assert row["section15_reference_note"].startswith("Diagnostic only")
        checks[f"threshold_{k}_arithmetic"] = True

    verification = {
        "status": "verified",
        "report": str(args.report.resolve()),
        "report_sha256": sha256(args.report),
        "verifier_sha256": sha256(Path(__file__).resolve()),
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

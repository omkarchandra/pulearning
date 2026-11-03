#!/usr/bin/env python3
"""Bind completed sorted-fingerprint evidence to the retained full CRE certificate."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "certificate", "dependency-report", "values-qc-report", "golden-report", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    certificate = json.loads(args.certificate.read_text())
    if certificate["status"] != "valid" or certificate["replay_mode"] != "full":
        raise ValueError("the retained certification must be a successful full replay")
    golden = json.loads(args.golden_report.read_text())
    if golden["verdict"] != "verified" or not golden["store_encoding_golden_replay"] or not golden["all_three_fixture_artifacts_match"]:
        raise ValueError("canonical-encoding/fixture evidence did not pass")
    if golden["report_hashes"][str(args.values_qc_report)] != sha(args.values_qc_report):
        raise ValueError("golden verifier did not bind the supplied v4 values/QC report")
    combined = {}
    reports = {}
    for report_path in (args.dependency_report, args.values_qc_report):
        report = json.loads(report_path.read_text())
        reports[str(report_path)] = {"sha256": sha(report_path), "source_snapshot_checksum": report["snapshot_checksum"],
                                     "settings": report["settings"], "peak_rss_kib": report["peak_rss_kib"],
                                     "elapsed_seconds": report["elapsed_seconds_total"]}
        for name, record in report["fingerprints"].items():
            artifact = Path(name)
            if artifact.parent.resolve() != args.bundle.resolve() or artifact.stem in combined:
                raise ValueError("fingerprint report points at another bundle or duplicates an artifact")
            original = manifest["outputs"][artifact.stem]
            physical = sha(artifact)
            if (record["artifact_sha256"] != physical or original["sha256"] != physical or
                    record["original_manifest_sha256"] != sha(manifest_path) or
                    record["original_file_order_logical_sha256"] != original["logical_sha256"] or
                    certificate["artifacts"][artifact.stem]["logical_sha256"] != original["logical_sha256"] or
                    certificate["artifacts"][artifact.stem]["rows"] != original["rows"] or
                    certificate["replay"][artifact.stem] != original["rows"] or record["rows"] != original["rows"]):
                raise ValueError("original artifact, full certificate and sorted sidecar do not agree")
            if set(record["key_columns"]) != set(original["keys"]):
                raise ValueError("sorted fingerprint does not include the full manifest key")
            schema = [[field["name"], field["type"], field["nullable"]] for field in original["schema"]]
            if schema != record["schema"]:
                raise ValueError("sorted fingerprint uses a different schema encoding")
            if artifact.stem == "dependencies" and any(field[1] in ("float", "double") for field in schema):
                raise ValueError("the original dependency encoder may not be reused for floating schemas")
            combined[artifact.stem] = {"rows": record["rows"], "artifact_sha256": physical,
                                       "file_order_logical_sha256": original["logical_sha256"],
                                       "sorted_content_sha256": record["sorted_content_sha256"],
                                       "sorted_key_columns": record["key_columns"], "schema": schema}
    if set(combined) != {"view", "aggregation_qc", "dependencies"}:
        raise ValueError("sidecar must bind all three certified artifacts")
    result = {"status": "verified", "fingerprint_scheme": "cre-sorted-content-v1",
              "bundle": str(args.bundle.resolve()), "bundle_manifest_sha256": sha(manifest_path),
              "artifacts": combined,
              "full_replay_certificate": {"path": str(args.certificate), "sha256": sha(args.certificate),
                                          "status": "valid", "replay_mode": "full"},
              "fingerprint_reports": reports,
              "golden_encoding_verifier": {"path": str(args.golden_report), "sha256": sha(args.golden_report)},
              "original_build_settings": manifest.get("build_settings", "not recorded in original manifest"),
              "interpretation": "sorted content is auxiliary reproducibility evidence; original file-order hashes and full semantic replay certificate remain bound to retained bytes",
              "dependency_encoder_scope": "integer/string/null schema; signed-zero float-cache correction is inapplicable",
              "finalizer_code_sha256": sha(__file__), "numbers_reread_from_completed_artifacts": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "bundle_manifest_sha256": result["bundle_manifest_sha256"],
                      "artifact_hashes": {name: {k: value[k] for k in ("rows", "file_order_logical_sha256", "sorted_content_sha256")}
                                          for name, value in combined.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()

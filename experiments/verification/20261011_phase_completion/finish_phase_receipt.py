#!/usr/bin/env python3
"""Read completed phase evidence and emit a create-only, checksum-bound receipt."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project
    records = {}

    def read_json(relative):
        path = root / relative
        records[relative] = {"sha256": sha(path), "bytes": path.stat().st_size}
        return json.loads(path.read_text())

    snapshots = {}
    expected = {
        "20261011_eval_repair_v1": "7b2a0e9abfccbd3271b47bdddca277e8da4bb1c8cf5b4552d96b12b36644ae10",
        "20261011_eval_repair_v2": "0aac9e946ba831a2c4b91de7c307b7b06c4618ae7120ff419d59e4a7de69394b",
        "20261011_fingerprint_v3": "d314ab74fbf465042b7afe798e6549ac38860a8fb64e1108296aa030678a9aac",
        "20261011_eval_hardened_v4": "ce7400b82cbbbf5db6e1bdc6bd5423e43aa24f3e4bad4a5444bd8a8aff60e640",
        "20261011_eval_hardened_v5": "ecfe0944c44a5b2c70c450c1f8a56eb677ea7a3a6266316531125c191b69c615",
    }
    for name, digest in expected.items():
        source = root / "releases" / name
        manifest = source / "SOURCE_SHA256SUMS"
        if sha(manifest) != digest:
            raise ValueError(f"source-manifest checksum changed: {name}")
        count = 0
        for line in manifest.read_text().splitlines():
            file_digest, relative = line.split("  ", 1)
            if sha(source / relative) != file_digest:
                raise ValueError(f"changed frozen source file: {name}/{relative}")
            count += 1
        snapshots[name] = {"source_manifest_sha256": digest, "files_verified": count}

    comparison = read_json("experiments/20261011_eval_repair_v2/selector_comparison.json")
    verdict = read_json("experiments/verification/20261011_eval_repair_v2/verdict.json")
    if (verdict["verdict"] != "verified" or verdict["run_files_checked"] != 20 or
            verdict["folds_checked"] != 100 or
            verdict["snapshot_checksum"] != expected["20261011_eval_repair_v2"] or
            verdict["summary_sha256"] != records["experiments/20261011_eval_repair_v2/selector_comparison.json"]["sha256"]):
        raise ValueError("comparison does not match its completed independent verdict")
    if not all(row["P"] == row["N"] for row in verdict["fold_counts"]):
        raise ValueError("not all comparison folds use equal class counts")
    for citation in comparison["cited_run_files"]:
        if sha(citation["path"]) != citation["sha256"]:
            raise ValueError("a cited run artifact changed after verification")
    isolation = read_json("experiments/20261011_eval_hardened_v5/v2_process_isolation_confirmation.json")
    if not isolation["separate_python_process_per_array_task_verified"] or isolation["distinct_completed_batch_jobs"] != 20:
        raise ValueError("v2 process-isolation confirmation is incomplete")

    tests_path = root / "experiments/20261011_eval_hardened_v5/native_tests.xml"
    cases = list(ET.parse(tests_path).iter("testcase"))
    skips = sum(case.find("skipped") is not None for case in cases)
    if len(cases) != 69 or skips != 1 or any(case.find("failure") is not None or case.find("error") is not None for case in cases):
        raise ValueError("final v5 native tests did not pass 68 tests with one skip")
    records[str(tests_path.relative_to(root))] = {"sha256": sha(tests_path), "bytes": tests_path.stat().st_size}

    sidecar = read_json("experiments/fingerprint/20261011_final_sidecar/v4_fingerprint_sidecar.json")
    if sidecar["status"] != "verified" or set(sidecar["artifacts"]) != {"view", "aggregation_qc", "dependencies"}:
        raise ValueError("the full three-artifact fingerprint sidecar is incomplete")
    expected_rows = {"view": 549600, "aggregation_qc": 549600, "dependencies": 225784420}
    for name, rows in expected_rows.items():
        if sidecar["artifacts"][name]["rows"] != rows:
            raise ValueError("final fingerprint has the wrong certified row counts")
    if sha(Path(sidecar["bundle"]) / "manifest.json") != sidecar["bundle_manifest_sha256"]:
        raise ValueError("bundle manifest changed after sidecar acceptance")
    for report_path, binding in sidecar["fingerprint_reports"].items():
        if sha(report_path) != binding["sha256"]:
            raise ValueError("a sorted-fingerprint report changed after sidecar acceptance")
    for binding in (sidecar["full_replay_certificate"], sidecar["golden_encoding_verifier"]):
        if sha(binding["path"]) != binding["sha256"]:
            raise ValueError("a bound certification/golden artifact changed")
    if sidecar["full_replay_certificate"]["replay_mode"] != "full":
        raise ValueError("sampled replay cannot certify this receipt")

    jobs = [9462517, 9462630, 9462989, 9463331, 9463604, 9463944, 9463950,
            9464148, 9464657, 9464776, 9464781]
    fields = ["JobIDRaw", "JobID", "JobName", "State", "ExitCode", "Elapsed", "MaxRSS", "AllocCPUS", "NodeList"]
    result = subprocess.run(["sacct", "-j", ",".join(map(str, jobs)), "--parsable2", "--noheader",
                             "--format=JobIDRaw,JobID,JobName%80,State,ExitCode,Elapsed,MaxRSS,AllocCPUS,NodeList"],
                            check=True, text=True, capture_output=True)
    job_records = [dict(zip(fields, line.split("|")[:len(fields)])) for line in result.stdout.splitlines()]
    receipt = {
        "status": "mechanical_phase_verified",
        "scientific_scoring": "held",
        "scientific_inputs_needed": ["official versioned curated disease-only DisGeNET data/panels",
                                     "primary task choice", "exact referenced BMC paper"],
        "source_snapshots": snapshots,
        "bound_completed_evidence": records,
        "selector_comparison": {"snapshot_checksum": comparison["snapshot_checksum"],
                                "evaluation_n": comparison["evaluation_n"],
                                "evaluation_positives": comparison["evaluation_positives"],
                                "selectors": comparison["selectors"], "scope": "one-disease shuffled-label references"},
        "final_native_tests": {"passed": 68, "skipped": 1, "failed": 0, "job_id": 9464776},
        "cre_sorted_fingerprints": sidecar["artifacts"],
        "slurm_job_records": job_records,
        "retained_failed_followup_job_id": 9464657,
        "comparison_processes": "twenty standalone Python processes, one per completed array task",
        "review_before_dependency_completion": {"url": "https://claude.ai/chat/9216c204-4aba-4266-8b4f-c5cedb3913dd",
                                               "last_read_message": 36,
                                               "decision": "§34 comparison confirmed; only full dependency fingerprint remained"},
        "receipt_job_id": os.environ.get("SLURM_JOB_ID"),
        "receipt_code_sha256": sha(__file__),
        "numbers_reread_from_completed_artifacts": True,
    }
    with args.output.open("x") as handle:
        json.dump(receipt, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({"status": receipt["status"], "receipt_sha256": sha(args.output),
                      "scientific_scoring": receipt["scientific_scoring"],
                      "dependency_rows": sidecar["artifacts"]["dependencies"]["rows"],
                      "dependency_sorted_sha256": sidecar["artifacts"]["dependencies"]["sorted_content_sha256"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

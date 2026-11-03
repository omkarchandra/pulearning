#!/usr/bin/env python3
"""Recheck retained bytes, source seals and v2/v7 roundoff at phase closeout."""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from sklearn.metrics import average_precision_score


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def threshold_group_ap(labels, scores):
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")
    labels, scores = labels[order], scores[order]
    ends = np.flatnonzero(np.r_[scores[:-1] != scores[1:], True])
    positives = np.cumsum(labels)[ends]
    return float(np.dot(np.diff(np.r_[0, positives]), positives / (ends + 1)) / labels.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    export = project / "experiments/20261011_paired_phase_evidence"
    manifest = json.loads((export / "import_manifest.json").read_text())
    assert sha(export / "evidence.tar.gz") == manifest["archive_sha256"]
    for entry in manifest["files"]:
        path = export / "files" / entry["path"]
        assert sha(path) == entry["sha256"], entry["path"]
        assert path.stat().st_size == entry["bytes"], entry["path"]

    root = export / "files"
    source = root / "releases/20261011_eval_paired_v7"
    source_lines = (source / "SOURCE_SHA256SUMS").read_text().splitlines()
    current_differences = []
    for line in source_lines:
        digest, relative = line.split("  ", 1)
        assert sha(source / relative) == digest, relative
        if sha(project / relative) != digest:
            current_differences.append(relative)

    def retained_path(quest_path):
        for prefix in ("/home/mes7956/projects/pulearning/", "/gpfs/projects/p32724/pulearning/"):
            if str(quest_path).startswith(prefix):
                return root / str(quest_path)[len(prefix):]
        raise ValueError(f"unexpected retained source path: {quest_path}")

    result = root / "experiments/20261011_eval_paired_v7"
    summary = json.loads((result / "selector_comparison.json").read_text())
    verdict = json.loads((result / "independent_artifact_verdict.json").read_text())
    assert verdict["summary_sha256"] == sha(result / "selector_comparison.json")
    assert verdict["snapshot_checksum"] == summary["snapshot_checksum"] == sha(source / "SOURCE_SHA256SUMS")
    assert verdict["verdict"] == "verified" and verdict["run_files_checked"] == 20 and verdict["folds_checked"] == 100
    assert verdict["verifier_code_sha256"] == sha(source / "scripts/verify_sealed_comparison_artifacts.py")
    xml = result / "native_tests.xml"
    assert sha(xml) == verdict["native_test_xml_sha256"]
    cases = list(ET.parse(xml).iter("testcase"))
    assert len(cases) == 69 and sum(case.find("skipped") is not None for case in cases) == 1
    assert not any(case.find("failure") is not None or case.find("error") is not None for case in cases)

    old_root = project / "experiments/20261011_completed_evidence_v2/files/experiments"
    old_result = old_root / "20261011_eval_repair_v2"
    old_verdict = json.loads((old_root / "verification/20261011_eval_repair_v2/verdict.json").read_text())
    old_aps = {row["task_index"]: row["replayed_auprc"] for row in old_verdict["replayed_auprcs"]}
    native_verdict_ap_delta = max(abs(row["replayed_auprc"] - old_aps[row["task_index"]])
                                  for row in verdict["replayed_auprcs"])
    worker_ids, comparisons = [], []
    changed_payloads, max_score_delta = 0, 0.0
    for citation in summary["cited_run_files"]:
        run_path = retained_path(citation["path"])
        assert sha(run_path) == citation["sha256"]
        run = json.loads(run_path.read_text())
        task, execution = run["task"], run["execution_provenance"]
        task_index = task["task_index"]
        assert execution["slurm_array_job_id"] == "9469050" and execution["slurm_array_task_id"] == str(task_index)
        assert execution["script_sha256"] == sha(source / "scripts/run_sealed_selector_comparison.py")
        worker_ids.append(execution["slurm_job_id"])
        before = result / f"source_check_{execution['slurm_job_id']}_{task_index}.log"
        after = result / f"source_check_after_{execution['slurm_job_id']}_{task_index}.log"
        assert before.read_bytes() == after.read_bytes()
        assert len(before.read_text().splitlines()) == 175
        assert all(line.endswith(": OK") for line in before.read_text().splitlines())
        prepared_path = retained_path(task["label_manifest"])
        prepared = json.loads(prepared_path.read_text())
        label_records = {record["seal_key"]: record for record in prepared["diseases"][summary["disease"]]}
        labels, new_scores, old_scores = [], [], []
        paths = sorted((run_path.parent / "ledger").glob("sealed__*.json"))
        assert len(paths) == 5
        for path in paths:
            old_path = old_result / f"results/task_{task_index}/ledger" / path.name
            new, old = json.loads(path.read_text()), json.loads(old_path.read_text())
            assert new["seal_key"] == old["seal_key"]
            record = label_records[new["seal_key"]]
            test_path = prepared_path.parent / record["test_path"]
            assert sha(test_path) == record["test_sha256"]
            test = json.loads(test_path.read_text())
            true = set(test["positives"])
            assert len(new["predictions"]) == len(old["predictions"])
            for row, previous in zip(new["predictions"], old["predictions"]):
                assert row["gene_id"] == previous["gene_id"]
                labels.append(int(row["gene_id"] in true))
                new_scores.append(row["score"])
                old_scores.append(previous["score"])
                max_score_delta = max(max_score_delta, abs(row["score"] - previous["score"]))
            changed_payloads += path.read_bytes() != old_path.read_bytes()
        old_ap, new_ap = average_precision_score(labels, old_scores), average_precision_score(labels, new_scores)
        old_group_ap, new_group_ap = threshold_group_ap(labels, old_scores), threshold_group_ap(labels, new_scores)
        reported = run["diseases"][summary["disease"]]["pooled_auprc"]
        assert round(old_ap, 6) == round(new_ap, 6) == round(old_group_ap, 6) == round(new_group_ap, 6) == reported
        comparisons.append({"task_index": task_index, "sklearn_ap_delta": float(new_ap - old_ap),
                            "threshold_group_ap_delta": new_group_ap - old_group_ap})
    assert len(set(worker_ids)) == 20
    for job_id in ("9468835", "9469060"):
        assert (result / f"source_check_{job_id}_single.log").read_bytes() == (result / f"source_check_after_{job_id}_single.log").read_bytes()

    report = {
        "status": "closeout_bytes_and_bindings_verified",
        "scope": "retention/binding and numerical-roundoff check for the accepted v7 phase",
        "script_sha256": sha(__file__),
        "export_payloads_verified": len(manifest["files"]),
        "export_archive_sha256": manifest["archive_sha256"],
        "import_manifest_sha256": sha(export / "import_manifest.json"),
        "v7_source_manifest_sha256": sha(source / "SOURCE_SHA256SUMS"),
        "frozen_v7_source_files_verified": len(source_lines),
        "current_source_differences": current_differences,
        "native_tests": {"passed": 68, "skipped": 1, "failed": 0},
        "same_job_source_seal_pairs_checked": 22,
        "distinct_worker_job_ids": worker_ids,
        "comparison_sha256": sha(result / "selector_comparison.json"),
        "verdict_sha256": sha(result / "independent_artifact_verdict.json"),
        "audit_diff_report_sha256": sha(result / "audit_source_diff_from_v2.json"),
        "max_native_verdict_ap_delta_from_v2": native_verdict_ap_delta,
        "max_same_environment_sklearn_ap_delta": max(abs(row["sklearn_ap_delta"]) for row in comparisons),
        "max_same_environment_threshold_group_ap_delta": max(abs(row["threshold_group_ap_delta"]) for row in comparisons),
        "max_prediction_score_delta_from_v2": max_score_delta,
        "prediction_payloads_with_byte_differences_from_v2": changed_payloads,
        "per_run_ap_comparisons": comparisons,
        "all_rounded_run_aps_reproduced": True,
    }
    with args.output.open("x") as handle:
        json.dump(report, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "per_run_ap_comparisons"}, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

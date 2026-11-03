#!/usr/bin/env python3
"""Read-only artifact verifier for the section 32 breast comparison.

Executed after all fitting workers and the summary process finish. This script
does not modify pipeline code or run artifacts; it writes only its verdict.
"""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def genes_hash(genes):
    return hashlib.sha256(json.dumps(sorted(genes), separators=(",", ":"),
                                     ensure_ascii=True).encode("ascii")).hexdigest()


def replay_average_precision(labels, scores):
    """Independent threshold-group AP calculation, including tied scores."""
    y, s = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if len(y) != len(s) or not np.isfinite(s).all() or not y.sum():
        raise ValueError("invalid prediction replay inputs")
    order = np.argsort(-s, kind="stable")
    y, s = y[order], s[order]
    ends = np.flatnonzero(np.r_[s[:-1] != s[1:], True])
    positives = np.cumsum(y)[ends]
    precision = positives / (ends + 1)
    return float(np.dot(np.diff(np.r_[0, positives]), precision) / y.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "summary", "tests-xml", "source-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    summary = json.loads(args.summary.read_text())
    source_checksum = sha(args.source_root / "SOURCE_SHA256SUMS")
    if source_checksum != plan["snapshot_checksum"] or source_checksum != summary["snapshot_checksum"]:
        raise ValueError("source snapshot checksum mismatch")
    source_files = 0
    for line in (args.source_root / "SOURCE_SHA256SUMS").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        if sha(args.source_root / relative) != digest:
            raise ValueError(f"changed source file: {relative}")
        source_files += 1
    suites = ET.parse(args.tests_xml)
    tests = {case.get("name"): case for case in suites.iter("testcase")}
    required = ["test_live_memory_audit_detects_planted_read_inside_real_fit",
                "test_live_file_audit_detects_consolidated_label_read_inside_real_fit",
                "test_gated_prepared_real_run_reaches_fit_through_same_worker",
                "test_postrun_review_replays_predictions_and_rejects_false_scalar",
                "test_full_key_binding_blocks_cross_configuration_test_read"]
    required += [f"test_selector_receives_exact_shuffled_anchors_and_pool_through_sealed_entry[{name}]"
                 for name in ("centroid", "uniform_random", "confounder_matched", "biotype_stratified")]
    for name in required:
        case = tests.get(name)
        if case is None or any(case.find(tag) is not None for tag in ("failure", "error", "skipped")):
            raise ValueError(f"required live-audit/entry-point test did not pass: {name}")
    source = Path(plan["protected_true_label_source"])
    if sha(source) != plan["inputs"]["positives"]["sha256"]:
        raise ValueError("changed true-label review input")
    true = set(json.loads(source.read_text())[plan["disease"]])
    for name, entry in plan["files"].items():
        if sha(entry["path"]) != entry["sha256"]:
            raise ValueError(f"changed prepared input: {name}")
    split = pd.read_parquet(plan["files"]["split"]["path"])
    universe = set(pd.read_parquet(plan["files"]["universe"]["path"]).index)
    by_selector = {}
    seen_tasks = set()
    rows = []
    replayed_aps = []
    cited = summary["cited_run_files"]
    if len(cited) != 20:
        raise ValueError("expected five completed runs of all four selectors")
    for citation in cited:
        path = Path(citation["path"])
        if sha(path) != citation["sha256"]:
            raise ValueError("run file changed after summary")
        run = json.loads(path.read_text())
        task = run["task"]
        if task != plan["tasks"][task["task_index"]] or task != citation["task"]:
            raise ValueError("run task fields differ from the prepared plan or citation")
        seen_tasks.add(task["task_index"])
        disease = run["diseases"][plan["disease"]]
        if (run["plan_sha256"] != sha(args.plan) or run["classifier_config"] != plan["classifier_config"] or
                run["feature_columns"] != plan["feature_columns"] or disease["folds_scored"] != 5 or
                sorted(f["fold"] for f in disease["fold_audits"]) != list(range(5))):
            raise ValueError("run did not use the full declared plan and five folds")
        if run["true_label_map_supplied"] or run["snapshot_checksum"] != source_checksum:
            raise ValueError("wrong fitting scope or source snapshot")
        if run["verification"]["violations"] != 0 or run["verification"]["status"] != "order_verified":
            raise ValueError("run did not pass its label-file audit")
        prepared_path = Path(task["label_manifest"])
        if sha(prepared_path) != task["label_manifest_sha256"]:
            raise ValueError("prepared-label manifest changed")
        prepared = json.loads(prepared_path.read_text())
        events = [json.loads(line) for line in (path.parent / "ledger/seal_ledger.jsonl").read_text().splitlines()]
        seals = {}
        for event in events:
            key = event["seal_key"]
            if event["event"] == "seal":
                if key in seals:
                    raise ValueError("duplicate seal event")
                seals[key] = event
            elif event["event"] == "label_load" and key not in seals:
                raise ValueError("label-load event precedes its own full-key seal")
        label_bindings = {}
        for record in prepared["diseases"][plan["disease"]]:
            for kind in ("train", "test"):
                label_path = prepared_path.parent / record[f"{kind}_path"]
                if sha(label_path) != record[f"{kind}_sha256"]:
                    raise ValueError("prepared label file changed")
                label_bindings[str(label_path.resolve())] = (kind, record["seal_key"])
        if run["verification"]["file_opens_recorded"] != len(run["open_events"]):
            raise ValueError("file-open audit count does not match its retained events")
        for event in run["open_events"]:
            if event["path"] == str(source.resolve()):
                raise ValueError("fit process accessed consolidated true labels")
            if event["path"] not in label_bindings:
                raise ValueError("unexpected label access in retained audit")
            kind, key = label_bindings[event["path"]]
            if kind == "test" and (event["seal_key"] != key or key not in seals or
                                    event["timestamp"] < seals[key]["timestamp"]):
                raise ValueError("test-label open is not authorized by its own full-key seal")
        for fold in disease["fold_audits"]:
            selected_path = Path(fold["selection_inputs_path"])
            if sha(selected_path) != fold["selection_inputs_sha256"]:
                raise ValueError("selection artifact changed")
            selected = json.loads(selected_path.read_text())
            if (genes_hash(selected["candidates"]) != fold["candidate_sha256"] or
                    genes_hash(selected["negatives"]) != fold["negative_sha256"] or
                    len(selected["candidates"]) != fold["candidate_count"] or
                    len(selected["pool"]) != fold["pool_count"]):
                raise ValueError("candidate/negative hashes or counts do not match")
            view = split.loc[split["outer_fold"].eq(fold["fold"])]
            eligible = set(view.loc[view["inner_role"].eq("inner_train") &
                                    view["purged_from_folds"].map(lambda fs: fold["fold"] not in fs), "gene_id"]) & universe
            ordered = sorted(eligible)
            true_train = true & eligible
            permutation_seed = task["seed"] * 1_000_003 + fold["fold"]
            indices = np.random.default_rng(permutation_seed).choice(len(ordered), size=len(true_train), replace=False)
            anchors = {ordered[int(i)] for i in indices}
            pool = eligible - anchors
            if set(selected["positives"]) != anchors or genes_hash(anchors) != fold["positive_sha256"]:
                raise ValueError("anchors are not the declared shuffled positives")
            if set(selected["pool"]) != pool or genes_hash(pool) != fold["pool_sha256"]:
                raise ValueError("candidate pool did not exclude exactly the shuffled positives")
            if genes_hash(true_train) != fold["true_positive_sha256"] or len(pool & true_train) != fold["true_positives_in_pool"]:
                raise ValueError("true training positive hash/count is false")
            if not pool & true_train or set(selected["eligible"]) != eligible:
                raise ValueError("pool excludes all true positives or crosses inner-train scope")
            candidates = sorted(set(selected["candidates"]))
            sampling_seed = task["seed"] * 77_777 + fold["fold"]
            chosen = np.random.default_rng(sampling_seed).choice(len(candidates), size=len(anchors), replace=False)
            negatives = sorted(candidates[int(i)] for i in chosen)
            if (fold["N"] != fold["P"] or fold["P"] != len(anchors) or
                    negatives != selected["negatives"] or not set(candidates) <= pool or
                    fold["permutation_seed"] != permutation_seed or fold["sampling_seed"] != sampling_seed):
                raise ValueError("class counts, membership or recorded seeds do not reproduce")
            params = fold["classifier_parameters"]
            if (params["random_state"] != run["classifier_config"]["project"]["seed"] or
                    params["n_estimators"] != 100 or params["max_depth"] != 6):
                raise ValueError("classifier parameters differ from the declared comparison")
            rows.append({"task_index": task["task_index"], "selector": task["selector"],
                         "run_index": task["run_index"], "fold": fold["fold"],
                         "P": fold["P"], "N": fold["N"], "true_positives_in_pool": fold["true_positives_in_pool"]})
        all_genes, scores = [], []
        for fold in sorted(disease["seals"], key=lambda item: item["fold"]):
            key = fold["seal_key"]
            seal = seals[key]
            prediction_path = path.parent / "ledger" / seal["path"]
            if sha(prediction_path) != seal["sha256"] or fold["sha256"] != seal["sha256"]:
                raise ValueError("sealed prediction changed")
            prediction = json.loads(prediction_path.read_text())
            view = split.loc[split["outer_fold"].eq(fold["fold"])]
            expected_genes = sorted(set(view.loc[view["inner_role"].eq("test"), "gene_id"]) & universe)
            genes = [row["gene_id"] for row in prediction["predictions"]]
            if genes != expected_genes or prediction["seal_key"] != key:
                raise ValueError("prediction payload does not match the test-fold genes/key")
            all_genes.extend(genes)
            scores.extend(row["score"] for row in prediction["predictions"])
        labels = [int(g in true) for g in all_genes]
        ap = replay_average_precision(labels, scores)
        if (round(ap, 6) != disease["pooled_auprc"] or set(all_genes) != universe or
                len(all_genes) != len(universe) or len(all_genes) != disease["evaluation_n"] or
                sum(labels) != disease["evaluation_positives"]):
            raise ValueError("independent AP/pooled-count replay disagrees with the report")
        replayed_aps.append({"task_index": task["task_index"], "selector": task["selector"],
                             "reported_auprc": disease["pooled_auprc"], "replayed_auprc": ap})
        by_selector.setdefault(task["selector"], []).append(disease["pooled_auprc"])
    if seen_tasks != set(range(20)):
        raise ValueError("duplicate or missing run citations")
    for selector, values in by_selector.items():
        reported = summary["selectors"][selector]
        if values != reported["run_auprcs"] or not np.isclose(np.mean(values), reported["mean_auprc"], atol=1e-12, rtol=0):
            raise ValueError("summary AP values do not match the cited completed runs")
    pub = plan["publication_provenance"]
    audit = json.loads(Path(pub["path"]).read_text())
    counts = pd.read_parquet(plan["inputs"]["publication_counts"]["path"])
    if (sha(pub["path"]) != pub["sha256"] or len(counts) != audit["mapped_genes"] or
            sha(plan["inputs"]["publication_counts"]["path"]) != plan["inputs"]["publication_counts"]["sha256"] or
            int(counts["n_publications"].sum()) != audit["distinct_gene_pubmed_pairs"] or
            audit["rows_differing_from_retained_counts"] != 0):
        raise ValueError("publication provenance citation/counts mismatch")
    verdict = {"verdict": "verified", "summary_path": str(args.summary), "summary_sha256": sha(args.summary),
               "snapshot_checksum": source_checksum, "source_files_checked": source_files,
               "run_files_checked": len(cited), "folds_checked": len(rows),
               "anchors_pool_seeds_and_equal_class_counts_reproduced": True,
               "auprc_recomputed_with_independent_threshold_group_algorithm": True,
               "replayed_auprcs": replayed_aps,
               "required_live_canary_tests": required, "native_test_xml_sha256": sha(args.tests_xml),
               "publication_values_checked": {name: audit[name] for name in (
                   "human_source_rows", "distinct_gene_pubmed_pairs", "mapped_genes", "rows_differing_from_retained_counts")},
               "fold_counts": rows, "verifier_code_sha256": sha(__file__),
               "scope": "read-only independent artifact check; no separate model agent used",
               "numbers_reread_after_completion": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(verdict, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({k: v for k, v in verdict.items() if k != "fold_counts"}, indent=2))


if __name__ == "__main__":
    main()

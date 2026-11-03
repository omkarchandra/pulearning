#!/usr/bin/env python3
"""Compute schema-bound sorted-content fingerprints for one weighted CRE bundle."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import duckdb
import pyarrow


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "releases/20261011_fingerprint_v3"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from src.features.contracts import AGGREGATION_QC_SCHEMA, DEPENDENCIES_SCHEMA, VIEW_SCHEMA  # noqa: E402
from src.features.sorted_fingerprint import sorted_logical_parquet_sha256  # noqa: E402
from src.features.store import file_sha256  # noqa: E402


KEY_COLUMNS = {
    "view.parquet": ("gene_id", "feature_id"),
    "aggregation_qc.parquet": ("gene_id", "feature_id"),
    "dependencies.parquet": (
        "feature_id", "entity_type", "entity_id", "selection_role",
        "source_element_id", "dependency_type",
    ),
}
SCHEMAS = {
    "view.parquet": VIEW_SCHEMA,
    "aggregation_qc.parquet": AGGREGATION_QC_SCHEMA,
    "dependencies.parquet": DEPENDENCIES_SCHEMA,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-limit", default="12GB")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=10_000)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.monotonic()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    if args.threads != 4 or args.memory_limit != "12GB" or args.batch_size != 10_000:
        raise SystemExit("fingerprint settings are frozen to 4 threads, 12GB, batch size 10,000")
    manifest_path = args.bundle.resolve() / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("weighted bundle manifest is not complete")
    manifest_sha256 = file_sha256(manifest_path)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)

    fingerprints = {}
    for filename in ("view.parquet", "aggregation_qc.parquet", "dependencies.parquet"):
        artifact = args.bundle.resolve() / filename
        started_artifact = time.monotonic()
        physical = file_sha256(artifact)
        record = manifest["outputs"][artifact.stem]
        if physical != record["sha256"]:
            raise RuntimeError(f"{filename} does not match weighted manifest")
        digest, rows = sorted_logical_parquet_sha256(
            artifact,
            SCHEMAS[filename],
            KEY_COLUMNS[filename],
            scratch_dir=args.scratch_dir,
            memory_limit=args.memory_limit,
            threads=args.threads,
            batch_size=args.batch_size,
        )
        if file_sha256(artifact) != physical or file_sha256(manifest_path) != manifest_sha256:
            raise RuntimeError("bundle changed during fingerprinting")
        fingerprints[str(artifact)] = {
            "sorted_content_sha256": digest,
            "artifact_sha256": physical,
            "original_file_order_logical_sha256": record["logical_sha256"],
            "original_manifest_sha256": manifest_sha256,
            "original_build_settings": manifest.get("build_settings", "not recorded"),
            "schema": [[field.name, str(field.type), field.nullable] for field in SCHEMAS[filename]],
            "rows": rows,
            "key_columns": list(KEY_COLUMNS[filename]),
            "unique_full_key_verified": True,
            "elapsed_seconds": round(time.monotonic() - started_artifact, 1),
        }
        print(f"{filename}: {rows} rows, {digest[:16]}...", flush=True)

    report = {
        "phase": "section 30 sorted-content fingerprints",
        "fingerprint_scheme": "cre-sorted-content-v1",
        "hash_format": "store logical hash: schema header + canonical JSON rows, sorted on unique full key",
        "fingerprints": fingerprints,
        "settings": {
            "threads": args.threads,
            "memory_limit": args.memory_limit,
            "preserve_insertion_order": True,
            "batch_size": args.batch_size,
            "duckdb_version": duckdb.__version__,
            "pyarrow_version": pyarrow.__version__,
        },
        "weighted_manifest_sha256": manifest_sha256,
        "fingerprint_code_release": str(CODE_ROOT),
        "fingerprint_script_sha256": file_sha256(Path(__file__).resolve()),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "elapsed_seconds_total": round(time.monotonic() - started, 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

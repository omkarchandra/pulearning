#!/usr/bin/env python3
"""Build and fully replay the preregistered absolute-distance weighted CRE bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import struct
import sys
import tempfile
import time
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "releases/20261005_quest_cre_phase3_v4"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from src.features.contracts import (  # noqa: E402
    AGGREGATION_QC_SCHEMA,
    DEPENDENCIES_SCHEMA,
    VIEW_SCHEMA,
)
from src.features.cre_views_duckdb import build_permitted_cre_view_from_parquet  # noqa: E402
from src.features.store import file_sha256, logical_parquet_sha256  # noqa: E402
from src.features.validate_cre_view_bundle import validate_cre_view_bundle  # noqa: E402


FROZEN_LAMBDA_BP = 68_152.45706
EXPECTED_METHOD = "tier1_distance_window_unconstrained"
EXPECTED_V4_INPUTS = {
    "catalog": "92762255f4e9d1e665a58a2e1bca7f5420d29e3bffa883a528707c10e8ab350d",
    "genes": "4b4e35f858a4845d4d339573b22608405e0ffab0f20bb1d2aac8847a517c21fe",
    "measurements": "e4de76ef18ff1ee8921fb3de4a8dbf2682a557641de1e6367690ad21035212b3",
    "partitions": "0248ed02c299ee9a999a5388c5e947de008be3df0a144dfcf0732f10192562ef",
    "retained_links": "b313dbcf01bdb4c4d04d5a83cc0e82df7c5dc3877988a80007aa7143361aeb5a",
    "candidate_links": "b313dbcf01bdb4c4d04d5a83cc0e82df7c5dc3877988a80007aa7143361aeb5a",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--lambda-bp", type=float, default=FROZEN_LAMBDA_BP)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="8GB")
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def create_weighted_links(
    source: Path,
    destination: Path,
    *,
    lambda_bp: float,
    scratch_dir: Path,
) -> dict:
    """Create one score-transformed link table without changing pair admission."""
    if destination.exists():
        raise FileExistsError(f"weighted link destination already exists: {destination}")
    if not (math.isinf(lambda_bp) or (math.isfinite(lambda_bp) and lambda_bp > 0)):
        raise ValueError("lambda_bp must be positive finite or positive infinity")
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    source_schema = pq.read_schema(source)
    source_rows = pq.ParquetFile(source).metadata.num_rows
    with tempfile.TemporaryDirectory(prefix=".weighted-links-", dir=destination.parent) as temporary:
        staged = Path(temporary) / destination.name
        connection = duckdb.connect()
        try:
            connection.execute("SET threads=1")
            connection.execute("SET memory_limit='2GB'")
            connection.execute("SET preserve_insertion_order=true")
            connection.execute("SET temp_directory=?", [str(scratch_dir.resolve())])
            if math.isinf(lambda_bp):
                score_expression = "CAST(score AS FLOAT)"
            else:
                score_expression = (
                    "CAST(CAST(score AS DOUBLE) * "
                    f"exp(-abs(CAST(distance_to_tss AS DOUBLE))/{lambda_bp!r}) AS FLOAT)"
                )
            connection.execute(
                f"""
                COPY (
                  SELECT * REPLACE ({score_expression} AS score)
                  FROM read_parquet('{sql_path(source)}')
                ) TO '{sql_path(staged)}'
                  (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
                """
            )
        finally:
            connection.close()
        if pq.read_schema(staged) != source_schema:
            raise RuntimeError("weighted link schema differs from source link schema")
        if pq.ParquetFile(staged).metadata.num_rows != source_rows:
            raise RuntimeError("weighted link row count differs from source")
        os.replace(staged, destination)

    connection = duckdb.connect()
    try:
        connection.execute("SET threads=4")
        check = connection.execute(
            f"""
            SELECT count(*)::BIGINT AS rows,
                   count(*)-count(DISTINCT (cre_id,gene_id))::BIGINT AS duplicate_pairs,
                   count(*) FILTER (WHERE method!='{EXPECTED_METHOD}')::BIGINT AS wrong_method,
                   count(*) FILTER (WHERE coalesce(is_lncRNA_fallback,false))::BIGINT AS fallback,
                   count(*) FILTER (WHERE score IS NULL OR NOT isfinite(score) OR score<=0)::BIGINT AS bad_score,
                   min(CAST(score AS DOUBLE)),max(CAST(score AS DOUBLE))
            FROM read_parquet('{sql_path(destination)}')
            """
        ).fetchone()
        if any(int(value) for value in check[1:5]):
            raise RuntimeError(f"weighted link integrity failure: {check}")
        pair_mismatches = connection.execute(
            f"""
            SELECT
              (SELECT count(*) FROM read_parquet('{sql_path(source)}') s
               ANTI JOIN read_parquet('{sql_path(destination)}') w USING (cre_id,gene_id)),
              (SELECT count(*) FROM read_parquet('{sql_path(destination)}') w
               ANTI JOIN read_parquet('{sql_path(source)}') s USING (cre_id,gene_id)),
              (SELECT count(*) FROM read_parquet('{sql_path(source)}') s
               JOIN read_parquet('{sql_path(destination)}') w USING (cre_id,gene_id)
               WHERE CAST(w.score AS FLOAT) != CAST(
                 CAST(s.score AS DOUBLE) *
                 {('1.0' if math.isinf(lambda_bp) else f'exp(-abs(CAST(s.distance_to_tss AS DOUBLE))/{lambda_bp!r})')}
                 AS FLOAT))
            """
        ).fetchone()
        if any(int(value) for value in pair_mismatches):
            raise RuntimeError(f"weighted link transform mismatch: {pair_mismatches}")
    finally:
        connection.close()
    return {
        "path": str(destination.resolve()),
        "sha256": file_sha256(destination),
        "rows": int(check[0]),
        "duplicate_pairs": int(check[1]),
        "wrong_method_rows": int(check[2]),
        "fallback_rows": int(check[3]),
        "bad_score_rows": int(check[4]),
        "minimum_score": float(check[5]),
        "maximum_score": float(check[6]),
        "source_only_pairs": int(pair_mismatches[0]),
        "weighted_only_pairs": int(pair_mismatches[1]),
        "score_formula_mismatches": int(pair_mismatches[2]),
        "lambda_bp": "infinity" if math.isinf(lambda_bp) else lambda_bp,
    }


def write_fixture_inputs(root: Path) -> dict[str, Path]:
    if root.exists():
        raise FileExistsError(f"fixture directory exists: {root}")
    root.mkdir(parents=True)
    method = [EXPECTED_METHOD] * 4
    links_schema = pa.schema(
        [
            pa.field("cre_id", pa.string()),
            pa.field("gene_id", pa.string()),
            pa.field("score", pa.float32()),
            pa.field("method", pa.string()),
            pa.field("is_lncRNA_fallback", pa.bool_()),
            pa.field("distance_to_tss", pa.int64()),
        ]
    )
    tables = {
        "catalog": pa.table(
            {"cre_id": ["c1", "c2", "c3", "c4"], "chrom": ["chr1"] * 4,
             "start": pa.array([10, 20, 30, 40], type=pa.int64()),
             "end": pa.array([15, 25, 35, 45], type=pa.int64())}
        ),
        "genes": pa.table({"gene_id": ["g1", "g2"]}),
        "links": pa.Table.from_pylist(
            [
                {"cre_id": "c1", "gene_id": "g1", "score": 1.0, "method": method[0],
                 "is_lncRNA_fallback": False, "distance_to_tss": 0},
                {"cre_id": "c2", "gene_id": "g1", "score": 0.75, "method": method[1],
                 "is_lncRNA_fallback": False, "distance_to_tss": -10_000},
                {"cre_id": "c3", "gene_id": "g2", "score": 1.25, "method": method[2],
                 "is_lncRNA_fallback": False, "distance_to_tss": 50_000},
                {"cre_id": "c4", "gene_id": "g2", "score": 2.0, "method": method[3],
                 "is_lncRNA_fallback": False, "distance_to_tss": -100_000},
            ],
            schema=links_schema,
        ),
        "measurements": pa.table(
            {
                "cre_id": ["c1", "c1", "c2", "c2", "c3", "c3", "c4", "c4"],
                "feature_id": ["f1", "f2"] * 4,
                "value": pa.array([2.0, 0.0, 4.0, None, 3.0, 1.0, None, 5.0], type=pa.float64()),
                "reason": ["observed", "observed_zero", "observed", "qc_filtered",
                           "observed", "observed", "unmapped", "observed"],
            }
        ),
        "partitions": pa.table(
            {"entity_type": ["gene", "gene", "cre", "cre", "cre", "cre"],
             "entity_id": ["g1", "g2", "c1", "c2", "c3", "c4"],
             "partition": ["train"] * 6}
        ),
    }
    paths = {}
    for name, table in tables.items():
        path = root / f"{name}.parquet"
        pq.write_table(table, path, compression="zstd")
        paths[name] = path
    return paths


def valid_float_bits(path: Path, schema: pa.Schema) -> dict[str, str]:
    table = pq.read_table(path).cast(schema).combine_chunks()
    result = {}
    for field in schema:
        if not pa.types.is_floating(field.type):
            continue
        array = table[field.name]
        digest = hashlib.sha256()
        for value in array.to_pylist():
            digest.update(b"N" if value is None else b"V")
            if value is not None:
                digest.update(struct.pack(">d" if pa.types.is_float64(field.type) else ">f", value))
        result[field.name] = digest.hexdigest()
    return result


def run_infinity_fixture(run_dir: Path, scratch: Path) -> dict:
    root = run_dir / "lambda_infinity_fixture"
    inputs = write_fixture_inputs(root / "inputs")
    infinity_links = root / "infinity_links.parquet"
    transform = create_weighted_links(
        inputs["links"], infinity_links, lambda_bp=math.inf, scratch_dir=scratch / "transform"
    )
    source_score = pq.read_table(inputs["links"], columns=["score"])["score"].combine_chunks()
    inf_score = pq.read_table(infinity_links, columns=["score"])["score"].combine_chunks()
    source_bits = b"".join(struct.pack(">f", value) for value in source_score.to_pylist())
    inf_bits = b"".join(struct.pack(">f", value) for value in inf_score.to_pylist())
    if source_bits != inf_bits:
        raise RuntimeError("lambda=infinity did not preserve source link score bits")

    outputs = {}
    for name, link_path in (("control", inputs["links"]), ("lambda_infinity", infinity_links)):
        output = root / f"{name}_bundle"
        build_permitted_cre_view_from_parquet(
            inputs["catalog"], inputs["genes"], link_path, link_path,
            inputs["measurements"], inputs["partitions"], output,
            expected_method=EXPECTED_METHOD,
            min_observed_weight_fraction=0.8,
            memory_limit="256MB",
            threads=1,
            scratch_dir=scratch / name,
        )
        outputs[name] = output

    schemas = {
        "view.parquet": VIEW_SCHEMA,
        "aggregation_qc.parquet": AGGREGATION_QC_SCHEMA,
        "dependencies.parquet": DEPENDENCIES_SCHEMA,
    }
    artifacts = {}
    for filename, schema in schemas.items():
        control_path = outputs["control"] / filename
        infinity_path = outputs["lambda_infinity"] / filename
        control_table = pq.read_table(control_path)
        infinity_table = pq.read_table(infinity_path)
        logical_equal = (
            logical_parquet_sha256(control_path, schema, batch_size=2)
            == logical_parquet_sha256(infinity_path, schema, batch_size=2)
        )
        table_equal = control_table.equals(infinity_table)
        bit_hash_control = valid_float_bits(control_path, schema)
        bit_hash_infinity = valid_float_bits(infinity_path, schema)
        if not logical_equal or not table_equal or bit_hash_control != bit_hash_infinity:
            raise RuntimeError(f"lambda=infinity fixture mismatch for {filename}")
        artifacts[filename] = {
            "rows": control_table.num_rows,
            "logical_values_equal": logical_equal,
            "arrow_tables_equal": table_equal,
            "floating_value_bit_hashes_equal": True,
            "floating_value_bit_hashes": bit_hash_control,
            "control_sha256": file_sha256(control_path),
            "lambda_infinity_sha256": file_sha256(infinity_path),
        }

    report = {
        "status": "passed",
        "definition": "same publisher input fixture; lambda positive infinity versus unweighted scores",
        "score_float32_bits_equal": source_bits == inf_bits,
        "transform": transform,
        "artifacts": artifacts,
    }
    report_path = root / "fixture_report.json"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {**report, "report": str(report_path), "report_sha256": sha256(report_path)}


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.monotonic()
    if args.lambda_bp != FROZEN_LAMBDA_BP:
        raise SystemExit(
            f"production lambda must equal frozen {FROZEN_LAMBDA_BP}; observed {args.lambda_bp}"
        )
    if args.threads != 4 or args.memory_limit != "8GB":
        raise SystemExit("production publisher settings are frozen to threads=4, memory_limit=8GB")
    if os.environ.get("PU_ATOMIC_DIRECTORY_MODE") != "symlink":
        raise SystemExit("Quest GPFS publication requires PU_ATOMIC_DIRECTORY_MODE=symlink")
    if not args.run_dir.is_dir():
        raise SystemExit(f"run directory must already exist: {args.run_dir}")
    args.scratch_dir.mkdir(parents=True, exist_ok=True)

    source_bundle = args.source_bundle.resolve()
    source_manifest_path = source_bundle / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "complete":
        raise RuntimeError("source v4 bundle is not complete")
    if source_manifest.get("expected_method") != EXPECTED_METHOD:
        raise RuntimeError("source v4 expected method changed")
    source_inputs = {
        name: Path(source_manifest["inputs"][name]["path"]).resolve()
        for name in ("catalog", "genes", "candidate_links", "retained_links", "measurements", "partitions")
    }
    for name, expected in EXPECTED_V4_INPUTS.items():
        observed = file_sha256(source_inputs[name])
        if observed != expected or source_manifest["inputs"][name]["sha256"] != expected:
            raise RuntimeError(f"source input drift for {name}: {observed} != {expected}")
    if source_inputs["candidate_links"] != source_inputs["retained_links"]:
        raise RuntimeError("v4 candidate and retained links are not the same artifact")

    fixture = run_infinity_fixture(args.run_dir, args.scratch_dir / "fixture")
    weighted_links = args.run_dir / "weighted_links.parquet"
    link_report = create_weighted_links(
        source_inputs["retained_links"],
        weighted_links,
        lambda_bp=args.lambda_bp,
        scratch_dir=args.scratch_dir / "weighted_links",
    )
    bundle = args.run_dir / "weighted_bundle"
    result = build_permitted_cre_view_from_parquet(
        source_inputs["catalog"],
        source_inputs["genes"],
        weighted_links,
        weighted_links,
        source_inputs["measurements"],
        source_inputs["partitions"],
        bundle,
        expected_method=EXPECTED_METHOD,
        min_observed_weight_fraction=source_manifest["min_observed_weight_fraction"],
        memory_limit=args.memory_limit,
        threads=args.threads,
        scratch_dir=args.scratch_dir / "publisher",
    )
    replay = validate_cre_view_bundle(
        bundle,
        memory_limit=args.memory_limit,
        threads=args.threads,
        hash_batch_size=10_000,
        replay_mode="full",
        scratch_dir=args.scratch_dir / "independent_replay",
    )
    replay_path = args.run_dir / "full_replay.json"
    with replay_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(replay, indent=2, sort_keys=True) + "\n")
    if replay["status"] != "valid" or replay["replay_mode"] != "full":
        raise RuntimeError("independent full replay did not validate")

    weighted_manifest_path = bundle / "manifest.json"
    weighted_manifest = json.loads(weighted_manifest_path.read_text(encoding="utf-8"))
    source_code = [
        CODE_ROOT / "src/features/cre_views_duckdb.py",
        CODE_ROOT / "src/features/validate_cre_view_bundle.py",
        CODE_ROOT / "src/features/contracts.py",
        CODE_ROOT / "src/features/store.py",
    ]
    script_path = Path(__file__).resolve()
    build_report = {
        "phase": "section 39.2 preregistered weighted CRE bundle",
        "status": "complete",
        "weight": {
            "formula": "score_v4 * exp(-abs(distance_to_tss_bp) / lambda_bp)",
            "distance_definition": "signed cCRE-midpoint-minus-TSS distance from v4 link table",
            "lambda_bp": FROZEN_LAMBDA_BP,
            "w_at_zero": 1.0,
            "amplitude_used": False,
        },
        "source_bundle": {
            "path": str(source_bundle),
            "manifest_sha256": sha256(source_manifest_path),
            "status": source_manifest["status"],
            "view_rows": source_manifest["view_rows"],
            "qc_rows": source_manifest["qc_rows"],
            "dependency_rows": source_manifest["dependency_rows"],
        },
        "inputs": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in source_inputs.items()
        },
        "weighted_links": link_report,
        "weighted_bundle": {
            "path": str(bundle.resolve()),
            "manifest": str(weighted_manifest_path.resolve()),
            "manifest_sha256": sha256(weighted_manifest_path),
            "view_rows": result.view_rows,
            "qc_rows": result.qc_rows,
            "dependency_rows": result.dependency_rows,
            "feature_ids": list(result.feature_ids),
            "peak_spill_bytes": result.peak_spill_bytes,
            "output_sha256": weighted_manifest["output_sha256"],
        },
        "lambda_infinity_fixture": {
            "status": fixture["status"],
            "report": fixture["report"],
            "report_sha256": fixture["report_sha256"],
            "score_float32_bits_equal": fixture["score_float32_bits_equal"],
            "artifact_count": len(fixture["artifacts"]),
        },
        "full_replay": {
            "status": replay["status"],
            "mode": replay["replay_mode"],
            "report": str(replay_path.resolve()),
            "report_sha256": sha256(replay_path),
        },
        "provenance": {
            "design": str(args.design.resolve()),
            "design_sha256": sha256(args.design),
            "build_script": str(script_path),
            "build_script_sha256": sha256(script_path),
            "source_code_release": str(CODE_ROOT),
            "source_code_sha256": {str(path.relative_to(CODE_ROOT)): sha256(path) for path in source_code},
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "settings": {
            "threads": args.threads,
            "memory_limit": args.memory_limit,
            "min_observed_weight_fraction": source_manifest["min_observed_weight_fraction"],
            "atomic_directory_mode": os.environ["PU_ATOMIC_DIRECTORY_MODE"],
            "duckdb_version": duckdb.__version__,
            "pyarrow_version": pa.__version__,
        },
        "resources": {
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        },
    }
    if (
        result.view_rows != source_manifest["view_rows"]
        or result.qc_rows != source_manifest["qc_rows"]
        or result.dependency_rows != source_manifest["dependency_rows"]
    ):
        raise RuntimeError("weighted bundle row counts differ from source v4")
    report_path = args.run_dir / "build_report.json"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(build_report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": build_report["status"],
        "slurm_job_id": build_report["provenance"]["slurm_job_id"],
        "weighted_links_rows": link_report["rows"],
        "bundle_rows": {
            "view": result.view_rows,
            "aggregation_qc": result.qc_rows,
            "dependencies": result.dependency_rows,
        },
        "full_replay": replay["status"],
        "lambda_infinity_fixture": fixture["status"],
        "build_report_sha256": sha256(report_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

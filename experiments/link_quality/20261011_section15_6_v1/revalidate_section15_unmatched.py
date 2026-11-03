#!/usr/bin/env python3
"""Count rE2G target IDs absent from GENCODE v50, by source biotype."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import duckdb


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--re2g", type=Path, required=True)
    parser.add_argument("--tss", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="8GB")
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.monotonic()
    if args.output_dir.exists():
        raise SystemExit(f"output already exists: {args.output_dir}")
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    re2g = args.re2g.resolve()
    tss = args.tss.resolve()

    connection = duckdb.connect()
    try:
        connection.execute("SET threads = ?", [args.threads])
        connection.execute("SET memory_limit = ?", [args.memory_limit])
        connection.execute("SET preserve_insertion_order = false")
        connection.execute("SET temp_directory = ?", [str(args.scratch_dir.resolve())])
        connection.execute(
            f"""
            CREATE VIEW re2g AS
            SELECT CAST(column1 AS VARCHAR) AS gene_id,
                   CAST(column3 AS VARCHAR) AS source_biotype
            FROM read_csv('{sql_path(re2g)}', header=false, delim='\t',
                          all_varchar=true, ignore_errors=false, nullstr='')
            """
        )
        connection.execute(
            f"CREATE VIEW tss AS SELECT * FROM read_parquet('{sql_path(tss)}')"
        )
        connection.execute(
            """
            CREATE TEMP TABLE targets AS
            SELECT gene_id, min(source_biotype) AS source_biotype,
                   count(DISTINCT source_biotype)::BIGINT AS source_biotypes,
                   count(*)::BIGINT AS source_rows
            FROM re2g GROUP BY gene_id
            """
        )
        integrity = connection.execute(
            """
            SELECT sum(source_rows)::HUGEINT, count(*)::BIGINT,
                   count(*) FILTER (WHERE source_biotypes <> 1)::BIGINT
            FROM targets
            """
        ).fetchone()
        if integrity[2]:
            raise RuntimeError(
                f"{integrity[2]} rE2G target IDs have multiple source biotypes"
            )
        rows = connection.execute(
            """
            SELECT coalesce(nullif(source_biotype, ''), '<missing>') AS source_biotype,
                   count(*)::BIGINT AS target_ids,
                   count(*) FILTER (WHERE t.gene_id IS NOT NULL)::BIGINT AS matched,
                   count(*) FILTER (WHERE t.gene_id IS NULL)::BIGINT AS unmatched
            FROM targets r LEFT JOIN tss t USING (gene_id)
            GROUP BY source_biotype ORDER BY source_biotype
            """
        ).fetchall()
        crosswalk_rows = connection.execute(
            """
            SELECT coalesce(nullif(r.source_biotype, ''), '<missing>') AS source_biotype,
                   t.gene_biotype AS gencode_biotype,
                   count(*)::BIGINT AS target_ids
            FROM targets r JOIN tss t USING (gene_id)
            GROUP BY source_biotype, gencode_biotype
            ORDER BY source_biotype, gencode_biotype
            """
        ).fetchall()
    finally:
        connection.close()

    by_biotype = [
        {
            "source_biotype": biotype,
            "target_ids": int(total),
            "matched_gencode_v50": int(matched),
            "unmatched_gencode_v50": int(unmatched),
            "unmatched_fraction": round(float(unmatched) / total, 8),
        }
        for biotype, total, matched, unmatched in rows
    ]
    total_ids = sum(row["target_ids"] for row in by_biotype)
    unmatched_ids = sum(row["unmatched_gencode_v50"] for row in by_biotype)
    report = {
        "phase": "section 15 unmatched rE2G target audit",
        "status": "complete",
        "contract": {
            "gene_id_column": "column1",
            "source_biotype_column": "column3",
            "unit": "distinct rE2G gene_id",
        },
        "inputs": {
            "re2g": {"path": str(re2g), "sha256": sha256(re2g)},
            "tss": {"path": str(tss), "sha256": sha256(tss)},
        },
        "integrity": {
            "re2g_rows": int(integrity[0]),
            "distinct_target_ids": int(integrity[1]),
            "target_ids_with_multiple_source_biotypes": int(integrity[2]),
        },
        "summary": {
            "distinct_target_ids": total_ids,
            "matched_gencode_v50": total_ids - unmatched_ids,
            "unmatched_gencode_v50": unmatched_ids,
            "unmatched_fraction": round(float(unmatched_ids) / total_ids, 8),
        },
        "by_source_biotype": by_biotype,
        "matched_biotype_crosswalk": [
            {
                "source_biotype": source,
                "gencode_biotype": gencode,
                "target_ids": int(count),
            }
            for source, gencode, count in crosswalk_rows
        ],
        "settings": {
            "threads": args.threads,
            "memory_limit": args.memory_limit,
            "preserve_insertion_order": False,
            "duckdb_version": duckdb.__version__,
        },
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "elapsed_seconds_total": round(time.monotonic() - started, 1),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "section15_unmatched_targets.json"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "integrity": report["integrity"],
        "summary": report["summary"],
        "by_source_biotype": report["by_source_biotype"],
        "elapsed_seconds_total": report["elapsed_seconds_total"],
        "peak_rss_kib": report["peak_rss_kib"],
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Revalidate the retracted Section 15 distance split and target coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import duckdb
import pyarrow.parquet as pq


RE2G_COLUMNS = (
    "cre_id", "gene_id", "gene_symbol", "gene_biotype", "method", "region",
    "source_accession", "biosample", "score",
)
OLD_REFERENCE = {
    "distance_fractions": {
        "0-10kb": 0.099414,
        "10-50kb": 0.207115,
        "50-100kb": 0.167427,
        ">100kb": 0.526044,
    },
    "lncRNA_proximity_coverage": {"linked": 34774, "total": 34866},
    "documented_lncRNA_rE2G_coverage": {"linked": 1434, "total": 34866},
    "protein_coding_rE2G_coverage": {"linked": 17809},
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--re2g", type=Path, required=True)
    parser.add_argument("--tss", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="20GB")
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
    bundle = args.bundle_dir.resolve()
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise SystemExit("bundle manifest is not complete")
    re2g = args.re2g.resolve()
    tss = args.tss.resolve()
    links = Path(manifest["inputs"]["retained_links"]["path"]).resolve()
    for path in (re2g, tss, links):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    links_sha256 = sha256(links)
    if links_sha256 != manifest["inputs"]["retained_links"]["sha256"]:
        raise SystemExit("retained-link hash does not match bundle manifest")

    connection = duckdb.connect()
    try:
        connection.execute("SET threads = ?", [args.threads])
        connection.execute("SET memory_limit = ?", [args.memory_limit])
        connection.execute("SET preserve_insertion_order = false")
        connection.execute("SET temp_directory = ?", [str(args.scratch_dir.resolve())])
        connection.execute(
            f"""
            CREATE VIEW re2g AS
            SELECT CAST(column0 AS VARCHAR) AS cre_id,
                   CAST(column1 AS VARCHAR) AS gene_id,
                   CAST(column5 AS VARCHAR) AS region
            FROM read_csv('{sql_path(re2g)}', header=false, delim='\t',
                          all_varchar=true, ignore_errors=false, nullstr='')
            """
        )
        connection.execute(
            f"CREATE VIEW tss AS SELECT * FROM read_parquet('{sql_path(tss)}')"
        )
        connection.execute(
            """
            CREATE TEMP TABLE pc_gene_ids AS
            SELECT DISTINCT CAST(gene_id AS VARCHAR) AS gene_id
            FROM tss WHERE gene_biotype = 'protein_coding'
            """
        )
        total_re2g_rows = int(
            connection.execute("SELECT count(*) FROM re2g").fetchone()[0]
        )
        connection.execute(
            "CREATE TEMP TABLE re2g_target_ids AS SELECT DISTINCT gene_id FROM re2g"
        )
        re2g_target_gene_count = int(
            connection.execute("SELECT count(*) FROM re2g_target_ids").fetchone()[0]
        )
        connection.execute(
            """
            CREATE TEMP TABLE re2g_pairs AS
            SELECT r.cre_id, r.gene_id, r.region
            FROM re2g r SEMI JOIN pc_gene_ids g USING (gene_id)
            GROUP BY r.cre_id, r.gene_id, r.region
            """
        )
        pair_integrity = connection.execute(
            """
            SELECT count(*)::BIGINT,
                   count(DISTINCT gene_id)::BIGINT
            FROM re2g_pairs
            """
        ).fetchone()
        multi_region_pairs = int(
            connection.execute(
                """
                SELECT count(*) FROM (
                    SELECT cre_id, gene_id FROM re2g_pairs
                    GROUP BY cre_id, gene_id HAVING count(*) <> 1
                )
                """
            ).fetchone()[0]
        )
        if multi_region_pairs:
            raise RuntimeError(
                f"{multi_region_pairs} rE2G pairs map to more than one region"
            )

        lncrna_coverage = connection.execute(
            """
            WITH lnc AS (
                SELECT CAST(gene_id AS VARCHAR) AS gene_id
                FROM tss WHERE gene_biotype = 'lncRNA'
            ), linked AS (
                SELECT DISTINCT CAST(gene_id AS VARCHAR) AS gene_id
                FROM read_parquet(?)
                WHERE NOT coalesce(is_lncRNA_fallback, false)
            )
            SELECT count(*) FILTER (WHERE linked.gene_id IS NOT NULL)::BIGINT,
                   count(*)::BIGINT
            FROM lnc LEFT JOIN linked USING (gene_id)
            """
            , [str(links)]
        ).fetchone()
        lncrna_re2g_coverage = connection.execute(
            """
            WITH lnc AS (
                SELECT CAST(gene_id AS VARCHAR) AS gene_id
                FROM tss WHERE gene_biotype = 'lncRNA'
            )
            SELECT count(*) FILTER (WHERE targets.gene_id IS NOT NULL)::BIGINT,
                   count(*)::BIGINT
            FROM lnc LEFT JOIN re2g_target_ids targets USING (gene_id)
            """
        ).fetchone()
        protein_coding_total = int(
            connection.execute("SELECT count(*) FROM pc_gene_ids").fetchone()[0]
        )

        connection.execute(
            """
            CREATE TEMP TABLE pc_distances AS
            WITH parsed AS (
                SELECT p.cre_id, p.gene_id,
                       split_part(p.region, '_', 1) AS region_chrom,
                       try_cast(split_part(p.region, '_', 2) AS BIGINT) AS region_start,
                       try_cast(split_part(p.region, '_', 3) AS BIGINT) AS region_end
                FROM re2g_pairs p
            )
            SELECT p.cre_id, p.gene_id, p.region_chrom, t.chrom AS gene_chrom,
                   CASE WHEN t.tss < p.region_start THEN p.region_start - t.tss
                        WHEN t.tss > p.region_end THEN t.tss - p.region_end
                        ELSE 0 END::BIGINT AS distance
            FROM parsed p JOIN tss t USING (gene_id)
            WHERE t.gene_biotype = 'protein_coding'
            """
        )
        distance_checks = connection.execute(
            """
            SELECT count(*)::BIGINT,
                   count(*) FILTER (WHERE region_chrom IS NULL OR distance IS NULL)::BIGINT,
                   count(*) FILTER (WHERE region_chrom <> gene_chrom)::BIGINT,
                   min(distance), max(distance)
            FROM pc_distances
            """
        ).fetchone()
        if distance_checks[1] or distance_checks[2]:
            raise RuntimeError(
                "distance integrity failure: "
                f"malformed={distance_checks[1]}, chromosome_mismatch={distance_checks[2]}"
            )
        bins = connection.execute(
            """
            SELECT count(*) FILTER (WHERE distance <= 10000)::BIGINT,
                   count(*) FILTER (WHERE distance > 10000 AND distance <= 50000)::BIGINT,
                   count(*) FILTER (WHERE distance > 50000 AND distance <= 100000)::BIGINT,
                   count(*) FILTER (WHERE distance > 100000)::BIGINT,
                   count(*)::BIGINT
            FROM pc_distances
            """
        ).fetchone()
    finally:
        connection.close()

    labels = ("0-10kb", "10-50kb", "50-100kb", ">100kb")
    distance = {
        label: {
            "pairs": int(bins[index]),
            "fraction": round(float(bins[index]) / bins[4], 8),
            "old_reference_fraction": OLD_REFERENCE["distance_fractions"][label],
            "difference": round(
                float(bins[index]) / bins[4] - OLD_REFERENCE["distance_fractions"][label],
                8,
            ),
        }
        for index, label in enumerate(labels)
    }
    report = {
        "phase": "section 15 context revalidation",
        "status": "complete",
        "contract": {
            "zero_based_columns": {
                str(index): name for index, name in enumerate(RE2G_COLUMNS)
            },
            "gene_id_column": "column1",
            "region_column": "column5",
            "distance_definition": "distance from TSS to the closed rE2G region interval",
            "lncRNA_coverage_definition": "GENCODE v50 lncRNA gene IDs present in at least one non-fallback bundle retained link",
            "lncRNA_rE2G_coverage_definition": "GENCODE v50 lncRNA gene IDs present in at least one row of the pinned rE2G source",
            "protein_coding_coverage_definition": "GENCODE v50 protein-coding gene IDs present in at least one rE2G pair",
            "bundle_tier_semantics": {
                "admitted_method": "tier1_distance_window_unconstrained",
                "primary_rule": "same-chromosome cCRE midpoint within plus or minus 100,000 bp of the gene TSS",
                "nonfallback": "primary fixed-window proximity link with is_lncRNA_fallback false; candidate and retained links coincide and no ranking is applied",
                "excluded_fallback": "wider rescue links applied only to lncRNAs uncovered by primary links; the v4 input contract rejects every is_lncRNA_fallback=true row",
            },
        },
        "bundle": {
            "path": str(bundle),
            "manifest_sha256": sha256(manifest_path),
            "status": manifest["status"],
        },
        "inputs": {
            "re2g": {"path": str(re2g), "sha256": sha256(re2g)},
            "retained_links": {"path": str(links), "sha256": links_sha256},
            "tss": {
                "path": str(tss),
                "sha256": sha256(tss),
                "schema": [
                    [field.name, str(field.type), field.nullable]
                    for field in pq.read_schema(tss)
                ],
            },
        },
        "integrity": {
            "re2g_rows": total_re2g_rows,
            "re2g_distinct_target_genes": re2g_target_gene_count,
            "re2g_distinct_protein_coding_pairs": int(pair_integrity[0]),
            "pairs_with_multiple_regions": multi_region_pairs,
            "re2g_distinct_protein_coding_genes": int(pair_integrity[1]),
            "protein_coding_pairs_with_tss": int(distance_checks[0]),
            "malformed_region_or_distance": int(distance_checks[1]),
            "chromosome_mismatches": int(distance_checks[2]),
            "minimum_distance_bp": int(distance_checks[3]),
            "maximum_distance_bp": int(distance_checks[4]),
        },
        "lncRNA_proximity_coverage": {
            "linked": int(lncrna_coverage[0]),
            "total": int(lncrna_coverage[1]),
            "fraction": round(float(lncrna_coverage[0]) / lncrna_coverage[1], 8),
        },
        "lncRNA_rE2G_coverage": {
            "linked": int(lncrna_re2g_coverage[0]),
            "total": int(lncrna_re2g_coverage[1]),
            "fraction": round(
                float(lncrna_re2g_coverage[0]) / lncrna_re2g_coverage[1], 8
            ),
        },
        "protein_coding_rE2G_coverage": {
            "linked": int(pair_integrity[1]),
            "total": protein_coding_total,
            "fraction": round(float(pair_integrity[1]) / protein_coding_total, 8),
        },
        "old_reference_coverage": OLD_REFERENCE,
        "protein_coding_pair_distance": {
            "total_pairs": int(bins[4]),
            "bins": distance,
        },
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
    output = args.output_dir / "section15_context_revalidation.json"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "integrity": report["integrity"],
        "lncRNA_proximity_coverage": report["lncRNA_proximity_coverage"],
        "lncRNA_rE2G_coverage": report["lncRNA_rE2G_coverage"],
        "protein_coding_rE2G_coverage": report["protein_coding_rE2G_coverage"],
        "protein_coding_pair_distance": report["protein_coding_pair_distance"],
        "elapsed_seconds_total": report["elapsed_seconds_total"],
        "peak_rss_kib": report["peak_rss_kib"],
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

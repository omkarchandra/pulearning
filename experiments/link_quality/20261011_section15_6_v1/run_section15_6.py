#!/usr/bin/env python3
"""Section 15.6: rE2G-supported precision by distance bin and cCRE class.

The proximity links and SCREEN catalog are resolved from the published v4
bundle manifest.  The bundle and all source artifacts are read-only.  Results
are written with create-only semantics beneath this experiment directory.

"Precision" retains the operational definition used by the preceding Section
15 analysis: a proximity (cCRE, gene) pair is supported when the same pair is
present in rE2G in at least k distinct biosamples.  rE2G is a comparator, not
biological ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
import time
from pathlib import Path

import duckdb
import pyarrow
import pyarrow.parquet as pq


EXPECTED_CLASSES = (
    "PLS",
    "pELS",
    "dELS",
    "CA-CTCF",
    "CA-H3K4me3",
    "CA-TF",
    "CA",
    "TF",
)
DISTANCE_BINS = (
    ("0-10kb", 0, 10_000),
    ("10-50kb", 10_000, 50_000),
    ("50-100kb", 50_000, 100_000),
)
REFERENCE_SECTION15 = {
    1: {"precision_median": 0.463415, "recall_median": 0.605882},
    5: {"precision_median": 0.270588, "recall_median": 0.753334},
    20: {"precision_median": 0.151079, "recall_median": 0.833333},
}
RE2G_COLUMNS = (
    "cre_id",
    "gene_id",
    "gene_symbol",
    "gene_biotype",
    "method",
    "region",
    "source_accession",
    "biosample",
    "score",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--re2g", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="20GB")
    parser.add_argument("--k-thresholds", type=int, nargs="+", default=[1, 5, 20])
    return parser.parse_args(argv)


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def rounded(value):
    return None if value is None else round(float(value), 6)


def schema_record(path: Path):
    schema = pq.read_schema(path)
    return [(field.name, str(field.type), field.nullable) for field in schema]


def precision_query(connection, dimensions: tuple[str, ...], k: int):
    select_dimensions = ", ".join(dimensions)
    prefix = f"{select_dimensions}, " if dimensions else ""
    group_dimensions = f"GROUP BY {select_dimensions}" if dimensions else ""
    per_gene_group = f"GROUP BY {select_dimensions}, gene_id" if dimensions else "GROUP BY gene_id"
    order_dimensions = f"ORDER BY {select_dimensions}" if dimensions else ""
    rows = connection.execute(
        f"""
        WITH per_gene AS (
            SELECT {prefix}gene_id,
                   count(*)::BIGINT AS proximity_pairs,
                   count(*) FILTER (WHERE re2g_support >= {int(k)})::BIGINT
                       AS supported_pairs
            FROM annotated_proximity
            {per_gene_group}
        )
        SELECT {prefix}
               count(*)::BIGINT AS genes_with_proximity,
               sum(proximity_pairs)::HUGEINT AS proximity_pairs,
               sum(supported_pairs)::HUGEINT AS supported_pairs,
               median(supported_pairs::DOUBLE / nullif(proximity_pairs, 0))
                   AS macro_precision_median,
               quantile_cont(supported_pairs::DOUBLE / nullif(proximity_pairs, 0),
                             [0.25, 0.75]) AS macro_precision_iqr,
               sum(supported_pairs)::DOUBLE / nullif(sum(proximity_pairs), 0)
                   AS micro_precision
        FROM per_gene
        {group_dimensions}
        {order_dimensions}
        """
    ).fetchall()
    names = list(dimensions) + [
        "genes_with_proximity",
        "proximity_pairs",
        "supported_pairs",
        "macro_precision_median",
        "macro_precision_iqr",
        "micro_precision",
    ]
    result = []
    for values in rows:
        row = dict(zip(names, values))
        iqr = row["macro_precision_iqr"]
        row["genes_with_proximity"] = int(row["genes_with_proximity"])
        row["proximity_pairs"] = int(row["proximity_pairs"])
        row["supported_pairs"] = int(row["supported_pairs"])
        row["macro_precision_median"] = rounded(row["macro_precision_median"])
        row["macro_precision_iqr"] = [rounded(iqr[0]), rounded(iqr[1])]
        row["micro_precision"] = rounded(row["micro_precision"])
        result.append(row)
    return result


def overall_summary(connection, k: int):
    precision = precision_query(connection, (), k)[0]
    recall = connection.execute(
        f"""
        WITH detected AS (
            SELECT gene_id,
                   count(*) FILTER (WHERE re2g_support >= {int(k)})::BIGINT AS tp
            FROM annotated_proximity GROUP BY gene_id
        ), truth AS (
            SELECT gene_id, count(*)::BIGINT AS truth_pairs
            FROM re2g_pairs WHERE support >= {int(k)} GROUP BY gene_id
        )
        SELECT median(coalesce(d.tp, 0)::DOUBLE / nullif(t.truth_pairs, 0)),
               quantile_cont(coalesce(d.tp, 0)::DOUBLE / nullif(t.truth_pairs, 0),
                             [0.25, 0.75]),
               sum(coalesce(d.tp, 0))::DOUBLE / nullif(sum(t.truth_pairs), 0),
               count(*)
        FROM truth t LEFT JOIN detected d USING (gene_id)
        """
    ).fetchone()
    reference = REFERENCE_SECTION15.get(k)
    summary = {
        **precision,
        "macro_recall_median": rounded(recall[0]),
        "macro_recall_iqr": (
            [rounded(recall[1][0]), rounded(recall[1][1])]
            if recall[1] is not None
            else [None, None]
        ),
        "micro_recall": rounded(recall[2]),
        "genes_with_re2g_at_threshold": int(recall[3]),
    }
    if reference:
        summary["section15_reference"] = reference
        summary["section15_reference_note"] = (
            "Diagnostic only: the preceding Section 15 reader bound column8 "
            "(score) as biosample. This run correctly binds column7 as biosample."
        )
        summary["difference_from_section15"] = {
            "precision_median": rounded(
                summary["macro_precision_median"] - reference["precision_median"]
            ),
            "recall_median": rounded(
                summary["macro_recall_median"] - reference["recall_median"]
            ),
        }
    return summary


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"refusing to write empty table: {path.name}")
    fields = list(rows[0])
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            encoded["macro_precision_iqr"] = json.dumps(
                encoded["macro_precision_iqr"], separators=(",", ":")
            )
            writer.writerow(encoded)


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.monotonic()
    if args.threads < 1 or any(k < 1 for k in args.k_thresholds):
        raise SystemExit("threads and k thresholds must be positive")
    if len(set(args.k_thresholds)) != len(args.k_thresholds):
        raise SystemExit("k thresholds must be unique")
    if args.output_dir.exists():
        raise SystemExit(f"output directory already exists: {args.output_dir}")
    args.scratch_dir.mkdir(parents=True, exist_ok=True)

    bundle_dir = args.bundle_dir.resolve()
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise SystemExit("bundle manifest is not complete")
    links_path = Path(manifest["inputs"]["retained_links"]["path"]).resolve()
    catalog_path = Path(manifest["inputs"]["catalog"]["path"]).resolve()
    re2g_path = args.re2g.resolve()
    for path in (links_path, catalog_path, re2g_path):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")

    input_hashes = {
        "retained_links": file_sha256(links_path),
        "catalog": file_sha256(catalog_path),
        "re2g": file_sha256(re2g_path),
        "bundle_manifest": file_sha256(manifest_path),
    }
    for key in ("retained_links", "catalog"):
        expected = manifest["inputs"][key]["sha256"]
        if input_hashes[key] != expected:
            raise SystemExit(
                f"{key} hash does not match bundle manifest: "
                f"{input_hashes[key]} != {expected}"
            )

    connection = duckdb.connect()
    try:
        connection.execute("SET threads = ?", [args.threads])
        connection.execute("SET memory_limit = ?", [args.memory_limit])
        connection.execute("SET preserve_insertion_order = false")
        connection.execute("SET temp_directory = ?", [str(args.scratch_dir.resolve())])
        connection.execute(
            f"""
            CREATE TEMP TABLE proximity AS
            SELECT CAST(l.cre_id AS VARCHAR) AS cre_id,
                   CAST(l.gene_id AS VARCHAR) AS gene_id,
                   CAST(l.distance_to_tss AS BIGINT) AS signed_distance_to_tss,
                   abs(CAST(l.distance_to_tss AS BIGINT)) AS distance_to_tss,
                   CAST(c.class AS VARCHAR) AS ccre_class,
                   CASE
                       WHEN abs(l.distance_to_tss) BETWEEN 0 AND 10000 THEN '0-10kb'
                       WHEN abs(l.distance_to_tss) > 10000
                            AND abs(l.distance_to_tss) <= 50000
                           THEN '10-50kb'
                       WHEN abs(l.distance_to_tss) > 50000
                            AND abs(l.distance_to_tss) <= 100000
                           THEN '50-100kb'
                       ELSE NULL
                   END AS distance_bin
            FROM read_parquet('{sql_path(links_path)}') l
            LEFT JOIN read_parquet('{sql_path(catalog_path)}') c USING (cre_id)
            WHERE l.gene_biotype = 'protein_coding'
              AND NOT coalesce(l.is_lncRNA_fallback, false)
            """
        )
        proximity_checks = connection.execute(
            """
            SELECT count(*)::BIGINT,
                   count(*) FILTER (WHERE ccre_class IS NULL)::BIGINT,
                   count(*) FILTER (WHERE distance_bin IS NULL)::BIGINT,
                   min(distance_to_tss), max(distance_to_tss),
                   min(signed_distance_to_tss), max(signed_distance_to_tss),
                   count(DISTINCT gene_id)::BIGINT,
                   count(DISTINCT cre_id)::BIGINT
            FROM proximity
            """
        ).fetchone()
        duplicate_pairs = int(
            connection.execute(
                """
                SELECT count(*) FROM (
                    SELECT cre_id, gene_id FROM proximity
                    GROUP BY cre_id, gene_id HAVING count(*) <> 1
                )
                """
            ).fetchone()[0]
        )
        observed_classes = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT ccre_class FROM proximity WHERE ccre_class IS NOT NULL"
            ).fetchall()
        }
        if proximity_checks[1] or proximity_checks[2] or duplicate_pairs:
            raise RuntimeError(
                "proximity integrity failure: "
                f"missing_class={proximity_checks[1]}, "
                f"invalid_distance={proximity_checks[2]}, "
                f"duplicate_pairs={duplicate_pairs}"
            )
        if observed_classes != set(EXPECTED_CLASSES):
            raise RuntimeError(
                f"unexpected cCRE classes: {sorted(observed_classes)}"
            )
        connection.execute(
            "CREATE TEMP TABLE pc_gene_ids AS "
            "SELECT DISTINCT gene_id FROM proximity"
        )

        connection.execute(
            f"""
            CREATE VIEW re2g AS
            SELECT CAST(column0 AS VARCHAR) AS cre_id,
                   CAST(column1 AS VARCHAR) AS gene_id,
                   CAST(column7 AS VARCHAR) AS biosample
            FROM read_csv('{sql_path(re2g_path)}', header=false, delim='\t',
                          all_varchar=true, ignore_errors=false, nullstr='')
            """
        )
        total_re2g_rows = int(
            connection.execute("SELECT count(*) FROM re2g").fetchone()[0]
        )
        connection.execute(
            """
            CREATE TEMP TABLE re2g_pairs AS
            SELECT r.cre_id, r.gene_id,
                   count(DISTINCT nullif(r.biosample, ''))::BIGINT AS support
            FROM re2g r SEMI JOIN pc_gene_ids g USING (gene_id)
            GROUP BY r.cre_id, r.gene_id
            """
        )
        re2g_pair_rows = int(
            connection.execute("SELECT count(*) FROM re2g_pairs").fetchone()[0]
        )
        connection.execute(
            """
            CREATE TEMP TABLE cohort_genes AS
            SELECT DISTINCT gene_id FROM re2g_pairs
            """
        )
        connection.execute(
            """
            CREATE TEMP TABLE annotated_proximity AS
            SELECT p.*, coalesce(r.support, 0)::BIGINT AS re2g_support
            FROM proximity p
            SEMI JOIN cohort_genes g USING (gene_id)
            LEFT JOIN re2g_pairs r USING (cre_id, gene_id)
            """
        )
        cohort = connection.execute(
            """
            SELECT count(*)::BIGINT, count(DISTINCT gene_id)::BIGINT,
                   count(DISTINCT cre_id)::BIGINT
            FROM annotated_proximity
            """
        ).fetchone()

        overall = {}
        by_distance = []
        by_class = []
        by_distance_and_class = []
        for k in sorted(args.k_thresholds):
            overall[str(k)] = overall_summary(connection, k)
            for row in precision_query(connection, ("distance_bin",), k):
                by_distance.append({"k": k, **row})
            for row in precision_query(connection, ("ccre_class",), k):
                by_class.append({"k": k, **row})
            for row in precision_query(
                connection, ("distance_bin", "ccre_class"), k
            ):
                by_distance_and_class.append({"k": k, **row})
    finally:
        connection.close()

    distance_order = {name: index for index, (name, _, _) in enumerate(DISTANCE_BINS)}
    class_order = {name: index for index, name in enumerate(EXPECTED_CLASSES)}
    by_distance.sort(key=lambda row: (row["k"], distance_order[row["distance_bin"]]))
    by_class.sort(key=lambda row: (row["k"], class_order[row["ccre_class"]]))
    by_distance_and_class.sort(
        key=lambda row: (
            row["k"],
            distance_order[row["distance_bin"]],
            class_order[row["ccre_class"]],
        )
    )

    report = {
        "phase": "section 15.6 link-quality follow-up",
        "status": "complete",
        "definition": {
            "unit": "distinct (cCRE, gene) proximity pair",
            "distance": "absolute cCRE-midpoint to TSS distance; bundle distance_to_tss is signed",
            "cohort": (
                "protein-coding genes in the published bundle retained-link table "
                "with at least one rE2G pair"
            ),
            "support": "same (cCRE, gene) pair in at least k distinct rE2G biosamples",
            "macro_precision": (
                "per-gene supported/proximity precision summarized by median and IQR; "
                "genes with no proximity pair in a stratum are excluded from that stratum"
            ),
            "micro_precision": "pooled supported pairs / pooled proximity pairs",
            "distance_bins_bp": [
                {"name": name, "lower_exclusive_except_zero": low, "upper_inclusive": high}
                for name, low, high in DISTANCE_BINS
            ],
            "ccre_classes": list(EXPECTED_CLASSES),
            "caution": "rE2G support is a comparator, not biological ground truth",
        },
        "bundle": {
            "path": str(bundle_dir),
            "manifest_sha256": input_hashes["bundle_manifest"],
            "schema_version": manifest.get("schema_version"),
            "status": manifest.get("status"),
            "dependency_rows": manifest.get("dependency_rows"),
            "output_sha256": manifest.get("output_sha256"),
        },
        "inputs": {
            "retained_links": {
                "path": str(links_path),
                "sha256": input_hashes["retained_links"],
                "schema": schema_record(links_path),
            },
            "catalog": {
                "path": str(catalog_path),
                "sha256": input_hashes["catalog"],
                "schema": schema_record(catalog_path),
            },
            "re2g": {
                "path": str(re2g_path),
                "sha256": input_hashes["re2g"],
                "zero_based_columns": {
                    str(index): name for index, name in enumerate(RE2G_COLUMNS)
                },
                "support_column": "column7 (biosample)",
                "prior_reader_defect": (
                    "The preceding Section 15 script used column8 (score) as "
                    "biosample; its k-threshold results are retained only as a "
                    "diagnostic reference and are not treated as reproduced."
                ),
            },
        },
        "integrity": {
            "bundle_protein_coding_nonfallback_proximity_pairs": int(proximity_checks[0]),
            "missing_ccre_class": int(proximity_checks[1]),
            "invalid_distance_bin": int(proximity_checks[2]),
            "minimum_distance_bp": int(proximity_checks[3]),
            "maximum_distance_bp": int(proximity_checks[4]),
            "minimum_signed_distance_bp": int(proximity_checks[5]),
            "maximum_signed_distance_bp": int(proximity_checks[6]),
            "bundle_protein_coding_genes_with_proximity": int(proximity_checks[7]),
            "distinct_proximity_ccres": int(proximity_checks[8]),
            "duplicate_proximity_pairs": duplicate_pairs,
            "observed_ccre_classes": sorted(observed_classes),
            "re2g_rows": total_re2g_rows,
            "re2g_distinct_pairs_for_bundle_protein_coding_genes": re2g_pair_rows,
            "analysis_proximity_pairs": int(cohort[0]),
            "analysis_genes": int(cohort[1]),
            "analysis_distinct_ccres": int(cohort[2]),
        },
        "thresholds": overall,
        "precision_by_distance_bin": by_distance,
        "precision_by_ccre_class": by_class,
        "precision_by_distance_and_ccre_class": by_distance_and_class,
        "settings": {
            "threads": args.threads,
            "memory_limit": args.memory_limit,
            "preserve_insertion_order": False,
            "duckdb_version": duckdb.__version__,
            "pyarrow_version": pyarrow.__version__,
        },
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "elapsed_seconds_total": round(time.monotonic() - started, 1),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "precision_by_distance_bin.csv", by_distance)
    write_csv(args.output_dir / "precision_by_ccre_class.csv", by_class)
    write_csv(
        args.output_dir / "precision_by_distance_and_ccre_class.csv",
        by_distance_and_class,
    )
    with (args.output_dir / "link_quality_section15_6.json").open(
        "x", encoding="utf-8"
    ) as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "integrity": report["integrity"],
        "thresholds": report["thresholds"],
        "elapsed_seconds_total": report["elapsed_seconds_total"],
        "peak_rss_kib": report["peak_rss_kib"],
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

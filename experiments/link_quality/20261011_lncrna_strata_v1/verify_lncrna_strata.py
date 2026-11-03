#!/usr/bin/env python3
"""Independently verify the frozen GENCODE v50 lncRNA strata artifact."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


EXPECTED = {
    "gencode_gtf": "83fba3e9b03f0b8c958f3595c6c350adc55f468abf8b0e47b6d5284cfe13a453",
    "gene_tss": "24c2a0180f6ab7feb2e6d9e6008a8555694cd9574d16abaa30f70f06c7cc328b",
    "retained_links": "b313dbcf01bdb4c4d04d5a83cc0e82df7c5dc3877988a80007aa7143361aeb5a",
    "catalog": "92762255f4e9d1e665a58a2e1bca7f5420d29e3bffa883a528707c10e8ab350d",
}
EXPECTED_CLASSES = ("divergent", "antisense", "sense-overlapping", "intergenic")
EXPECTED_BINS = ("<5kb", "5-50kb", "50-200kb", ">200kb")
ID_PATTERN = re.compile(r"^(?P<base>[^.]+)\.(?P<version>[0-9]+)(?P<suffix>_PAR_Y)?$")
EXPECTED_SCHEMA = pa.schema(
    [
        pa.field("gene_id", pa.string(), nullable=False),
        pa.field("gene_id_versioned", pa.string(), nullable=False),
        pa.field("gene_symbol", pa.string(), nullable=False),
        pa.field("chrom", pa.string(), nullable=False),
        pa.field("strand", pa.string(), nullable=False),
        pa.field("tss", pa.int64(), nullable=False),
        pa.field("gene_start", pa.int64(), nullable=False),
        pa.field("gene_end", pa.int64(), nullable=False),
        pa.field("locus_class", pa.string(), nullable=False),
        pa.field("nearest_pc_gene_id", pa.string(), nullable=False),
        pa.field("nearest_pc_tss", pa.int64(), nullable=False),
        pa.field("nearest_pc_distance_bp", pa.int64(), nullable=False),
        pa.field("nearest_pc_distance_bin", pa.string(), nullable=False),
        pa.field("self_overlap_tss_1kb", pa.bool_(), nullable=False),
    ]
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="16GB")
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def gtf_gene_rows(path: Path) -> list[dict]:
    rows = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "gene":
                continue
            attributes = {}
            for item in fields[8].split(";"):
                key, separator, value = item.strip().partition(" ")
                if separator and key not in attributes:
                    attributes[key] = value.strip().strip('"')
            biotype = attributes.get("gene_type", attributes.get("gene_biotype"))
            if biotype not in {"lncRNA", "protein_coding"}:
                continue
            versioned = attributes["gene_id"]
            match = ID_PATTERN.fullmatch(versioned)
            if match is None:
                raise AssertionError(f"invalid versioned GENCODE ID: {versioned}")
            rows.append(
                {
                    "gene_id": match.group("base") + (match.group("suffix") or ""),
                    "gene_biotype": biotype,
                    "chrom_gtf": fields[0],
                    "strand_gtf": fields[6],
                    "start_gtf": int(fields[3]) - 1,
                    "end_gtf": int(fields[4]),
                }
            )
    return rows


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    assert report["phase"] == "section 8 lncRNA locus strata"
    assert report["status"] == "complete"
    assert report["universe"] == {
        "definition": "full pinned GENCODE v50 gene set",
        "lncRNA_genes": 34_866,
        "protein_coding_genes": 20_107,
    }
    assert report["rules"]["precedence"] == list(EXPECTED_CLASSES)
    assert list(report["rules"]["distance_bins"]) == list(EXPECTED_BINS)
    assert report["inputs"]["frozen_rules"]["sha256"] == sha256(args.rules)
    assert report["output"]["sha256"] == sha256(args.table)
    assert report["output"]["rows"] == 34_866
    assert report["execution"]["script_sha256"] == sha256(
        Path(report["execution"]["script"])
    )

    input_paths = {}
    for name, expected_hash in EXPECTED.items():
        record = report["inputs"][name]
        path = Path(record["path"])
        assert record["sha256"] == expected_hash == sha256(path)
        input_paths[name] = path

    parquet = pq.ParquetFile(args.table)
    assert parquet.schema_arrow == EXPECTED_SCHEMA
    assert parquet.metadata.num_rows == 34_866
    output_table = pq.read_table(args.table)
    output_rows = output_table.to_pylist()
    ids = [row["gene_id"] for row in output_rows]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids)) == 34_866

    gtf_rows = gtf_gene_rows(input_paths["gencode_gtf"])
    assert len(gtf_rows) == len({row["gene_id"] for row in gtf_rows}) == 54_973
    gtf_table = pa.Table.from_pylist(gtf_rows)

    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = ?", [args.threads])
        connection.execute("SET memory_limit = ?", [args.memory_limit])
        connection.execute("SET preserve_insertion_order = false")
        connection.execute("SET temp_directory = ?", [str(args.scratch_dir.resolve())])
        connection.register("gtf_verify", gtf_table)
        connection.register("observed_output", output_table)
        connection.execute(
            f"""
            CREATE TEMP TABLE eligible AS
            SELECT CAST(t.gene_id AS VARCHAR) gene_id,
                   CAST(t.gene_biotype AS VARCHAR) gene_biotype,
                   CAST(t.chrom AS VARCHAR) chrom,
                   CAST(t.strand AS VARCHAR) strand,
                   CAST(t.tss AS BIGINT) tss,
                   CAST(g.start_gtf AS BIGINT) gene_start,
                   CAST(g.end_gtf AS BIGINT) gene_end
            FROM read_parquet('{sql_path(input_paths['gene_tss'])}') t
            JOIN gtf_verify g ON CAST(t.gene_id AS VARCHAR)=g.gene_id
            WHERE t.gene_biotype IN ('lncRNA','protein_coding')
            """
        )
        eligible_counts = dict(
            connection.execute(
                "SELECT gene_biotype,count(*)::BIGINT FROM eligible GROUP BY gene_biotype"
            ).fetchall()
        )
        assert eligible_counts == {"lncRNA": 34_866, "protein_coding": 20_107}
        identity_mismatches = int(
            connection.execute(
                """
                SELECT count(*) FROM observed_output o
                FULL JOIN eligible e USING (gene_id)
                WHERE e.gene_biotype='lncRNA'
                  AND (o.gene_id IS NULL OR o.chrom != e.chrom OR o.strand != e.strand
                       OR o.tss != e.tss OR o.gene_start != e.gene_start
                       OR o.gene_end != e.gene_end)
                """
            ).fetchone()[0]
        )
        assert identity_mismatches == 0

        connection.execute(
            "CREATE TEMP TABLE pc AS SELECT * FROM eligible WHERE gene_biotype='protein_coding'"
        )
        connection.execute(
            """
            CREATE TEMP TABLE expected_geometry AS
            WITH flags AS (
              SELECT l.gene_id,
                     coalesce(bool_or(abs(l.tss-p.tss)<=1000 AND l.strand!=p.strand
                                        AND NOT (l.gene_start<p.gene_end
                                                 AND p.gene_start<l.gene_end)),false) divergent,
                     coalesce(bool_or(l.strand!=p.strand AND l.gene_start<p.gene_end
                                        AND p.gene_start<l.gene_end),false) antisense,
                     coalesce(bool_or(l.strand=p.strand AND l.gene_start<p.gene_end
                                        AND p.gene_start<l.gene_end),false) sense_overlap
              FROM observed_output l LEFT JOIN pc p
                ON l.chrom=p.chrom
               AND (abs(l.tss-p.tss)<=1000
                    OR (l.gene_start<p.gene_end AND p.gene_start<l.gene_end))
              GROUP BY l.gene_id
            ), nearest AS (
              SELECT l.gene_id, p.gene_id nearest_pc_gene_id, p.tss nearest_pc_tss,
                     abs(l.tss-p.tss)::BIGINT nearest_pc_distance_bp
              FROM observed_output l JOIN pc p ON l.chrom=p.chrom
              QUALIFY row_number() OVER (
                PARTITION BY l.gene_id ORDER BY abs(l.tss-p.tss),p.gene_id
              )=1
            )
            SELECT f.gene_id,
                   CASE WHEN divergent THEN 'divergent'
                        WHEN antisense THEN 'antisense'
                        WHEN sense_overlap THEN 'sense-overlapping'
                        ELSE 'intergenic' END locus_class,
                   n.nearest_pc_gene_id,n.nearest_pc_tss,n.nearest_pc_distance_bp,
                   CASE WHEN n.nearest_pc_distance_bp<5000 THEN '<5kb'
                        WHEN n.nearest_pc_distance_bp<=50000 THEN '5-50kb'
                        WHEN n.nearest_pc_distance_bp<=200000 THEN '50-200kb'
                        ELSE '>200kb' END nearest_pc_distance_bin
            FROM flags f JOIN nearest n USING (gene_id)
            """
        )
        geometry_mismatches = int(
            connection.execute(
                """
                SELECT count(*) FROM observed_output o
                JOIN expected_geometry e USING (gene_id)
                WHERE o.locus_class != e.locus_class
                   OR o.nearest_pc_gene_id != e.nearest_pc_gene_id
                   OR o.nearest_pc_tss != e.nearest_pc_tss
                   OR o.nearest_pc_distance_bp != e.nearest_pc_distance_bp
                   OR o.nearest_pc_distance_bin != e.nearest_pc_distance_bin
                """
            ).fetchone()[0]
        )
        assert geometry_mismatches == 0

        self_overlap_mismatches = int(
            connection.execute(
                f"""
                WITH expected AS (
                  SELECT o.gene_id,
                         count(c.cre_id)>0 AS self_overlap_tss_1kb
                  FROM observed_output o
                  LEFT JOIN read_parquet('{sql_path(input_paths['retained_links'])}') r
                    ON CAST(r.gene_id AS VARCHAR)=o.gene_id
                  LEFT JOIN read_parquet('{sql_path(input_paths['catalog'])}') c
                    ON CAST(c.cre_id AS VARCHAR)=CAST(r.cre_id AS VARCHAR)
                   AND CAST(c.chrom AS VARCHAR)=o.chrom
                   AND CAST(c."end" AS BIGINT)>o.tss-1000
                   AND CAST(c."start" AS BIGINT)<=o.tss+1000
                  GROUP BY o.gene_id
                )
                SELECT count(*) FROM observed_output o JOIN expected e USING (gene_id)
                WHERE o.self_overlap_tss_1kb != e.self_overlap_tss_1kb
                """
            ).fetchone()[0]
        )
        assert self_overlap_mismatches == 0
    finally:
        connection.close()

    class_counts = Counter(row["locus_class"] for row in output_rows)
    bin_counts = Counter(row["nearest_pc_distance_bin"] for row in output_rows)
    assert set(class_counts) == set(EXPECTED_CLASSES)
    assert set(bin_counts) == set(EXPECTED_BINS)
    assert sum(class_counts.values()) == sum(bin_counts.values()) == 34_866
    assert report["counts"]["locus_class"] == {
        name: class_counts[name] for name in EXPECTED_CLASSES
    }
    assert report["counts"]["nearest_pc_distance_bin"] == {
        name: bin_counts[name] for name in EXPECTED_BINS
    }
    true_count = sum(row["self_overlap_tss_1kb"] for row in output_rows)
    assert report["counts"]["self_overlap_tss_1kb"] == {
        "true": true_count,
        "false": 34_866 - true_count,
    }

    receipt = {
        "status": "verified",
        "report": str(args.report.resolve()),
        "report_sha256": sha256(args.report),
        "table": str(args.table.resolve()),
        "table_sha256": sha256(args.table),
        "rules_sha256": sha256(args.rules),
        "verifier": str(Path(__file__).resolve()),
        "verifier_sha256": sha256(Path(__file__).resolve()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "checks": {
            "pinned_inputs": True,
            "exact_schema_and_rows": True,
            "full_gene_identity_and_span_replay": True,
            "full_classification_replay": True,
            "full_nearest_tss_replay": True,
            "full_self_overlap_replay": True,
            "reported_counts_recomputed": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

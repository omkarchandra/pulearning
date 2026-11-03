#!/usr/bin/env python3
"""Build the frozen, label-free GENCODE v50 lncRNA locus-strata table."""

from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import json
import os
import re
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


EXPECTED_LNCRNA = 34_866
EXPECTED_PROTEIN_CODING = 20_107
EXPECTED_GTF_SHA256 = "83fba3e9b03f0b8c958f3595c6c350adc55f468abf8b0e47b6d5284cfe13a453"
EXPECTED_TSS_SHA256 = "24c2a0180f6ab7feb2e6d9e6008a8555694cd9574d16abaa30f70f06c7cc328b"
EXPECTED_LINKS_SHA256 = "b313dbcf01bdb4c4d04d5a83cc0e82df7c5dc3877988a80007aa7143361aeb5a"
EXPECTED_CATALOG_SHA256 = "92762255f4e9d1e665a58a2e1bca7f5420d29e3bffa883a528707c10e8ab350d"
EXPECTED_METHOD = "tier1_distance_window_unconstrained"
ID_PATTERN = re.compile(r"^(?P<base>[^.]+)\.(?P<version>[0-9]+)(?P<suffix>_PAR_Y)?$")

OUTPUT_SCHEMA = pa.schema(
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

RULES = {
    "precedence": ["divergent", "antisense", "sense-overlapping", "intergenic"],
    "divergent": (
        "its TSS lies within 1,000 bp of a protein-coding gene's TSS on the "
        "opposite strand, and the two genes do not overlap in genomic span"
    ),
    "antisense": (
        "its genomic span overlaps a protein-coding gene's span, and the two "
        "are on opposite strands"
    ),
    "sense-overlapping": (
        "its span overlaps a protein-coding gene's span on the same strand"
    ),
    "intergenic": "none of the above",
    "span": (
        "GENCODE gene-level 0-based half-open [start-1,end); overlap iff "
        "lnc_start < pc_end and pc_start < lnc_end"
    ),
    "nearest_tss": (
        "unsigned same-chromosome distance for every lncRNA; equal-distance "
        "ties use lexicographically smallest unversioned protein-coding gene ID"
    ),
    "distance_bins": {
        "<5kb": "distance < 5,000 bp",
        "5-50kb": "5,000 <= distance <= 50,000 bp",
        "50-200kb": "50,000 < distance <= 200,000 bp",
        ">200kb": "distance > 200,000 bp",
    },
    "self_overlap": (
        "own TSS +/-1,000 bp intersects any cCRE admitted to its own v4 feature "
        "vector; for cCRE [start,end), end > TSS-1,000 and start <= TSS+1,000"
    ),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--tss", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="16GB")
    return parser.parse_args(argv)


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def parse_attributes(raw: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        key, separator, value = item.partition(" ")
        if not separator:
            raise ValueError(f"invalid GTF attribute: {item!r}")
        result.setdefault(key, value.strip().strip('"'))
    return result


def unversioned_gene_id(value: str) -> str:
    match = ID_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"GENCODE gene ID lacks a numeric version: {value!r}")
    return match.group("base") + (match.group("suffix") or "")


def read_gtf_genes(path: Path) -> pa.Table:
    rows = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "gene":
                continue
            attributes = parse_attributes(fields[8])
            biotype = attributes.get("gene_type", attributes.get("gene_biotype"))
            if biotype not in {"lncRNA", "protein_coding"}:
                continue
            versioned = attributes["gene_id"]
            rows.append(
                {
                    "gene_id": unversioned_gene_id(versioned),
                    "gene_id_versioned_gtf": versioned,
                    "gene_symbol_gtf": attributes.get("gene_name", unversioned_gene_id(versioned)),
                    "gene_biotype_gtf": biotype,
                    "chrom_gtf": fields[0],
                    "strand_gtf": fields[6],
                    "gene_start_gtf": int(fields[3]) - 1,
                    "gene_end_gtf": int(fields[4]),
                }
            )
    schema = pa.schema(
        [
            pa.field("gene_id", pa.string(), nullable=False),
            pa.field("gene_id_versioned_gtf", pa.string(), nullable=False),
            pa.field("gene_symbol_gtf", pa.string(), nullable=False),
            pa.field("gene_biotype_gtf", pa.string(), nullable=False),
            pa.field("chrom_gtf", pa.string(), nullable=False),
            pa.field("strand_gtf", pa.string(), nullable=False),
            pa.field("gene_start_gtf", pa.int64(), nullable=False),
            pa.field("gene_end_gtf", pa.int64(), nullable=False),
        ]
    )
    return pa.Table.from_pylist(rows, schema=schema)


def distance_bin(distance: int) -> str:
    if distance < 5_000:
        return "<5kb"
    if distance <= 50_000:
        return "5-50kb"
    if distance <= 200_000:
        return "50-200kb"
    return ">200kb"


def nearest_pc_records(lnc_rows: list[dict], pc_rows: list[dict]) -> dict[str, tuple[str, int, int]]:
    grouped: dict[str, dict[int, str]] = defaultdict(dict)
    for row in pc_rows:
        by_tss = grouped[row["chrom"]]
        tss = int(row["tss"])
        gene_id = row["gene_id"]
        if tss not in by_tss or gene_id < by_tss[tss]:
            by_tss[tss] = gene_id
    ordered = {
        chrom: sorted((tss, gene_id) for tss, gene_id in values.items())
        for chrom, values in grouped.items()
    }
    result = {}
    for row in lnc_rows:
        candidates = ordered.get(row["chrom"], [])
        if not candidates:
            raise RuntimeError(f"no protein-coding TSS on {row['chrom']}")
        positions = [item[0] for item in candidates]
        index = bisect.bisect_left(positions, int(row["tss"]))
        nearby = []
        if index < len(candidates):
            nearby.append(candidates[index])
        if index:
            nearby.append(candidates[index - 1])
        nearest_tss, nearest_id = min(
            nearby,
            key=lambda item: (abs(item[0] - int(row["tss"])), item[1]),
        )
        result[row["gene_id"]] = (
            nearest_id,
            nearest_tss,
            abs(nearest_tss - int(row["tss"])),
        )
    return result


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.monotonic()
    if args.output_dir.exists():
        raise SystemExit(f"output directory already exists: {args.output_dir}")
    if args.threads < 1:
        raise SystemExit("threads must be positive")
    args.scratch_dir.mkdir(parents=True, exist_ok=True)

    bundle = args.bundle_dir.resolve()
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("v4 bundle is not complete")
    links = Path(manifest["inputs"]["retained_links"]["path"]).resolve()
    catalog = Path(manifest["inputs"]["catalog"]["path"]).resolve()
    inputs = {
        "gencode_gtf": args.gtf.resolve(),
        "gene_tss": args.tss.resolve(),
        "retained_links": links,
        "catalog": catalog,
        "bundle_manifest": manifest_path,
        "frozen_rules": args.rules.resolve(),
    }
    input_hashes = {name: sha256(path) for name, path in inputs.items()}
    expected = {
        "gencode_gtf": EXPECTED_GTF_SHA256,
        "gene_tss": EXPECTED_TSS_SHA256,
        "retained_links": EXPECTED_LINKS_SHA256,
        "catalog": EXPECTED_CATALOG_SHA256,
    }
    for name, digest in expected.items():
        if input_hashes[name] != digest:
            raise RuntimeError(f"unexpected {name} SHA-256: {input_hashes[name]} != {digest}")
    if manifest["inputs"]["retained_links"]["sha256"] != input_hashes["retained_links"]:
        raise RuntimeError("retained-link hash disagrees with v4 manifest")
    if manifest["inputs"]["catalog"]["sha256"] != input_hashes["catalog"]:
        raise RuntimeError("catalog hash disagrees with v4 manifest")

    gtf_table = read_gtf_genes(args.gtf.resolve())
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = ?", [args.threads])
        connection.execute("SET memory_limit = ?", [args.memory_limit])
        connection.execute("SET preserve_insertion_order = false")
        connection.execute("SET temp_directory = ?", [str(args.scratch_dir.resolve())])
        connection.register("gtf_arrow", gtf_table)
        connection.execute(
            f"""
            CREATE TEMP TABLE genes AS
            SELECT CAST(t.gene_id AS VARCHAR) AS gene_id,
                   CAST(t.gene_id_versioned AS VARCHAR) AS gene_id_versioned,
                   CAST(t.gene_symbol AS VARCHAR) AS gene_symbol,
                   CAST(t.gene_biotype AS VARCHAR) AS gene_biotype,
                   CAST(t.chrom AS VARCHAR) AS chrom,
                   CAST(t.strand AS VARCHAR) AS strand,
                   CAST(t.tss AS BIGINT) AS tss,
                   CAST(g.gene_start_gtf AS BIGINT) AS gene_start,
                   CAST(g.gene_end_gtf AS BIGINT) AS gene_end
            FROM read_parquet('{sql_path(args.tss)}') t
            JOIN gtf_arrow g ON CAST(t.gene_id AS VARCHAR) = g.gene_id
            WHERE CAST(t.gene_biotype AS VARCHAR) IN ('lncRNA', 'protein_coding')
            """
        )
        counts = dict(
            connection.execute(
                "SELECT gene_biotype, count(*)::BIGINT FROM genes GROUP BY gene_biotype"
            ).fetchall()
        )
        duplicate_gtf = int(
            connection.execute(
                "SELECT count(*) - count(DISTINCT gene_id) FROM gtf_arrow"
            ).fetchone()[0]
        )
        tss_integrity = connection.execute(
            f"""
            SELECT
              (SELECT count(*) FROM read_parquet('{sql_path(args.tss)}')
               WHERE gene_biotype IN ('lncRNA','protein_coding')) AS eligible_tss,
              (SELECT count(*) FROM genes) AS joined_genes,
              count(*) FILTER (
                WHERE t.gene_id_versioned != g.gene_id_versioned_gtf
                   OR t.gene_symbol != g.gene_symbol_gtf
                   OR t.gene_biotype != g.gene_biotype_gtf
                   OR t.chrom != g.chrom_gtf OR t.strand != g.strand_gtf
                   OR t.gene_start != g.gene_start_gtf OR t.gene_end != g.gene_end_gtf
              ) AS metadata_mismatches
            FROM read_parquet('{sql_path(args.tss)}') t
            JOIN gtf_arrow g ON CAST(t.gene_id AS VARCHAR) = g.gene_id
            WHERE t.gene_biotype IN ('lncRNA','protein_coding')
            """
        ).fetchone()
        if duplicate_gtf or counts != {
            "lncRNA": EXPECTED_LNCRNA,
            "protein_coding": EXPECTED_PROTEIN_CODING,
        }:
            raise RuntimeError(f"GENCODE universe integrity failure: duplicates={duplicate_gtf}, counts={counts}")
        if int(tss_integrity[0]) != int(tss_integrity[1]) or int(tss_integrity[2]):
            raise RuntimeError(f"GTF/TSS crosswalk failure: {tss_integrity}")

        link_integrity = connection.execute(
            f"""
            SELECT count(*)::BIGINT,
                   count(*) FILTER (WHERE method != '{EXPECTED_METHOD}')::BIGINT,
                   count(*) FILTER (WHERE coalesce(is_lncRNA_fallback, false))::BIGINT,
                   count(*) - count(DISTINCT (cre_id, gene_id))::BIGINT
            FROM read_parquet('{sql_path(links)}')
            """
        ).fetchone()
        if any(int(value) for value in link_integrity[1:]):
            raise RuntimeError(f"v4 retained-link contract failure: {link_integrity}")

        connection.execute(
            "CREATE TEMP TABLE lnc AS SELECT * FROM genes WHERE gene_biotype='lncRNA'"
        )
        connection.execute(
            "CREATE TEMP TABLE pc AS SELECT * FROM genes WHERE gene_biotype='protein_coding'"
        )
        class_rows = connection.execute(
            """
            SELECT l.gene_id,
                   coalesce(bool_or(abs(l.tss-p.tss) <= 1000
                                      AND l.strand != p.strand
                                      AND NOT (l.gene_start < p.gene_end
                                               AND p.gene_start < l.gene_end)), false)
                       AS divergent,
                   coalesce(bool_or(l.strand != p.strand
                                      AND l.gene_start < p.gene_end
                                      AND p.gene_start < l.gene_end), false)
                       AS antisense,
                   coalesce(bool_or(l.strand = p.strand
                                      AND l.gene_start < p.gene_end
                                      AND p.gene_start < l.gene_end), false)
                       AS sense_overlapping
            FROM lnc l
            LEFT JOIN pc p
              ON l.chrom = p.chrom
             AND (abs(l.tss-p.tss) <= 1000
                  OR (l.gene_start < p.gene_end AND p.gene_start < l.gene_end))
            GROUP BY l.gene_id
            """
        ).fetchall()
        class_map = {}
        raw_class_flags = {}
        for gene_id, divergent, antisense, sense in class_rows:
            raw_class_flags[gene_id] = (bool(divergent), bool(antisense), bool(sense))
            class_map[gene_id] = (
                "divergent" if divergent else
                "antisense" if antisense else
                "sense-overlapping" if sense else
                "intergenic"
            )

        self_overlap = {
            row[0]
            for row in connection.execute(
                f"""
                SELECT DISTINCT l.gene_id
                FROM lnc l
                JOIN read_parquet('{sql_path(links)}') r
                  ON CAST(r.gene_id AS VARCHAR) = l.gene_id
                JOIN read_parquet('{sql_path(catalog)}') c
                  ON CAST(c.cre_id AS VARCHAR) = CAST(r.cre_id AS VARCHAR)
                WHERE CAST(c.chrom AS VARCHAR) = l.chrom
                  AND CAST(c."end" AS BIGINT) > l.tss - 1000
                  AND CAST(c."start" AS BIGINT) <= l.tss + 1000
                """
            ).fetchall()
        }
        lnc_rows = [dict(zip(
            ["gene_id", "gene_id_versioned", "gene_symbol", "gene_biotype", "chrom", "strand", "tss", "gene_start", "gene_end"],
            row,
        )) for row in connection.execute("SELECT * FROM lnc ORDER BY gene_id").fetchall()]
        pc_rows = [dict(zip(
            ["gene_id", "gene_id_versioned", "gene_symbol", "gene_biotype", "chrom", "strand", "tss", "gene_start", "gene_end"],
            row,
        )) for row in connection.execute("SELECT * FROM pc ORDER BY gene_id").fetchall()]
    finally:
        connection.close()

    nearest = nearest_pc_records(lnc_rows, pc_rows)
    output_rows = []
    for row in lnc_rows:
        nearest_id, nearest_tss, nearest_distance = nearest[row["gene_id"]]
        output_rows.append(
            {
                "gene_id": row["gene_id"],
                "gene_id_versioned": row["gene_id_versioned"],
                "gene_symbol": row["gene_symbol"],
                "chrom": row["chrom"],
                "strand": row["strand"],
                "tss": int(row["tss"]),
                "gene_start": int(row["gene_start"]),
                "gene_end": int(row["gene_end"]),
                "locus_class": class_map[row["gene_id"]],
                "nearest_pc_gene_id": nearest_id,
                "nearest_pc_tss": int(nearest_tss),
                "nearest_pc_distance_bp": int(nearest_distance),
                "nearest_pc_distance_bin": distance_bin(int(nearest_distance)),
                "self_overlap_tss_1kb": row["gene_id"] in self_overlap,
            }
        )
    if len(output_rows) != EXPECTED_LNCRNA:
        raise RuntimeError(f"expected {EXPECTED_LNCRNA} output rows, found {len(output_rows)}")

    table = pa.Table.from_pylist(output_rows, schema=OUTPUT_SCHEMA)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    table_path = args.output_dir / "lncrna_strata.parquet"
    pq.write_table(table, table_path, compression="zstd", use_dictionary=True, row_group_size=10_000)
    if pq.read_schema(table_path) != OUTPUT_SCHEMA:
        raise RuntimeError("published lncRNA strata schema mismatch")

    class_counts = Counter(row["locus_class"] for row in output_rows)
    distance_counts = Counter(row["nearest_pc_distance_bin"] for row in output_rows)
    class_distance_counts = Counter(
        (row["locus_class"], row["nearest_pc_distance_bin"]) for row in output_rows
    )
    class_self_counts = Counter(
        (row["locus_class"], row["self_overlap_tss_1kb"]) for row in output_rows
    )
    script_path = Path(__file__).resolve()
    report = {
        "phase": "section 8 lncRNA locus strata",
        "status": "complete",
        "rules": RULES,
        "universe": {
            "definition": "full pinned GENCODE v50 gene set",
            "lncRNA_genes": EXPECTED_LNCRNA,
            "protein_coding_genes": EXPECTED_PROTEIN_CODING,
        },
        "counts": {
            "rows": len(output_rows),
            "locus_class": {name: class_counts[name] for name in RULES["precedence"]},
            "nearest_pc_distance_bin": {
                name: distance_counts[name] for name in ("<5kb", "5-50kb", "50-200kb", ">200kb")
            },
            "self_overlap_tss_1kb": {
                "true": sum(row["self_overlap_tss_1kb"] for row in output_rows),
                "false": sum(not row["self_overlap_tss_1kb"] for row in output_rows),
            },
            "by_class_and_distance_bin": [
                {"locus_class": locus_class, "distance_bin": distance_name,
                 "count": class_distance_counts[(locus_class, distance_name)]}
                for locus_class in RULES["precedence"]
                for distance_name in ("<5kb", "5-50kb", "50-200kb", ">200kb")
            ],
            "by_class_and_self_overlap": [
                {"locus_class": locus_class, "self_overlap_tss_1kb": flag,
                 "count": class_self_counts[(locus_class, flag)]}
                for locus_class in RULES["precedence"] for flag in (False, True)
            ],
        },
        "integrity": {
            "gtf_duplicate_eligible_gene_ids": duplicate_gtf,
            "eligible_tss_rows": int(tss_integrity[0]),
            "gtf_tss_joined_rows": int(tss_integrity[1]),
            "gtf_tss_metadata_mismatches": int(tss_integrity[2]),
            "v4_retained_link_rows": int(link_integrity[0]),
            "v4_wrong_method_rows": int(link_integrity[1]),
            "v4_fallback_rows": int(link_integrity[2]),
            "v4_duplicate_pairs": int(link_integrity[3]),
            "lncRNAs_matching_multiple_raw_class_conditions": sum(
                sum(flags) > 1 for flags in raw_class_flags.values()
            ),
            "lncRNAs_without_same_chromosome_pc_tss": 0,
        },
        "inputs": {
            name: {"path": str(path), "sha256": input_hashes[name]}
            for name, path in inputs.items()
        },
        "output": {
            "path": str(table_path.resolve()),
            "sha256": sha256(table_path),
            "rows": table.num_rows,
            "schema": [[field.name, str(field.type), field.nullable] for field in OUTPUT_SCHEMA],
        },
        "execution": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "script": str(script_path),
            "script_sha256": sha256(script_path),
            "threads": args.threads,
            "memory_limit": args.memory_limit,
            "duckdb_version": duckdb.__version__,
            "pyarrow_version": pa.__version__,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        },
    }
    report_path = args.output_dir / "lncrna_strata_report.json"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "rows": report["counts"]["rows"],
        "locus_class": report["counts"]["locus_class"],
        "self_overlap_tss_1kb": report["counts"]["self_overlap_tss_1kb"],
        "report_sha256": sha256(report_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

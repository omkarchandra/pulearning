#!/usr/bin/env python3
"""Consolidate intact v1/v2 downloads and the verified HGNC copy, create-only."""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def copy(source, target, expected):
    if sha(source) != expected:
        raise ValueError("source checksum differs: " + str(source))
    with Path(source).open("rb") as src, target.open("xb") as dst:
        for block in iter(lambda: src.read(1048576), b""):
            dst.write(block)
    if sha(source) != expected or sha(target) != expected:
        raise ValueError("source or copy changed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    root, out = a.project_root, a.output_dir
    out.mkdir(parents=True, exist_ok=False)
    old = root / "datasets/raw/go/2026-08-05/acquisition_20261011_v2"
    prior = json.loads((old / "acquisition_failure.json").read_text())
    records = {}
    for name, record in prior["completed_resources"].items():
        copy(old / name, out / name, record["sha256"])
        records[name] = dict(record, retained_from=str(old / name))
    sources = {
        "hgnc_complete_set.txt": ("datasets/raw/go/hgnc_snapshot_20261011/hgnc_complete_set.txt", "2b4224ea847df2fc6982f5b2a52804c5d92fbb8810b134afb636f6029452dc03"),
        "universe.parquet": ("experiments/20261011_eval_paired_v7/prepared/universe.parquet", "6e2a8c661e265deeb00c3b53ea33594e1f256989690528c00d49ae2a1b26b245"),
        "split.parquet": ("experiments/20261011_eval_paired_v7/prepared/split.parquet", "2b2b44e074078997a0223edc703d4094f10ade2cece5e36ca53fcfbe3d30bbd1"),
        "lncrna_strata.parquet": ("experiments/link_quality/20261011_lncrna_strata_v1/runs/9471069/results/lncrna_strata.parquet", "53bbf82ed00fcb95d5de2bd57bf364c03040e570f09f41db5d0b022ddda464e1"),
    }
    for name, (relative, expected) in sources.items():
        copy(root / relative, out / name, expected)
        records[name] = {"retained_from": str(root / relative), "sha256": expected,
                         "bytes": (out / name).stat().st_size}
    for name, source in {
        "hgnc.http_headers": root / "datasets/raw/go/hgnc_snapshot_20261011/hgnc_complete_set.txt.http_headers",
        "PROTOCOL.json": Path(__file__).with_name("PROTOCOL.json"),
    }.items():
        copy(source, out / name, sha(source))
        records[name] = {"retained_from": str(source), "sha256": sha(source), "bytes": source.stat().st_size}
    protocol = json.loads((out / "PROTOCOL.json").read_text())
    metadata = (out / "release-date.json").read_text().strip()
    if metadata != "{date: 2026-08-05}":
        raise ValueError("unexpected archived bundle metadata")
    with (out / "go-basic.obo").open() as h:
        header = []
        for line in h:
            if line.startswith("[Term]"):
                break
            header.append(line.rstrip("\n"))
    if [s for s in header if s.startswith("data-version:")] != ["data-version: releases/" + protocol["go_ontology_version"]]:
        raise ValueError("ontology version differs")
    records["go-basic.obo"]["header"] = header
    gaf_versions = []
    for name in ("HUMAN-uniprot.gaf.gz", "HUMAN-uniprot.gpi.gz", "goa_human_rna.gaf.gz", "goa_human_rna.gpi.gz"):
        with gzip.open(out / name, "rt") as h:
            header = [line.rstrip("\n") for line in h if line.startswith("!")]
        records[name]["header"] = header
        records[name]["gzip_full_stream_verified"] = True
        if name.endswith("gaf.gz"):
            if "!gaf-version: 2.2" not in header:
                raise ValueError("GAF format differs")
            versions = [s for s in header if s.startswith("!go-version:")]
            dates = [s for s in header if s.startswith("!date-generated:")]
            if len(versions) != 1 or "/releases/" + protocol["go_ontology_version"] + "/" not in versions[0]:
                raise ValueError("GAF ontology differs")
            if len(dates) != 1 or protocol["gaf_generation_date"] not in dates[0]:
                raise ValueError("protein/RNA source vintages differ")
            gaf_versions.append(versions[0])
    if len(set(gaf_versions)) != 1:
        raise ValueError("protein/RNA go-version headers differ")
    if "RNAcentral Release 27" not in (out / "rnacentral_release_notes.txt").read_text():
        raise ValueError("RNAcentral release differs")
    report = {"status": "source_integrity_verified", "resources": records,
              "go_bundle_release_date": protocol["go_bundle_release_date"],
              "go_ontology_version": protocol["go_ontology_version"],
              "gaf_generation_date": protocol["gaf_generation_date"],
              "previous_failed_acquisition_manifest_sha256": sha(old / "acquisition_failure.json"),
              "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(__file__),
              "rna_archive_status": "official matching mutable endpoint snapshot retained by full hash",
              "real_scoring_authorized": False}
    with (out / "acquisition_manifest.json").open("x") as h:
        json.dump(report, h, sort_keys=True, indent=2)
        h.write("\n")
    print(json.dumps({"status": report["status"], "manifest_sha256": sha(out / "acquisition_manifest.json")}))


if __name__ == "__main__":
    main()

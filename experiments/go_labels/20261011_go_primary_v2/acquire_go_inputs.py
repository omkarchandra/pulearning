#!/usr/bin/env python3
"""Retain the matching GO/GOA/RNAcentral sources without replacing v1 evidence."""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import urllib.request


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_checked(source, target, expected):
    if sha(source) != expected:
        raise ValueError("retained source checksum changed: " + str(source))
    with Path(source).open("rb") as src, target.open("xb") as dst:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            dst.write(block)
    if sha(target) != expected or sha(source) != expected:
        raise ValueError("source copy checksum changed")


def compressed_header(path):
    headers = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("!"):
                headers.append(line.rstrip("\n"))
    return headers  # Reading through EOF also validates the gzip CRC.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.project_root, args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    protocol = Path(__file__).with_name("PROTOCOL.json")
    records = {}
    try:
        old = root / "datasets/raw/go/2026-08-05/acquisition_20261011_v1"
        failure = json.loads((old / "acquisition_failure.json").read_text())
        for name, previous in failure["completed_resources"].items():
            copy_checked(old / name, out / name, previous["sha256"])
            records[name] = dict(previous)
            records[name]["retained_from"] = str(old / name)
        sources = {
            "goa_human_rna.gaf.gz": "https://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human_rna.gaf.gz",
            "goa_human_rna.gpi.gz": "https://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human_rna.gpi.gz",
            "goa_HUMAN_README": "https://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/README",
            "rnacentral_hgnc.tsv": "https://ftp.ebi.ac.uk/pub/databases/RNAcentral/releases/27.0/id_mapping/database_mappings/hgnc.tsv",
            "rnacentral_ensembl_gencode.tsv": "https://ftp.ebi.ac.uk/pub/databases/RNAcentral/releases/27.0/id_mapping/database_mappings/ensembl_gencode.tsv",
            "rnacentral_release_notes.txt": "https://ftp.ebi.ac.uk/pub/databases/RNAcentral/releases/27.0/release_notes.txt",
            "rnacentral_mapping_readme.txt": "https://ftp.ebi.ac.uk/pub/databases/RNAcentral/releases/27.0/id_mapping/readme.txt",
        }
        for name, url in sources.items():
            request = urllib.request.Request(url, headers={"User-Agent": "pu-learning-go-input/2.0"})
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.status != 200:
                    raise ValueError("source did not return HTTP 200")
                headers, resolved = dict(response.headers), response.url
                with (out / name).open("xb") as handle:
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        handle.write(block)
            size = (out / name).stat().st_size
            if headers.get("Content-Length") and size != int(headers["Content-Length"]):
                raise ValueError("truncated source: " + name)
            records[name] = {"requested_url": url, "resolved_url": resolved,
                             "bytes": size, "sha256": sha(out / name),
                             "http_response_headers": headers,
                             "upstream_url_mutable": "/GO/goa/HUMAN/" in url}
            print(json.dumps({"downloaded": name, "bytes": size, "sha256": records[name]["sha256"]}), flush=True)
        retained = {
            "hgnc_complete_set.txt": ("datasets/raw/remap/hgnc_complete_set.txt", "2b4224ea847df2fc6982f5b2a52804c5d92fbb8810b134afb636f6029452dc03"),
            "universe.parquet": ("experiments/20261011_eval_paired_v7/prepared/universe.parquet", "6e2a8c661e265deeb00c3b53ea33594e1f256989690528c00d49ae2a1b26b245"),
            "split.parquet": ("experiments/20261011_eval_paired_v7/prepared/split.parquet", "2b2b44e074078997a0223edc703d4094f10ade2cece5e36ca53fcfbe3d30bbd1"),
            "lncrna_strata.parquet": ("experiments/link_quality/20261011_lncrna_strata_v1/runs/9471069/results/lncrna_strata.parquet", "53bbf82ed00fcb95d5de2bd57bf364c03040e570f09f41db5d0b022ddda464e1"),
        }
        for name, (relative, expected) in retained.items():
            source = root / relative
            copy_checked(source, out / name, expected)
            records[name] = {"retained_from": str(source), "sha256": expected,
                             "bytes": (out / name).stat().st_size}
        header_source = root / "datasets/raw/remap/hgnc_complete_set.txt.http_headers"
        copy_checked(header_source, out / "hgnc.http_headers", sha(header_source))
        records["hgnc.http_headers"] = {"sha256": sha(header_source), "retained_from": str(header_source),
                                        "bytes": header_source.stat().st_size}
        copy_checked(protocol, out / "PROTOCOL.json", sha(protocol))
        records["PROTOCOL.json"] = {"sha256": sha(protocol), "bytes": protocol.stat().st_size,
                                    "retained_from": str(protocol)}
        metadata = (out / "release-date.json").read_text().strip()
        if metadata != "{date: 2026-08-05}" and json.loads(metadata).get("date") != "2026-08-05":
            raise ValueError("GO bundle release metadata differs")
        with (out / "go-basic.obo").open() as handle:
            obo_header = []
            for line in handle:
                if line.startswith("[Term]"):
                    break
                obo_header.append(line.rstrip("\n"))
        version = [line for line in obo_header if line.startswith("data-version:")]
        if version != ["data-version: releases/2026-07-26"]:
            raise ValueError("unexpected ontology internal version")
        records["go-basic.obo"]["header"] = obo_header
        for name in ("HUMAN-uniprot.gaf.gz", "HUMAN-uniprot.gpi.gz", "goa_human_rna.gaf.gz", "goa_human_rna.gpi.gz"):
            header = compressed_header(out / name)
            records[name]["header"] = header
            records[name]["gzip_full_stream_verified"] = True
            if name.endswith("gaf.gz"):
                if "!gaf-version: 2.2" not in header:
                    raise ValueError("unsupported GAF version")
                versions = [line for line in header if line.startswith("!go-version:")]
                if len(versions) != 1 or "/releases/2026-07-26/" not in versions[0]:
                    raise ValueError("protein/RNA GAF and ontology versions differ")
                dates = [line for line in header if line.startswith("!date-generated:")]
                if len(dates) != 1 or "2026-07-28" not in dates[0]:
                    raise ValueError("protein/RNA annotation vintages differ")
        if "RNAcentral Release 27" not in (out / "rnacentral_release_notes.txt").read_text():
            raise ValueError("RNAcentral release differs")
        report = {"status": "source_integrity_verified", "go_bundle_release_date": "2026-08-05",
                  "go_ontology_version": "2026-07-26", "gaf_generation_date": "2026-07-28",
                  "rnacentral_release": "27.0", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                  "script_sha256": sha(__file__), "original_failed_acquisition_manifest_sha256": sha(old / "acquisition_failure.json"),
                  "resources": records, "real_scoring_authorized": False,
                  "rna_archive_status": "matching official mutable endpoint retained create-only by content hash"}
        with (out / "acquisition_manifest.json").open("x") as handle:
            json.dump(report, handle, sort_keys=True, indent=2)
            handle.write("\n")
        print(json.dumps({"status": report["status"], "manifest_sha256": sha(out / "acquisition_manifest.json")}), flush=True)
    except Exception as error:
        with (out / "acquisition_failure.json").open("x") as handle:
            json.dump({"status": "failed", "error": str(error), "completed_resources": records,
                       "script_sha256": sha(__file__)}, handle, sort_keys=True, indent=2)
            handle.write("\n")
        raise


if __name__ == "__main__":
    main()

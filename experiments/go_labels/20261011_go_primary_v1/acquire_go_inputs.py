#!/usr/bin/env python3
"""Download one archived official GO release into a new, checksum-bound directory."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-date", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.release_date):
        raise ValueError("a dated GO release is required")
    datetime.strptime(args.release_date, "%Y-%m-%d")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    archive = f"https://release.geneontology.org/{args.release_date}"
    resources = {
        "release-date.json": "metadata/release-date.json",
        "HUMAN-uniprot.gaf.gz": "annotations/gaf/HUMAN-uniprot.gaf.gz",
        "HUMAN-uniprot.gpi.gz": "annotations/gpi/HUMAN-uniprot.gpi.gz",
        "go-basic.obo": "ontology/go-basic.obo",
    }
    records = {}
    try:
        for name, relative in resources.items():
            url = f"{archive}/{relative}"
            request = urllib.request.Request(url, headers={"User-Agent": "pu-learning-go-input/1.0"})
            path = args.output_dir / name
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.status != 200:
                    raise ValueError("official GO resource did not return HTTP 200")
                headers = {key: value for key, value in response.headers.items()}
                resolved_url = response.url
                with path.open("xb") as handle:
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        handle.write(block)
            if headers.get("Content-Length") and path.stat().st_size != int(headers["Content-Length"]):
                raise ValueError(f"truncated GO resource: {name}")
            records[name] = {"requested_url": url, "resolved_url": resolved_url,
                             "sha256": sha(path), "bytes": path.stat().st_size,
                             "http_response_headers": headers}
            print(json.dumps({"downloaded": name, "bytes": records[name]["bytes"],
                              "sha256": records[name]["sha256"]}), flush=True)

        metadata = (args.output_dir / "release-date.json").read_text()
        # The official endpoint currently serves {date: YYYY-MM-DD}, despite
        # declaring application/json. Preserve the original bytes and accept
        # only that exact shape or a normal JSON object with the same date.
        try:
            observed_date = json.loads(metadata)["date"]
        except json.JSONDecodeError:
            match = re.fullmatch(r"\s*\{date:\s*(\d{4}-\d{2}-\d{2})\}\s*", metadata)
            if not match:
                raise ValueError("unsupported official release-date metadata encoding")
            observed_date = match.group(1)
        if observed_date != args.release_date:
            raise ValueError("archived metadata does not match the requested release")
        for name in ("HUMAN-uniprot.gaf.gz", "HUMAN-uniprot.gpi.gz"):
            with gzip.open(args.output_dir / name, "rt", encoding="utf-8") as handle:
                header = []
                for line in handle:
                    if line.startswith("!"):
                        header.append(line.rstrip("\n"))
                # Iteration through EOF validates gzip CRC/truncation without
                # deriving any biological support counts.
            records[name]["header"] = header
            records[name]["gzip_full_stream_verified"] = True
        with (args.output_dir / "go-basic.obo").open(encoding="utf-8") as handle:
            ontology_header = []
            for line in handle:
                if line.startswith("[Term]"):
                    break
                ontology_header.append(line.rstrip("\n"))
        if not any(line.startswith("data-version:") and args.release_date in line for line in ontology_header):
            raise ValueError("ontology header is not from the requested GO release")
        records["go-basic.obo"]["header"] = ontology_header
        report = {"status": "downloaded_and_integrity_checked", "release_date": args.release_date,
                  "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                  "script_sha256": sha(__file__), "resources": records,
                  "label_admission": "pending source/identifier/ontology replay",
                  "real_scoring_authorized": False}
        with (args.output_dir / "acquisition_manifest.json").open("x") as handle:
            json.dump(report, handle, sort_keys=True, indent=2)
            handle.write("\n")
        print(json.dumps({"status": report["status"], "manifest_sha256": sha(args.output_dir / "acquisition_manifest.json")}), flush=True)
    except Exception as error:
        with (args.output_dir / "acquisition_failure.json").open("x") as handle:
            json.dump({"status": "failed", "error": str(error), "completed_resources": records,
                       "release_date": args.release_date, "script_sha256": sha(__file__)}, handle, sort_keys=True, indent=2)
            handle.write("\n")
        raise


if __name__ == "__main__":
    main()

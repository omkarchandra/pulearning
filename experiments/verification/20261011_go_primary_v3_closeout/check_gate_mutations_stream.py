#!/usr/bin/env python3
"""Create-only stream-copy canaries for hosts that prohibit fixture symlinks."""

import argparse
import json
from pathlib import Path
import sys


def write(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")


def copy_named(source, target, names, exclusions):
    target.mkdir(exist_ok=False)
    for name in sorted(set(names) - set(exclusions)):
        with (source / name).open("rb") as src, (target / name).open("xb") as dst:
            for block in iter(lambda: src.read(1048576), b""):
                dst.write(block)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.adapter_root.resolve()))
    from src.evaluation.go_labels import check_go_label_gate, file_sha256
    args.output_dir.mkdir(exist_ok=False)
    source_hash = file_sha256(args.input_dir / "PROTOCOL.json")
    label_hash = file_sha256(args.result_dir / "propagated_positives.json")
    acquisition = json.loads((args.input_dir / "acquisition_manifest.json").read_text())
    changed = args.output_dir / "changed_source"
    copy_named(args.input_dir, changed, list(acquisition["resources"]) + ["acquisition_manifest.json"], {"PROTOCOL.json"})
    protocol = json.loads((args.input_dir / "PROTOCOL.json").read_text())
    protocol["positive_evidence_codes"].append("IEA")
    write(changed / "PROTOCOL.json", protocol)
    results = {}
    try:
        check_go_label_gate(changed, args.result_dir)
    except ValueError as error:
        if "GO source changed: PROTOCOL.json" not in str(error):
            raise
        results["source_mutation"] = {"rejected": True, "error": str(error)}
    else:
        raise AssertionError("source mutation passed")
    manifest = json.loads((args.result_dir / "label_manifest.json").read_text())
    forged = args.output_dir / "forged_labels"
    copy_named(args.result_dir, forged, list(manifest["outputs"]) + ["label_manifest.json"], {"label_manifest.json", "propagated_positives.json"})
    positives = json.loads((args.result_dir / "propagated_positives.json").read_text())
    positives[sorted(positives)[0]].append("ENSG_FORGED_LABEL")
    write(forged / "propagated_positives.json", positives)
    manifest["outputs"]["propagated_positives.json"] = {"sha256": file_sha256(forged / "propagated_positives.json"),
                                                       "bytes": (forged / "propagated_positives.json").stat().st_size}
    write(forged / "label_manifest.json", manifest)
    try:
        check_go_label_gate(args.input_dir, forged)
    except ValueError as error:
        if "GO positives do not rederive" not in str(error):
            raise
        results["rehash_forged_positive"] = {"rejected": True, "error": str(error)}
    else:
        raise AssertionError("forged positive passed")
    if file_sha256(args.input_dir / "PROTOCOL.json") != source_hash or file_sha256(args.result_dir / "propagated_positives.json") != label_hash:
        raise AssertionError("originals changed")
    results.update(status="passed", originals_unchanged=True, script_sha256=file_sha256(__file__),
                   adapter_sha256=file_sha256(args.adapter_root / "src/evaluation/go_labels.py"),
                   fixture_method="exclusive stream copies; no rename, unlink or symlink")
    write(args.output_dir / "verdict.json", results)
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()

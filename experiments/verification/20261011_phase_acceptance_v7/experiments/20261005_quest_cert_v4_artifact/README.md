# Quest Phase 3 v4 certification artifacts (local copies)

Pulled 2026-10-06 from `/home/mes7956/projects/pulearning/experiments/` on Quest.
Originals remain on Quest project storage. At initial import these copies were
compared by parsed row/hash fields, not by file SHA-256; byte identity had not
yet been established. All six copies were subsequently SHA-256 compared to
their Quest sources on 2026-10-11 and matched byte-for-byte; see
`experiments/20261011_eval_repair_v1/certification_copy_verification.json`.
The added `manifest.json` binds the original certified bundle, SHA-256
`21cccb8082d9c5217602507601209098d30eec71ce99eadf24e2ec24aef69d2f`.

| File | Quest source | Content |
|---|---|---|
| `build.json` | `.../20261005_quest_cre_phase3_v4/build.json` | Published bundle: 225,784,420 dependencies, 549,600 values/QC rows, 10 intrinsic features, 14.85 GiB peak spill |
| `prepare_inputs.json` | `.../20261005_quest_cre_phase3_v4/prepare_inputs.json` | Input audit: 11,289,221 admitted of 15,235,813 links; excluded-link biotype breakdown; pinned input hashes |
| `validate_full.json` | `.../20261005_quest_cre_phase3_v4/validate_full.json` | Separate-process FULL replay: status valid, all rows replayed, 1:26:02 wall, 10.9 GiB peak RSS, 0 spill |
| `validate_sampled_v5.json` | `.../20261005_quest_cre_phase3_v5/validate_sampled.json` | Sampled replay: 1,848/54,960 genes (3.36%), 8,547,380 dependencies, status valid |
| `focused_tests.log` | `.../20261005_quest_cre_phase3_v6/focused_tests.log` | 41 passed, 1 skipped (chrom-ambiguity fix, §14.2 null, atomic publish) |
| `link_quality.json` | `.../20261005_quest_cre_phase3_v4/7_link_quality_v8/link_quality.json` | Historical §7 prototype: precision 0.463/0.271/0.151 at k=1/5/20; parsing/metric semantics still require review before scientific use |

Logical content hashes (from `validate_full.json`):
- view: `b5bd715b401a904c0259297bff06eba7fe31a5e0e7a85a15a5eb91befc41b0eb`
- aggregation_qc: `9c83d25cddf807fd469a449fecc8aa761f5c1181b431f5843fafe2a6f7d3ddb1`
- dependencies: `ecaccf7aa2b98bf5da66996bae7f0ed93d9d717ff9dc7a83e6e10123159c2e48`

## Completed supplementary fingerprints (2026-10-11)

Quest job 9462630 completed the schema-bound full-key sort of all 225,784,420
dependency rows. Sorted SHA-256:
`9cff3122dd69ce86eb75e5e790c39ed50d12526a89ddcf359d93090dce2447a4`.
The all-three-artifact sidecar passed in job 9464781, binding original physical
hashes, file-order logical hashes, sorted fingerprints, schemas/full keys and
the retained full-replay certificate. Sidecar SHA-256:
`5409c19204fbe1c41a74a4fd7d353c118a070548de3be35bc7d02b261e4bf21d`.

Copies are retained under
`experiments/20261011_paired_phase_evidence/files/experiments/fingerprint/`
(dependency report in `20261011_eval_repair_v1/`, sidecar in
`20261011_final_sidecar/`). These are supplementary reproducibility evidence;
the original artifact identities and full semantic replay remain authoritative.
See `doc/evaluation-repair-20261011.md` for §39 acceptance and timing/settings.

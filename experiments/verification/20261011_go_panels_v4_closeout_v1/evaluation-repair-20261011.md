# Evaluation evidence repair — 2026-10-11

## Current review disposition

**The evaluation harness phase is complete and accepted as §38.6.** The v7
snapshot, paired-canary XML, twenty completed runs and independent replay are
the standing evidence. The original v2 report remains rejected as written;
its reproduced numerical results are historical, not retroactively re-certified.
The full CRE fingerprint/sidecar is complete and accepted as **§39**.

The separate reviewer found the missing same-ledger file-audit pairing in §38;
the corrected canary passed natively and was accepted in §38.5. The required
source-matched v7 rerun then verified all twenty runs / 100 folds. Claude's
completed §38.6 response was read on 2026-10-11 in the exact PU review thread.
Formal scientific scoring still requires official labels and supported panels.

## Corrections accepted by Claude (§32)

The current review thread is
<https://claude.ai/chat/9216c204-4aba-4266-8b4f-c5cedb3913dd>.
The correction was delivered and its reply read on 2026-10-11.

Re-read on Quest before the correction:

| Retained report under `experiments/20261005_quest_cre_phase3_v6/` | Actual configuration/result |
|---|---|
| `sealed_selectors_confounder_matched/sealed_run.json` | `negative_mode: centroid`, seed 20261004, breast AP 0.064592 |
| `sealed_selectors_biotype_stratified/sealed_run.json` | Same centroid configuration and AP; directory name is misleading |
| `sealed_selectors_debug/sealed_run.json` | Genuine confounder-matched AGS1, AP 0.000790, but all candidates used rather than N=P |

The previously reported **sealed breast AP 0.149 with both audits clean** is
retracted. The 0.149 value belonged to an earlier unsealed five-run diagnostic.
The manual breast AP 0.082711 came from a different all-candidate model, and
its in-memory audit object never observed fitting. A zero-event count from
that object is vacuous. The saved spy test did not execute: it referenced an
unimported function and passed a dictionary as the universe. No selector-property
or absence-of-leakage conclusion follows from those claims.

`logs/v4_fingerprint.log` on Quest says the process was Killed; no corresponding
v4 report was produced. The bounded sorted-content tool initially omitted the
schema header, so it did not use the feature store's logical hash format.

## Redo implementation and native execution

New source: `releases/20261011_eval_repair_v1/` on Quest; 172 files bound by
`SOURCE_SHA256SUMS`, whose SHA-256 is
`7b2a0e9abfccbd3271b47bdddca277e8da4bb1c8cf5b4552d96b12b36644ae10`.
This source is staged as a new directory rather than editing an old release.

Implemented paths:

- `src/evaluation/prepared_labels.py`: a separate preparation process writes
  shuffled training labels and true test labels in independent, full-key-bound
  files. Fitting accepts no consolidated true-label map.
- `scripts/run_sealed_selector_comparison.py`: prepare, fit-worker, and
  post-run artifact review commands for five breast runs of four selectors.
- `src/evaluation/sealed_predictions.py`: test reads require their own full
  configuration key; prediction/label hashes are checked; mutation and
  cross-configuration reads are rejected. Python file-open auditing is scoped
  explicitly; this is not an OS sandbox or an audit of arbitrary native IO.
- Exact N=P sampling from each selector's own candidate set, with shortages
  treated as failures. Matcher scaling is inner-train-only. Each fold writes
  anchors, pool, candidates, chosen negatives, counts, hashes and seeds.
- The memory canary pairs clean/planted fits on one true-label proxy. The
  earlier file canary detected a planted consolidated-label read but did not
  pair it with a clean fit on its ledger; that claim was withdrawn in §38.
  The corrected paired test and rerun are described below. The spy checks the
  actual sealed entry point and the pool still contains true training positives.
- Publication counts are explicitly supplied and pinned to
  `97f9fa1f0cad5e5225a72443e01a489e42deb992eac8001a98e66f0d30e160eb`.
  Their mere presence on disk no longer changes a five-column feature arm.
- `src/features/sorted_fingerprint.py`: spillable full-key sorting, unique-key
  checking, schema header and the exact store canonical value encoding.

Native v1 checks/preparation completed in job **9462517**: **61 passed, 1
skipped**, 42.69 s in `experiments/20261011_eval_repair_v1/native_tests.log`.
The JUnit XML records zero failures/errors and verifies all four entry-point
spies, the memory pairing, file-read detection, cross-configuration blocking
and mutation rejection. Claude accepted it as §33; §38 later narrowed the
file-audit interpretation after independent source review.

The follow-up source `releases/20261011_eval_repair_v2/` contains 173 files and
has source-manifest SHA-256
`0aac9e946ba831a2c4b91de7c307b7b06c4618ae7120ff419d59e4a7de69394b`.
It adds a gated prepared real-label path through the same worker, a toy real
fit, actual-release positive-set rederivation, recomputed panel content, and
post-run AP replay from sealed prediction payloads. Native job **9462989**
completed `0:0`: **65 passed, 1 skipped** in **30.33 s**, re-read from
`experiments/20261011_eval_repair_v2/native_tests.log`. Preparation produced
the six-column input matrix and twenty tasks (four selectors × five runs).

The five-run breast probe completed on this one v2 snapshot in array
**9463331**. Summary job **9463944** and independent artifact-verifier job
**9463950** completed `0:0`. The verifier re-read twenty run files and all
100 folds, reproduced anchors/pools/seeds/equal-count draws and full-key test
opens, and recomputed AP using a separate threshold-group algorithm. The
maximum difference from six-decimal run APs was 4.55e-7, within rounding.

Breast evaluation domain: 54,710 genes, 4,686 provisional positives. All
selectors use six pinned columns, classifier seed 42, RF 100 trees/depth 6,
and run permutation/sampling seeds 20261005–20261009. P=N by fold is
`[2881, 2927, 2677, 2838, 2941]`. Every pool contains 2,434–2,714 true training
positives, so the hypothesized exclusion of true positives is refuted for
these actual runs.

| Selector | Five-run mean AP | Descriptive offset from retired random-score reference |
|---|---:|---:|
| Centroid | 0.0487916 | −0.0370362 |
| Uniform | 0.1423246 | +0.0564968 |
| Confounder-matched | 0.1507546 | +0.0649268 |
| Biotype-stratified centroid | 0.0653504 | −0.0204774 |

The retired reference mean is 0.0858277523. These are one-disease shuffled
diagnostics; they are not predictive results or a random-score leakage gate.
The matched-minus-uniform mean difference is 0.00843; no mechanism or
significance claim follows from that number.

Quest evidence:

- `experiments/20261011_eval_repair_v2/selector_comparison.json`, SHA-256
  `b9f861c3a53973b88947224aea50fed295ef547ba262aeb3aac65c57b7eb83f5`.
- `experiments/verification/20261011_eval_repair_v2/verdict.json`, SHA-256
  `de36fcf30de2a99a579643c37309b865098ba137b191ef4bbdeb474a7538e1b5`.
- `experiments/20261011_eval_hardened_v5/v2_process_isolation_confirmation.json`,
  SHA-256 `5b2262a155bb0cc5c300305cacc2ce8517bf8b4893e61a60744a5d7a58739bd6`.
  Twenty distinct completed Slurm job IDs and twenty retained standalone Python
  command reports establish a separate Python process per array task; individual
  PIDs were not retained. The native-test job also used a different process.

Claude accepted the scoped comparison as **§34**. The read-only verifier is a
separate process/code calculation, not a separate model agent. The process
confirmation addresses the later audit-hook lifetime defect described below.
The earlier completed reply confirms that twenty distinct standalone worker
jobs establish process isolation and that the §34 numbers stand. The later
§38 review reproduced those values independently from sealed predictions and
prepared test labels but withdrew the file-audit claim until a genuinely paired
test and source-matched comparison are reviewed. Benchmark-adapter work waits
for the primary-task decision.

Local copies of 547 completed evidence files, including prepared inputs, labels,
selection artifacts, sealed predictions, ledgers and verifier output, are in
`experiments/20261011_completed_evidence_v2/files/`. Every copied payload was
SHA-256 checked; archive SHA-256 is
`08f66e307ce00ec5cc7b151a7747ee2b8fbdbf8ae14184850b5e98b19dac131a`.

## Final source and corrective checks

The new compatibility entry points `run_16_4_selectors.py` and
`check_28_selector_audits.py` dispatch to the shared sealed protocol. Their
former implementations remain available in preserved snapshots for inspection;
the old claims are not re-certified.

An added unbound-other-configuration test-file canary exposed a persistent
CPython audit-hook lifetime issue. Job **9464657** failed with **10 failed,
57 passed, 1 skipped**; its v4 source/logs are retained. A weak-reference hook
and explicit fixture teardown fix the lifetime, and unknown test-label files
cannot borrow another key's seal. The hardened source
`releases/20261011_eval_hardened_v5/` contains 175 checksum-bound files:
source-manifest SHA-256
`ecfe0944c44a5b2c70c450c1f8a56eb677ea7a3a6266316531125c191b69c615`.
Job **9464776** passed **68 tests, 1 skipped** in **25.98 s**, `0:0`.
V2 guarded job **9463604** separately passed **219 tests** in **23.49 s**.
The historical APs remain bound to v2, whose twenty workers were separate
processes. V5's test pass did not establish the missing file-audit pairing.

### §38.1 paired-file remedy

The new canary runs five clean folds with one active `SealLedger` and its one
installed hook, asserts zero reads of the protected consolidated-label source
and `order_verified`, then plants a source read inside the second fit. The
same ledger records exactly one such attempted read and raises from fitting.
The two fits have distinct prepared configuration keys to preserve create-only
selection/prediction artifacts. Legitimate prepared training files and authorized
post-seal test files are recorded and permitted; "zero" refers to forbidden
consolidated-source reads.

V6 native job **9468582** failed: **1 failed, 67 passed, 1 skipped**. Its clean
fit passed, but the second invocation collided with the existing immutable
selection key before fitting. That release/log/XML is retained. V7 corrects
the test setup with separately prepared keys while keeping the ledger/hook.

NEW source `releases/20261011_eval_paired_v7/`, 175 files, manifest SHA-256
`4d9e0e4c1b5663080d8183ad97024785861eaaf62e74ba538c2993f6301fc47d`.
Native job **9468835** passed **68 tests, 1 skipped** in **61.37 s**, `0:0`.
The native XML SHA-256 is
`1db46c309223b3ed03502d1226d37c32314dd0c85dccca0d82613ddeba6018d7`.

`audit_source_diff_from_v2.json` and `sealed_predictions.py.v2-v7.diff` record
the actual audit-module difference:
`0f2bbda930f1b726576424218644c2fe43b3338b58defbcd9fcb73e13b017e0e`
→ `33db6e8de399b0e55e038ccbe4c2c975e56b604a16695563b85e11f256834f87`.
The full-key/unbound-file enforcement and weak-reference lifecycle fixes changed
the audit path, so the pairing does not retroactively certify v2. Fresh array
**9469050** reran all twenty fits on v7; summary **9469060** completed afterward.
New run/summary JSONs embed their own script hash and Slurm job/array/task IDs
as well as the source-manifest checksum.
The separate reviewer's canary verdict was **VERIFIED**, based on the exact
v7 source, native XML and identical before/after source-check logs; Claude
accepted it as §38.5. The completed fresh comparison and final acceptance follow.

### Standing v7 shuffled reference (§38.6)

All **20** array tasks in **9469050** completed `0:0`. Summary **9469060** and
read-only artifact-verifier **9469506** also completed `0:0`. The verifier
checked **175** source files, **20** run files and **100** folds, replayed the
declared anchors/pools/seeds and exact N=P draws, checked full-key test-label
opens, and recomputed AP with its separate threshold-group algorithm. Maximum
AP rounding discrepancy was **4.5472313918437646e-7**. Each run covers **54,710**
genes / **4,686** provisional positives, and every fold's candidate pool retains
true training positives (range **2,434–2,714**).

| Selector | v7 five-run mean AP |
|---|---:|
| Centroid | 0.0487916 |
| Uniform | 0.1423246 |
| Confounder-matched | 0.1507546 |
| Biotype-stratified centroid | 0.0653504 |

These are the standing **one-disease shuffled-label references**. The separate
reviewer also opened the prepared train/test labels, selection payloads, v7/v2
sealed predictions, ledgers, declared inputs, native XML and audit-runtime diffs.
Its verdict was **VERIFIED WITH NOTES**, errors empty: full-precision AP deltas
from v2 were exactly zero. Some probabilities differ by last-bit floating-point
roundoff (maximum **4.440892098500626e-16**), so prediction-file byte identity is
not claimed. Claude recorded this verdict as **§38.6** and marked the evaluation
harness complete and certified. The supported audit scope remains the declared
Python worker/IO path.

Authoritative Quest evidence under `experiments/20261011_eval_paired_v7/`:

- `selector_comparison.json`, SHA-256
  `7966e58e75479db9c9cd98c7e5cb5bf836f30cadd4584311f8ef7e60b02a3d19`.
- `independent_artifact_verdict.json`, SHA-256
  `691a677bd91db983ed36b2fa558e7f1f958449e679b92191be9c89154037178c`.
  This is the retained separate-code/process verifier; the additional reviewer's
  browser-delivered verdict is a distinct review.
- `native_tests.xml`, SHA-256
  `1db46c309223b3ed03502d1226d37c32314dd0c85dccca0d82613ddeba6018d7`.
- `audit_source_diff_from_v2.json` and the exact runtime diffs.

Local `experiments/20261011_paired_phase_evidence/files/` retains **1,906**
SHA-256-checked files, including complete v7 inputs/labels/selection/prediction/
ledger artifacts, source snapshots, pre/post source-check logs and the failed
v6 attempt. Archive SHA-256:
`4db51f282551a2e00c5cfe01a3e75c6d0e388d1b957b7abe1bcf4cdb2464da90`.
`experiments/20261011_eval_paired_v7/v7_comparison_vs_v2.json` binds the new
summary/verifier and records all twenty rounded run APs matching v2.

The final local closeout check in `phase_closeout_recheck.json` rehashed all
1,906 imported payloads and the archive, verified 175 frozen source files and
22 same-job pre/post source seals, and compared all 100 v2/v7 prediction
payloads. The current declared source also matches v7 byte-for-byte. APs
recomputed from both versions in one local environment are identical under
both sklearn and the threshold-group algorithm. The two retained native
verdict JSONs differ in one AP by **2.7755575615628914e-17** (one last-bit
roundoff); thus cross-process JSON-level full-precision byte equality is not
claimed. Every reported six-decimal AP and selector mean reproduces. The
phase acceptance is retained in `review_acceptance.json` beside that check.

Environment and source changes are recorded in
`experiments/20261011_eval_hardened_v5/environment_and_source_provenance.json`.
The Python-label-IO ordering checks cover the declared worker path. They do
not constitute an OS sandbox or arbitrary native-IO audit.

## Sorted-fingerprint evidence

Golden job **9464148** completed `0:0` in **2m22s**. Three exact-encoding tests
passed, including signed zero/nonfinite/null/binary/Unicode cases. All six
artifacts from the two bounded builds, plus the full v4 values and QC tables,
matched the feature store's independent canonical encoder. A float-cache
signed-zero collision was corrected; that change is inapplicable to the
integer/string/null dependency schema used by the long v1 job.

- Golden verdict SHA-256:
  `fee3c6bccf5862390468e1b299424dcaddce33014f0ba6a8526b7bd039607ff7`.
- Both fixture dependency sorted hashes:
  `b2c921fc09bd8ed676e88204cbd40dc055fcbfa23b7c3fe3daaa8e7127d5bd24`.
- V4 values sorted hash:
  `564fd5badf321c8f64e55f9b0efe3ad0ab2ed1543e3e86618b06b303c22d56d6`.
- V4 QC sorted hash:
  `5b9f4f7fdd88dc7d9fa81fb1151059c08b9d4fc589013290d6c13f0df098786c`.

Full 225,784,420-row dependency fingerprint job **9462630** completed `0:0` in
**1:05:44** overall. The hashing process took **1:04:29**, **16,263,884 KiB
peak RSS (15.51 GiB)**; JSON reports 3868.1 s. Settings: four threads, 12GB
DuckDB limit, insertion-order preservation, 10,000-row batches and scratch spill.
Sorted dependency SHA-256:
`9cff3122dd69ce86eb75e5e790c39ed50d12526a89ddcf359d93090dce2447a4`.
The completed report's file SHA-256 is
`94ce49c658788e38b3cc02a956b97a7a1a2ccc03c9b2a126d49ba9369ede837e`.

Sidecar job **9464781** completed `0:0`, `status: verified`, binding all three
original physical artifact hashes, file-order logical hashes, sorted fingerprints,
schemas/full keys and the retained full-replay certificate. Sidecar file SHA-256:
`5409c19204fbe1c41a74a4fd7d353c118a070548de3be35bc7d02b261e4bf21d`.
Original publication settings were not recorded in the original manifest;
the sidecar states that explicitly. Sorted content remains auxiliary
reproducibility evidence, with original bytes and full semantic replay retained.

Receipt job **9466206** completed `0:0`; its source/artifact-integrity receipt
SHA-256 is `4d72da45501587c936b6b42481c32bfe1feb03cac7231d90e20d2c4a356b407c`.
It verifies all five preceding source manifests/files and the cited completed
artifacts, but does not establish the paired-canary requirement later identified
by §38. The original receipt is retained with that scope correction.

Local export `experiments/20261011_final_phase_evidence/files/` contains **963**
SHA-256-checked files, including all five frozen source snapshots, complete
new logs/XML, the failed v4 follow-up, completed fingerprint/sidecar and receipt.
Archive SHA-256:
`4d5076db69f980fd0eb494096896642ed48185ee47042b47c373087cac168078`.
Claude accepted the full CRE fingerprint/sidecar in **§39**. The distinct
evaluation baseline requirement was subsequently closed on v7 in **§38.6**.

The independent publication audit at
`experiments/20261011_eval_repair_v1/publication_count_audit.json` confirms
3,339,654 human source rows, 3,154,880 distinct mapped gene–PubMed pairs,
24,704 mapped genes and zero retained-count mismatches. It binds the raw
download, GENCODE metadata, certified transcript table and derived count file.

The actual DisGeNET adapter schema and allowlist requirements are recorded in
`doc/disgenet-gate-input-contract.md`. The official-release allowlist remains
unfrozen pending that release; the nominal toy source is not a scientific
source-scope decision.

## Scientific hold

Omkar confirmed **gene function prediction** on 2026-10-11. GO is the primary
target, reviewed in §41.1/§41.3; DisGeNET is a secondary application. The
historical BMC-paper question and DisGeNET credentials are outside the primary
GO task's critical path.

Human GO preparation v3 completed natively in **9479662**, `0:0`: six tests,
separate-code/process full raw-source replay, a source-rederivation gate and
both malicious-input canaries passed. The verifier checked 957,649 annotations,
42,695 product IDs and 16,996 support records. All eight native/local label,
audit and panel payloads are byte-identical. Retained evidence and source
bindings are in `experiments/go_labels/20261011_go_primary_v3/README.md`.

The v3 single-linkage=0.10 proposal produced only 28 clusters and remains
retained as incomplete. It was not §13.3's average-linkage=0.30 rule. The
prospective §42.2/§42.4 v4 revision completed in **9481056**, `0:0`: eight
tests, separate-code/process full raw and panel replay, source-rederivation
gate and both canaries passed. Both eligibility variants fill 40 development
and 10 final terms at the first per-aspect UPGMA=0.30 step; no fallback was
needed. The strict panel has ≥25 protein-coding positives in every fold for
all 50 selected terms. Evidence: `experiments/go_labels/20261011_go_panels_v4/`.

Formal GO scoring requires a selected supported panel, certified-feature
admission and a GO sealed-worker configuration. Experimentally supported
mapped lncRNAs number 25/22/11 in BP/MF/CC under this filter, with
7/5/4 in the isolated lncRNA stratum. No candidate term reaches 25 lncRNA
positives. Every reported AP retains the ≥25-positive gate in its actual
scoring domain; ≥5 per fold is panel admission only.

Omkar instructed us to retain lncRNA function prediction and ask Claude to
read his gene-function prediction paper. That exact instruction was delivered
in the PU thread; paper-specific implementation is pending the source review.
No GO models were fitted. The random-score reference remains descriptive only; full CRE
replay, label isolation, source-derived labels and predictive validity have
their separately stated scopes.

All six original local certification copies have now been SHA-256 compared to
their Quest sources with byte identity confirmed in
`experiments/20261011_eval_repair_v1/certification_copy_verification.json`.
The original bundle manifest is also retained locally; SHA-256
`21cccb8082d9c5217602507601209098d30eec71ce99eadf24e2ec24aef69d2f`.

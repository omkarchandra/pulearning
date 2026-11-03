# Quest Phase 3 CRE build — 2026-10-05

## Current disposition (updated 2026-10-11)

The v4 genome-wide bundle is published. The retained separate-process report
`experiments/20261005_quest_cert_v4_artifact/validate_full.json` records
`status: valid`, full replay of 549,600 values, 549,600 QC rows, and
225,784,420 dependencies. `build.json` reports the same output counts.
Sampled replay is supplementary and is retained in `validate_sampled_v5.json`.

Jobs 8683151 and 8725668 ended FAILED because supplementary steps failed after
successful validation; their terminal Slurm state must not be described as a
successful complete batch workflow. Corrective focused job 8731301 passed.
The original bundle and its permanent GPFS backing directory are retained.

The dependencies file-order logical hash is valid for that artifact, but
four-thread publication produced run-dependent row order in a bounded fixture.
A schema-bound sorted fingerprint completed in job 9462630, `0:0`, and the
three-artifact certificate-bound sidecar passed in job 9464781. Dependency sorted
SHA-256 is `9cff3122dd69ce86eb75e5e790c39ed50d12526a89ddcf359d93090dce2447a4`.
See `doc/evaluation-repair-20261011.md` for timings, source/sidecar bindings and
the separate evaluation canary remedy.
Earlier sorted row-only hashes omit the schema and are superseded as
reproducibility evidence. This does not replace the full semantic replay gate.

The v4/v6 code release directories were modified after their original source
checksums were created. Their original test records are historical evidence;
those directories cannot currently be called unchanged frozen snapshots.
New checks use new checksum-bound releases.

## Submission history

Current run ID: `20261005_quest_cre_phase3_v4`.

- Project: `/home/mes7956/projects/pulearning` (resolves to `/gpfs/projects/p32724/pulearning`).
- At submission, `releases/20261005_quest_cre_phase3_v4/` had 168 files checked against `SOURCE_SHA256SUMS`; subsequent edits invalidate an unchanged-snapshot claim.
- Verification job: **8683150**, completed with exit `0:0` on `qnode0150` in 5m30s.
- Genome-wide job: **8683151**, released after successful verification, later FAILED in a supplementary step after full validation.
- Current certification evidence is the separate full-replay JSON cited above.

Current-release native verification results (job 8683150, exit `0:0`):

| Check | Result |
|---|---|
| Full project suite | 459 passed, 1 opt-in skip; 57.37 s |
| Project-filesystem atomic publication probe | Passed |
| 100k-link / 10-feature gate | Passed; 100k values, 100k QC, 2M dependencies |
| Scale wall time | 177.849 s |
| Publisher / validator peak RSS | 515.371 / 468.281 MiB |
| Publisher reported spill | 22,347,776 bytes |
| Independent scale validation | `valid`, full replay |
| Guarded evaluation suite | 219 passed; 37.58 s |

Machine-readable copies are in `experiments/20261005_quest_cre_phase3_v4/`.
The complete raw logs, JUnit XML, environment freeze and `/usr/bin/time` reports
are retained in the corresponding Quest directory.

The second-release build 8677346 prepared all real inputs successfully, then
failed at the all-feature ordered aggregation with DuckDB's 8 GB memory limit
(18.70 s publisher time, 8,377,348 KiB peak RSS; job exit `1:0` after 46 s).
No genome-wide bundle was published. Prepared measurements contain 23,488,540
rows and are retained under the second release's input directory.

The third release bounds ordered aggregation to one feature at a time, caching
the feature-independent candidate statistics. Independent QC replay also visits
every feature separately. CRE-ID ordering, compensated sums, coverage arithmetic
and final canonical output ordering are preserved. Stage/feature progress is
written to stderr so the Slurm log identifies the active operation. A small
in-memory arithmetic/missingness check passed before submission. Third-release
verification passed (458 tests / 1 skip, 100k scale, 219 guarded tests), but real
aggregation still exceeded 8 GB on its first feature. A one-thread real-data
probe also failed at that limit.

The fourth release materializes one feature's joined rows in spillable storage,
sorted by gene batch, then computes ordered sums over at most 512 complete gene
neighborhoods at a time. Independent QC replay uses that bounded scope with its
separate source-recomputation query. A shuffled-input, multi-batch test compares
every value and QC field against the pandas oracle, including no-link genes and
partial observations across batch boundaries.

Before the full build, native Slurm probe **8683294** exercised the previously
failing feature against all 11,289,221 admitted real links: **54,960 gene results
in 10.966 s at 8 GB / 4 threads**. The full fourth-release run has now completed
all ten feature aggregations. Earlier failed attempts and their reports are
retained in the v1, v2 and v3 experiment directories.

The retained first attempt (`20261005_quest_cre_phase3_v1`) ran natively on
`qnode1026`: **424 passed, 25 failed, 1 skipped in 87.20 s**. Publication failed
because Quest GPFS rejects `renameat2(RENAME_NOREPLACE)` with `EINVAL`.
Verification job 8676175 exited 1 and dependent build 8676176 was cancelled
before executing. The scale and guarded evaluation steps were not reached.

The layout and scheduler settings follow the inspected `ear_dev/scripts/slurm/`
and `ear_dev/scripts/sh/` examples: account `b1042`, partition `genomics`, one
node, four CPUs. The build requests 32 GiB and 12 hours, with an 8 GB DuckDB
limit. A dedicated Python 3.12 virtual environment is installed under
`envs/<run-id>`; resolved package versions are retained with the run.

## Corrected real-input contract

The earlier draft job pointed `--measurements` at a catalog without measurement
columns and tried to use every raw proximity link. It was not runnable as
claimed. The submitted version adds an explicit, checksum-bound preparation
step in `scripts/prepare_feature_store_v1_cre_inputs.py` and
`config/quest_cre_phase3_v1.json`.

| Quantity | Rows |
|---|---:|
| Raw proximity links | 15,235,813 |
| Links admitted by the certified core gene universe | 11,289,221 |
| Links to genes outside that universe | 3,946,592 |
| Cross-partition links among assigned genes | 0 |
| Certified eligible genes | 54,960 |
| Intrinsic CRE catalog | 2,348,854 |
| Fixed intrinsic features | 10 |
| Expected values and QC rows, each | 549,600 |
| Expected dependency rows | 225,784,420 |

The output counts were subsequently confirmed in the retained build and full
validation reports. The universe includes chrY, following the certified core. Its partition
hash is `15899582172908ac1d1041bb25575cdc739e975bf5fbe1a955fc77ccd98b6dd4`.
This build uses the core's existing train/reward/test assignments.

The preparation step expands all ten wide intrinsic columns into explicit
`cre_id, feature_id, value, reason` measurements, retaining observed zeros. It
counts excluded links by biotype, rejects crossed/unassigned admitted links,
and preserves the fixed proximity rule with candidate and retained links
identical. Wider lncRNA fallback remains rejected by the input contract.

## Storage and validation

Heavy temporary databases, spill and schema-rewrite files go to
`/scratch/mes7956/pulearning/<run-id>`, following the reference project's scratch
pattern. The project allocation had about 35 GiB free at submission. Prepared
inputs and final bundles are versioned under the designated project directory.

### GPFS publication mode

The second release explicitly selects `PU_ATOMIC_DIRECTORY_MODE=symlink` on
Quest. A complete validated stage is moved into a newly reserved private
permanent sibling directory. One native `symlink(2)` call then creates the
public output path. That call refuses any occupied destination, including an
empty directory or dangling symlink; readers see either no output or the entire
completed artifact. Permission errors never trigger an automatic fallback.

The public path is a relative symlink to `.NAME.published-<unique>/data` in its
parent directory. **The backing directory is part of the retained artifact.**
Keep both when moving/backing up the parent, or dereference the public directory
when exporting it. An interrupted pre-commit process may leave an unpublished
backing directory for diagnosis; it is not automatically deleted as stale
staging. The original no-replace-rename mode remains the default outside the
explicit Quest setup.

Verification exercises the publication contract on the actual project
filesystem as well as the test fixtures. Added tests cover complete visibility,
late competing paths, failure restoration, and interruption after the pointer
has committed.

The verification job runs:

1. The full native project test suite, including scratch-routing and input-contract tests.
2. The 100,000-link / 10-feature acceptance run at 256 MB DuckDB and a 1,024 MiB RSS ceiling.
3. `scripts/check_evaluation.py` with its existing no-delete audit hook.

Only a successful verification job releases the genome-wide job. The publisher
performs full independent replay before publication. The batch job then records
a separate full validation and an experimental sampled validation (seed
20261004, 1% random genes plus chr22 and structural edge cases). Sampled replay
is supplementary; its broader edge-case and partition-hash work is unfinished.

Reports and timings: `experiments/<run-id>/` on Quest. Logs:

```text
logs/pu_cre_verify_8683150.out
logs/pu_cre_verify_8683150.err
logs/pu_cre_genomewide_8683151.out
logs/pu_cre_genomewide_8683151.err
```

Inspect the submission with:

```bash
ssh -o BatchMode=yes quest 'squeue -j 8683150,8683151'
ssh -o BatchMode=yes quest 'sacct -j 8683150,8683151 --format=JobID,State,Elapsed,ExitCode,MaxRSS'
```

For a future source release, stage and checksum that release first, then submit
once from its directory using a new run ID:

```bash
bash scripts/sh/submit_quest_cre_phase3.sh NEW_RUN_ID
```

## Interpretation of earlier results

The earlier 431-pass workstation run predates subsequent changes. Shim-assisted
runs do not certify real atomic publication. The new Quest run is native and
rejects an inherited `LD_PRELOAD`.

The retained Phase 2 baseline report is preliminary: it used logistic regression,
the label-control uncertainty calculation used a binomial prevalence interval
rather than an AP-null interval, and completion of the declared evaluation gates
has not been established. Those results do not establish a clean negative
control or authorize scientific claims from the CRE build.

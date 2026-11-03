# Claude/Opus CRE progress review, 2026-10-04

## Current disposition (2026-10-11)

The review below is historical. Subsequent Quest evidence certified the
standalone v4 CRE bundle by **separate-process full replay**, with sampled
replay retained as supplementary. All three schema-bound sorted fingerprints
and the original-certificate-bound sidecar are complete and accepted in §39.

Claude accepted the corrected evaluation harness in **§38.6** after the v7
same-ledger canary and twenty-run / 100-fold source-matched comparison were
independently verified. Standing reports cite v7; the original v2 report remains
rejected as written. The symbol-only DisGeNET support counts below were
provisional and do not establish official curated disease-only label eligibility.
Formal scoring still requires the official versioned release and rebuilt panels.
See `doc/evaluation-repair-20261011.md` and `doc/disgenet-gate-input-contract.md`.

## Submission

The miniature-bundle completion report was submitted exactly once to the
verified conversation
`https://claude.ai/chat/9216c204-4aba-4266-8b4f-c5cedb3913dd`
("PU learning pipeline assessment and RL novelty validation"). The composer was
empty beforehand, delivery was confirmed by the new user-message bubble
(message 15 of 16), and the completed response was read. It must not be
replayed.

The report covered: v2 bundle contents and manifest bindings, independent
source replay with mandatory staged validation, rehashed-corruption rejection,
the 100,000-link gate (100,000 values, 100,000 QC rows, 2,000,000 dependencies,
60.21 s, 702.03 MiB publisher RSS, 627.29 MiB validator RSS, 11,730,944 spill
bytes), 422 native tests plus 219 guarded evaluation tests, and the EXCEPT ALL
replay optimization (>10 min to 22.42 s standalone).

## Response (Opus 5.5 High, completed)

1. **Stop hardening; three lighter guards are enough.** The
   staged-validate-then-atomic-publish design already makes interrupted and
   disk-exhausted runs publish nothing. Replace the pending
   interruption/disk-exhaustion/competing-writer acceptance matrix with:
   - free-disk preflight (estimated temp + output ceiling with a 2x margin,
     abort if short);
   - an advisory lock file preventing a second writer;
   - one kill-mid-write test confirming nothing is published and the previous
     version stays intact.
   Then enable the 15,235,813-link build.
2. **Row-level dependency materialization may not scale.** Linear
   extrapolation (not a measurement) suggests about 305M dependency rows,
   roughly 1 h validation, 2.5 h build at ten features. For the genome-wide
   build: keep partition-level content hashes; replace exhaustive replay with
   exact recomputation on a seeded sample (1% of genes, plus all genes on one
   chromosome, plus edge cases: zero-link genes, maximum-link genes, lncRNAs
   with self-overlap); keep full replay as an optional mode. Report measured
   dependency row count and validator time either way.
3. **The questions that decide the project can run today**, without the CRE
   path: spec sections 1-2 counts (DisGeNET-to-GENCODE v50 mapping, diseases
   with at least 50 positives, lncRNA positives per disease), sections 4-5
   (freeze locus splits on the real universe, then assertions 9a, canaries 9b,
   permutation controls 9c), and section 6 arms 1-2 (confounder-only and
   nearest-positive baselines; `n_linked_CREs` is a simple interval-count join
   against SCREEN). Only arms 3-5 and the section 7-8 link-quality and
   lncRNA-stratum analyses need the genome-wide build.
4. **Decide the primary task before eligibility is frozen.** The novelty audit
   covered GO; the harness targets DisGeNET disease genes, where the closest RL
   work is RL-GenRisk (Nature Communications 2025, DQN for disease-risk genes,
   no PU learning). If disease genes are primary, a targeted novelty check
   against RL-GenRisk and disease-gene PU methods (NIAPU, KDGene) is warranted.

Ordered plan for the agent: (1) section 1-2 counts, report lncRNA numbers
first; (2) freeze splits on the real universe, then 9a-9c plus the two
baselines; (3) add the three guards and run the genome-wide proximity build
with sampled validation; (4) CRE arms 3-5 and the section 7-8 analyses. Report
with the section 12 template, marking uncompleted sections pending.

## Implementation status after the review

- The three guards are implemented in `src/features/cre_views_duckdb.py`
  (`_preflight_free_disk`, `_publication_lock`, `_remove_stale_staging`) with
  three new tests in `tests/test_cre_views_duckdb.py`. Syntax-verified; test
  execution was still blocked by the session sandbox at the time of writing
  and must be rerun before any publication.
- Section 1-2 counts are complete:
  `experiments/20261004_153245_disgenet_support_counts/counts.json`.
  Universe 54,710 genes (protein_coding/lncRNA, chr1-22+X). Of 9,828 diseases,
  2,109 are eligible at the frozen threshold of 50 universe positives.
  567,589 of 603,919 symbols mapped (36,330 unmapped symbols reported, never
  dropped). 735 distinct lncRNA genes appear as disease positives; the maximum
  per-disease lncRNA support is 75 (Breast Carcinoma). This answers the
  review's decisive question: the curated table provides far more than "tens"
  of lncRNA associations, so no alternative label source is required by
  support alone.

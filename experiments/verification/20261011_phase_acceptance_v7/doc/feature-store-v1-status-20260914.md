# feature_store_v1 implementation status, 2026-09-14

## Current disposition (2026-10-11)

The historical entries below describe the earlier milestones. The standalone
Quest v4 genome-wide CRE bundle now has separate-process full replay acceptance:
549,600 values/QC rows each and 225,784,420 dependencies, `status: valid` in
`experiments/20261005_quest_cert_v4_artifact/validate_full.json`. It remains
standalone rather than enabled in the certified core store. See
`doc/quest-cre-phase3-20261005.md` for terminal job states and publication details.
All three schema-bound sorted CRE fingerprints and the original-certificate-bound
sidecar are now complete and accepted in §39. The corrected same-ledger file
canary and fresh v7 twenty-run / 100-fold comparison were independently verified;
Claude accepted the evaluation harness as complete and certified in §38.6.
Standing reports cite the v7 snapshot and artifacts. The original v2 report
remains rejected as written; official scientific label/panel gates still apply.

The canonical evaluation split is now label-free and uses hash
`0dfef23274047a5547e63765208953e47323b582630c512441f74ab1fb98fe00`.
The provisional remapped panel hash is
`d5a8d8239948ece3e4793330f7de7c38d6a94eeee691c5321f80e4007a623443`.
These panels require replacement from an official curated disease-only
DisGeNET release before formal scoring. Exploratory real-label development fits
have occurred; the original no-disease-fit statement is historical, not current.

The random-score negative-control gate was retired. Earlier control results and
new-selector audit claims require the corrections in
`doc/evaluation-repair-20261011.md`. Fresh shuffled checks use a separate
preparation/fitting process, full-key label authorization, exact N=P sampling,
and a new checksum-bound Quest snapshot. The ≥25 frozen-test-positive gate and
certified-store requirements still apply to future scientific comparisons.

## Scope

The mechanical core of `feature_store_v1`, a one-way bridge for fixed
`gfpred-cre` 0.1.0 CRE features, and the first checksum-pinned empirical core
store are implemented and mechanically certified. This is not a biological
prediction result; no program or disease model has been fitted.

## Implemented

- Versioned Arrow schemas for genes, transcripts, promoters, CREs, partitions,
  sources, feature definitions, dependencies, coverage, and gene/CRE views.
- Deterministic GENCODE parsing with version retention, `_PAR_Y` handling,
  strand-aware transcript TSSs, exon counts, promoter/transcript relations, and
  strand-correct clipped promoter windows.
- Promoter aggregation with separate maximum, mean, observed-promoter count,
  effective-covered-base, and eligible-denominator features.
- The admitted proximity CRE adapter with candidate/retained dependency rows,
  exact retained-score checks, observed-weight coverage, no rE2G, and no wider
  fallback.
- Donor/study/context raw-count pseudobulk with full-library depth checks,
  explicit assayed-gene universe, cell count, detection fraction, and reporting
  of entirely QC-filtered groups.
- Atomic no-replace Parquet publication, physical and logical hashes,
  closed-world manifests, exact source-hash closure, enabled-arm enforcement,
  foreign-key checks, and explicit discovery/validation loaders.
- First-class CRE and partition/locus tables plus conservative candidate-window
  overlap certification across declared nonempty partitions.
- CLI orchestration for native promoter, CRE, and expression adapters and the
  versioned CRE bridge.
- Closed-world admission of nine real inputs under
  `datasets/raw/feature_store_v1/`, with provider MD5 and locally computed
  SHA-256 pins in `scripts/feature_store_v1_resources.tsv` and verified inventory
  `datasets/processed/feature_store_v1/source_inventory.json`.
- Bounded-memory GTEx v10 GCT ingestion. It uses the complete count universe for
  library depth, exact count-header/sample-metadata alignment, technical-aliquot
  summation within donor/tissue, fixed log1p-CPM tissue medians, donor detection
  fractions, and stable Ensembl ID reconciliation without symbol joins.
- Strand-paired ENCODE K562 CAGE extraction over all GENCODE v50 promoters.
  Sparse BigWig gaps are declared measured zero only for existing chromosome
  windows; promoter values are signal integrals divided by the full clipped
  promoter length.
- Label-independent 70/15/15 hash assignment of complete overlap components
  formed from gene spans, +/-100 kb-expanded cCRE windows, and consumed promoter
  windows. The corrected assignment has 158 components and split hash
  `15899582172908ac1d1041bb25575cdc739e975bf5fbe1a955fc77ccd98b6dd4`.
  Whole-component assignment yields 1,749,272 train, 300,771 reward and 353,771
  test gene/CRE entities; no component was split to force exact proportions.
- A certified corrected real core store at
  `datasets/processed/feature_store_v1/store_v1_real_core_promoter_v3_20260914`. It contains
  the full registry and SCREEN geometry plus 146 fixed GTEx/CAGE features,
  8,261,520 view rows and 8,681,942 dependency rows. Its manifest SHA-256 is
  `9ab7eebd2cb7f8cf5241ae55d3698214aa532cf7f9e897ee8dd1ed22fff1f837`.
- The submitted original store is retained unchanged, but an expanded audit
  found cross-partition consumed-promoter overlaps and invalidated it for fitted
  work. The corrected certificate reports zero gene/cCRE and promoter dependency
  overlap witnesses. Full disposition is in
  `doc/astra-feature-store-review-20260914.md`.
- A genome-wide fixed +/-100 kb proximity artifact with 15,235,813 links and
  99.7361% lncRNA coverage. No wider fallback was used; the link artifact is
  prepared but not admitted as a store view pending out-of-core native
  aggregation/dependency publication.

## gfPred CRE add-on

The original archive is retained unchanged at
`doc/gfpred_cre_extension_v0_1_0.zip`: 49,929 bytes, SHA-256
`c78ba8d343c93ba4383a76f0e0f6342e316a886326a9bf2fe761770b12453935`.
It contains 30 files and passes ZIP integrity checks. The extracted package is
under `external/gfpred_cre_extension_v0_1_0/gfpred_cre_extension`.

The canonical package manifest SHA-256 is
`9e12478bb7f2a48f6b88dfacab3c38249bcbb6533c28ec43903bd6dc70eee2ec`;
all 29 listed files match. Its implementation hash is
`c533e91055872be42710193f95ee48a08dd3b1c060cedeac1eb5bd86bd15423c`.

The add-on remains namespaced and is not merged into `src.features`. The native
bridge accepts only sealed `gfpred_cre_v1/features` artifacts, namespaces every
context-specific feature, preserves the original artifact and semantic hashes,
and rejects weak targets, models, predictions, and annotations from fixed
discovery views. The add-on's interval-edge proximity method remains distinct
from the native midpoint-distance method.

## Verification

- Focused feature-store suite: 34 passed in 0.74 seconds.
- Native project suite: 407 passed in 57.58 seconds.
- Existing guarded suite: 219 passed in 8.68 seconds.
- Immutable `gfpred-cre` suite: 37 passed with 8 subtests in 0.48 seconds.
- Real chr22 GENCODE parser check: 1,747 genes, 15,002 transcripts, 8,664
  unique promoters, and 15,002 promoter/transcript relations.
- GTEx v10 preparation: 19,788 samples, 946 donors, 68 detailed contexts,
  35,217 stable-ID-mapped eligible genes and 2,394,756 gene/context rows. Another
  19,743 eligible v50 genes are absent from v39 and remain explicitly unmapped.
- K562 CAGE preparation: 388,601 promoters and 777,202 replicate measurements;
  400,725 are nonzero and 376,477 are observed zeros, with no unavailable
  primary-chromosome windows.
- Read-only corrected-store certification: `status: valid`; 78,696 genes,
  644,255 transcripts, 388,601 promoters, 2,348,854 cCREs, 364,956 consumed
  eligible-gene promoters, and zero candidate-window or promoter-dependency
  overlap witnesses. The machine-readable certificate is
  `experiments/20260914_feature_store_v1_promoter_v3_validation.json`, SHA-256
  `e0bff03a966cbc7975956cf8e516e3c18b2f82c9db205cf64c68748063254ceb`.

## Open gates

- `config/feature_store_v1_promoter_v3.yaml` resolves the corrected certified
  core. Replaying must use a new output path because atomic publication correctly
  rejects replacement. The original config/store remain historical evidence.
- A real `gfpred-cre` fixed-feature artifact must use the same genome build as
  the native registry. Toy artifacts are not admitted to a GRCh38 store.
- Scalable bounded BigWig extraction is implemented for the two-replicate K562
  CAGE pilot. Multi-gigabyte ATAC/ChIP downloads remain deferred until their
  exact bounded feature scope is admitted.
- The standalone out-of-core CRE bundle now publishes path-backed values,
  mandatory aggregation QC, and candidate/retained dependencies. Its v2 manifest
  binds all six source Parquets and records physical plus streaming logical output
  hashes. The independent bounded validator replays source geometry, partitions,
  dependencies, and aggregation arithmetic, and must accept staging before the
  atomic rename. Thirty-one focused tests pass with one opt-in scale test skipped.
- Historical verification: the user reported 431 passed / 1 skipped (81.27 s)
  on the workstation before later edits. The 219-pass evaluation and 108.98 s
  scale runs used shims and do not establish current native acceptance.
- Quest release `20261005_quest_cre_phase3_v2` passes the project-filesystem
  publication probe and 458 tests / 1 skipped (56.49 s). Verification 8677345
  completed: scale passed in 179.367 s at 610.070 / 570.262 MiB RSS, and guarded
  evaluation passed 219 tests (39.01 s). Genome-wide job 8677346 prepared real
  inputs but hit the 8 GB limit at all-feature ordered aggregation; feature-only
  batching in v3 also exceeded the cap. The v4 release bounds both publisher
  aggregation and independent QC replay to 512 complete gene neighborhoods per
  feature batch. Native verification **8683150** passed **459 tests / 1 skip**,
  scale (177.849 s, 515.371 / 468.281 MiB RSS), and **219 guarded tests**.
   Genome-wide **8683151** later failed in a supplementary step after valid full
   replay; retained reports now certify the published standalone bundle.
  See `doc/quest-cre-phase3-20261005.md` for current submission details.
- The publisher's canonical schema pass now writes to a scratch path outside the
  staged tree and stream-copies into place, so no per-file rename is required;
  atomic publication uses the complete-directory commit. On Quest GPFS this is
  an explicit exclusive symlink to a permanent complete backing directory,
  since the first native run found `RENAME_NOREPLACE` unsupported (`EINVAL`).
- The 2026-10-04 Claude review replaced the planned interruption/disk-exhaustion
  test matrix with three lighter guards (free-disk preflight, advisory writer
  lock, stale staging cleanup, one kill-mid-write test). They are implemented
  with tests; execution was still sandbox-blocked at review time. It also ruled
  proposed sampled seeded recomputation for genome-wide builds. The experimental
  mode is implemented (`replay_mode="sampled"`) and
  measured: it keeps every whole-table invariant including content hashes, and
  restricts only the row-level replays to a seeded sample (1 percent of genes,
  one chromosome, plus zero-retained, maximum-retained, and candidate-only edge
  cases). A tampered gene inside the sample is rejected by both modes; one
  outside the sample is rejected only by full replay. Whole-table hashes cannot
  compensate for omitted semantic replay if the manifest is also rehashed.
  Full replay remains the certification default for the submitted Quest build.
- DisGeNET support is provisionally counted: 2,109 of 9,828 diseases have at
  least 50 positives on the 54,710-gene universe, and 735 distinct lncRNAs
  appear as disease positives (`experiments/20261004_153245_disgenet_support_counts/counts.json`).
  These counts do not establish per-task retained-test or independent-locus support.
- The Phase 5-E locus split is frozen (spec section 4, config
  `config/locus_eval_v1.yaml`): split hash
  `efac17ddea0216b2720e00400a243e181f72ed5d521bcab8377b7fc2c1912deb` after the
  section 13.1 purge correction to 200,350 bp (the 200 kb split is retained as
  history); 2 Mb blocks confirmed (purge removes only 4.4-5.2 percent of
  training per fold); all link-table-free 9a assertions pass.
- The Phase 5-E reward interface is implemented
  (`src/evaluation/reward_interface.py`): pools are derived from the frozen
  split, the public surface is a scalar `reward(rn_gene_ids, task_id,
  outer_fold)`, all membership checks precede any fit, and all three 9b canaries
  pass (test-fold genes, reward-validation genes, and oracle-style selections
  all raise). Nine interface tests pass.
- Claude's section 13 response is executed where no new data is required:
  lncRNA label concentration measured (top 10 lncRNAs = 24.3 percent of pairs;
  neoplasm 48.2 percent; the honest intergenic >200 kb stratum holds 80
  distinct positives), and disease panels frozen (101 Jaccard clusters;
  40-disease development panel, 10-disease final panel from untouched
  clusters; panel hash `a283053b89417fda2e75b5ba01bdf96ddd42ca01ebc1449fcd852b237057576d`).
  Section 13.2 NCBI Gene ID remap awaits a permitted reference download.
- Real proximity remains disabled pending guard-test execution, the
  sampled-validation mode, and measured row/validator counts for the possible
  304,716,260-row dependency bundle.
- The gene-program dictionary, cNMF/PCA/joint masked model comparison, CRE ridge
  comparator evaluation, disease fitting, and biological validation remain
  downstream work.
- The retained minimum of 25 test positives is unchanged. No seed, chromosome,
  disease definition, purge margin, or retained v1-v3 outcome may be searched
  to rescue support.

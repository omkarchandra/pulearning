# Human GO label preparation v3 — native replay complete

Omkar confirmed **gene function prediction** as the primary task. The input
contract and source-date interpretation were reviewed in §41.1/§41.3 of the
[PU review thread](https://claude.ai/chat/9216c204-4aba-4266-8b4f-c5cedb3913dd).
The native counts and blocked panel proposal were delivered for review after
the separate-code verifier completed.

## Retained sources and rules

- Official GO bundle: **2026-08-05**; internal ontology version:
  **2026-07-26**; protein and RNA GAFs generated **2026-07-28**.
- Matching human RNA GAF/GPI from EBI GOA are retained as content-bound
  snapshots of **mutable upstream URLs**, with headers/HTTP provenance.
- RNAcentral mapping release **27.0**; retained Approved-HGNC snapshot last
  modified **2026-10-02**, acquired 2026-10-05. This is cross-sectional,
  not a temporal/CAFA benchmark.
- Eleven experimental evidence codes; TAS/IC retained separately. Exact ID
  mapping, global one-to-many rejection, no symbol or isoform fallback,
  direction-specific biotype admission, alternate GO IDs, same-aspect
  `is_a`/`part_of` propagation, and root/NOT exclusions are frozen in
  `PROTOCOL.json`.
- The original 54,710-gene universe and label-free locus split are unchanged.
  Panel admission requires ≥50 positives, a ≤20% aspect-size cap, ≥25 pooled
  test positives, and ≥5 in each fold. **Every separately reported AP still
  needs ≥25 positives in its actual scoring domain.**

The archived `HUMAN-uniprot.gaf.gz` contains mixed databases: 862,694
UniProtKB, 38,224 RNAcentral and 9,558 ComplexPortal rows. Only UniProtKB is
admitted from that file; the other rows are audited. RNAcentral labels come
from the separately bound RNA GAF's 47,173 rows. Complexes are not genes.
See `source_database_inventory.json`.

## Verified label and support counts

Full replay checked **957,649 GAF rows**, **42,695 product IDs**, and
**16,996 term-support records**. Direct experimental labels have **133,507**
gene–term pairs; propagated labels have **760,145**. TAS/IC has **34,842**
separate direct gene–term pairs and is not primary positive evidence.

| Aspect | Experimentally annotated genes | lncRNA genes | Isolated-stratum lncRNA genes | Panel-supported terms | Terms with ≥25 positives in every test fold |
|---|---:|---:|---:|---:|---:|
| BP | 10,643 | 25 | 7 | 1,135 | 409 |
| MF | 15,411 | 22 | 5 | 304 | 129 |
| CC | 12,990 | 11 | 4 | 227 | 102 |

The full §40 intergenic/>50 kb lncRNA stratum has 12,038 genes; **11,913**
survive the exact evaluation-universe join. There are **27** distinct
experimentally annotated lncRNAs across aspects. Among candidate terms, the
largest lncRNA-positive count is **16**, and the isolated-stratum maximum is
**4**. Neither supports a per-term lncRNA AP under the retained 25-positive
reporting gate.

**Panel admission is blocked.** Jaccard=0.10 connected components yield only
**28** clusters. The original proposal has 28 development terms, zero final
terms, and explicitly reports 12/10 unfilled slots. The proposed 40/10 panel
cannot be frozen. Its rule and output remain retained; a prospective panel
design revision is being reviewed. Models fitted: **0**.

## Native and local verification

Quest job **9479662**: `COMPLETED`, `0:0`, **2m13s**, batch peak RSS
**793,756 KiB**. All **6** meaningful filter/ontology tests passed. Separate
processes ran the builder, independent raw-source verifier, production
source-rederivation gate, and malicious-input canaries. Both source seals
passed. The verifier imports no production GO adapter; its independence is
code/process independence, not a separate model reviewer.

The native canaries reject (1) an altered evidence protocol and (2) fabricated
positives even after their output-manifest hashes are recomputed. Local
symlink-based fixture setup was prohibited; `check_gate_mutations_stream.py`
retained create-only copies and passed the same attacks on the identical
frozen adapter. The original failed setup is retained. All **8** native/local
payload artifacts are byte-identical.

Authoritative Quest locations:

- `datasets/raw/go/2026-08-05/acquisition_20261011_v3/`
- `releases/20261011_go_primary_v3/`
- `experiments/go_labels/20261011_go_primary_v3/runs/9479662/`

Local evidence: `native_9479662/`, `local_run/`, `native_import_receipt.json`,
`input_import_receipt.json`, `snapshot_publication_receipt.json`, and
`prior_failure_manifests/`. The native export retains 24 regular payloads;
24 fixture symlinks are retained as path/target metadata, with original
sources already preserved separately.

| Binding | SHA-256 |
|---|---|
| Protocol | `25f52c579aac9855e971b2f5ae1cd6831558b27826a7de6915025341a7d31c2d` |
| Source manifest | `8d4e647b5ca14b7b6799a28dc0c5997c85ce1238ead414294c4d0fed94928729` |
| Acquisition manifest | `e8e7649063c78d50facc07952474900e990a2d222e35ad96934b4a492519ebba` |
| Native label manifest | `24b83fb3746e143dd277629997487b2c5eaafd8ddb0167738af628f34361aa3d` |
| Native independent replay | `6dbd8066ea45c4e73b6941629f8e2e0dc19df50edf4882e2ef29af13a53e26d2` |
| Native label gate | `8a8e28fe0c650509a3a21e678661352957003e5eb3cb8b5e15b526027128e9ae` |
| Native canaries | `8e4516dbbc490ff37564a50796ed530dc8ad04f305d2e074d5978e6727b72662` |
| Native export archive | `3ba33d5c71532589b865f4850eb15ceaf9e16876b699eb85404a5c44f2d7a9cb` |

The initial acquisition failed on an erroneous bundle-date/ontology-date
equality check. The second downloaded matching RNA inputs but assumed an HGNC
Quest path that was absent. Both failed manifests and the original bytes are
retained. A blocked local rsync rename left intact cache files; exclusive
stream copies verified all 18 imported input payloads without deleting the
cache evidence.

## Next admission requirements

Complete a prospectively frozen, supported GO panel design, then bind a GO
sealed-worker configuration and certified features. Future results require
pipeline-matched pooled shuffled references, propagation-aware aspect Fmax,
within-biotype/lncRNA-stratum support gates, and the own-TSS-cCRE exclusion
feature variant. The provenance gate verifies GO labels; it does not grant
real-scoring authorization.

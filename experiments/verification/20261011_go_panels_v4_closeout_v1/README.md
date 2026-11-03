# GO panels v4 — complete source-derived proposals

The §42.2/§42.4 panel protocol was frozen before revised counts. The new
implementation uses **UPGMA average linkage within each GO aspect**, at
Jaccard similarity ≥0.30, with explicit node-ID tie resolution and no epsilon.
Development representatives are size-tertile stratified; final representatives
come from unused clusters and exclude development ancestors/descendants.
Development nesting is allowed and reported. Full details are in
`PROTOCOL.json`, `REVIEW_BASIS.md` and `REVIEW_CLARIFICATIONS.md`.

The original v3 source/label payloads and incomplete panel remain retained.
V3's single-linkage=0.10 proposal was **not** the original §13.3 rule.

## Verified panel proposals

| Eligibility | Aspect | Eligible terms | Clusters | Development | Final | Hierarchy skips | Development nesting pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| ≥5 each fold, ≥25 pooled | BP | 1,135 | 368 | 27 | 6 | 2 | 5 |
| Same | MF | 304 | 131 | 7 | 2 | 1 | 0 |
| Same | CC | 227 | 106 | 6 | 2 | 0 | 0 |
| ≥25 each fold | BP | 409 | 135 | 27 | 6 | 5 | 4 |
| Same | MF | 129 | 45 | 7 | 2 | 0 | 0 |
| Same | CC | 102 | 48 | 6 | 2 | 1 | 0 |

**Both proposals are complete: 40 development and 10 final terms.** Both stop
at the first prespecified step, `average_j030`; the complete=0.30 and
average=0.50 fallbacks were not needed and were not executed. No cluster
definition is mixed across steps or eligibility variants.

The strict proposal's 50 selected terms have at least **25 protein-coding
positives in every test fold**, as well as ≥25 all-gene positives. The pooled
proposal has 92 of its 250 individual fold-scoring domains below 25, with a
minimum of 6; that permits pooled AP only, under its admission rule. The
strict proposal satisfies Omkar's standing per-fold reporting gate.
`reported_domain_support.json` retains every selected term's actual domain
counts. Neither proposal has a term with ≥25 pooled lncRNA or isolated-stratum
lncRNA positives. Omkar instructed us to retain lncRNA prediction and ask
Claude to read his gene-function paper; see `USER_LNCRNA_DIRECTION.md`.

## Native verification and provenance gate

Quest job **9481056** completed `0:0` in **1m56s**, batch peak RSS
**792,560 KiB**. All **8** meaningful linkage/selection fixtures passed:
UPGMA versus single/WPGMA, complete linkage, exact cut boundary, deterministic
ties, size-stratum shortfall, hierarchy exclusion/development nesting,
fresh ordered fallback and separate eligibility modes.

The separate-code/process verifier first repeated the full v3 raw-source
replay (957,649 GAF rows, 42,695 product IDs, 16,996 term-support records), then
recomputed **812,095 pair distances** and **1,473 selected merges**, all cut
memberships, tertiles, quotas, skips, nesting and selected-term support.
The producer uses bitsets and a minimum heap; the verifier uses set
intersections and a global matrix scan. It imports no production label or
panel adapter. This is code/process independence, not a separate model review.

The production gate separately rederives the complete proposal. Both
canaries passed: an altered clustering protocol was rejected by binding, and
substituting another eligible final-term representative was rejected even
after recomputing the proposal content and file-manifest hashes. Originals
remained unchanged. Pre/post frozen-source seals passed.

Native and local `panel_proposal.json` are **byte-identical**; source-binding
manifest fields also match. Sixteen native payloads were retained and fully
SHA-256 verified locally in `native_9481056/`.

Authoritative Quest paths:

- Inputs: `datasets/raw/go/2026-08-05/acquisition_20261011_v3/`
- Base labels: `experiments/go_labels/20261011_go_primary_v3/runs/9479662/results/`
- Frozen source: `releases/20261011_go_panels_v4/`
- Native evidence: `experiments/go_labels/20261011_go_panels_v4/runs/9481056/`

| Binding | SHA-256 |
|---|---|
| Protocol | `3a2c4c1d1316aee07436e006cd42538e880279ed6bfbb320aef8b73c88b90a1c` |
| 13-file source manifest | `9352c86c4aafb2dab6f3b440ea5e4564ab47bad52d6c61827a03d2a0c803f9ac` |
| Panel proposal | `0d305dd8c4a274f0c1583e798dd5037e237fbc7fd41d10ad6586c2beb2bef7df` |
| Panel content | `0cda37bef9fc033fedffa13c66b8d4c4425be4d8f1a020028b995bad6dd5f076` |
| Native panel manifest | `c456e5e7dd326d1cf42948c2db415fc654a9c44ac1f88b7a720e8bbd7896107c` |
| Native independent replay | `65829e863c4bc792bcd128c431f2ae2e80ef973a64b9047a3d05691d280f1ad3` |
| Native panel gate | `e323359bb6a4a5366922408fb6fd6e5b35b6427409cf45f0ce5807b9216ccc5e` |
| Native canaries | `c212c16b52501f9ed6a5ef13d4513efb95c412f42160f4fe3747bd4aa1bd53c2` |
| Native evidence archive | `e5a28a36eb5ea9662d8ab57402ff7f74645d8919f93d50d12ec9d8a3c11a600a` |

**Models fitted: 0.** Complete proposals remove the v3 slot shortfall; actual
scoring still requires a selected supported panel, certified features and a
GO sealed-worker configuration. The paper-specific lncRNA prediction path
is being recovered with Claude at Omkar's request.

# Verified GENCODE v50 lncRNA locus strata

## Frozen rules (written before counting)

Precedence is top to bottom; a lncRNA takes the first class that applies.

1. **Divergent:** its TSS lies within 1,000 bp of a protein-coding gene's TSS
   on the opposite strand, and the two genes do not overlap in genomic span.
2. **Antisense:** its genomic span overlaps a protein-coding gene's span, and
   the two are on opposite strands.
3. **Sense-overlapping:** its span overlaps a protein-coding gene's span on the
   same strand.
4. **Intergenic:** none of the above.

Gene span is the GENCODE gene-level start-end interval. TSS is from the pinned
gene-level TSS table. Distances to the nearest protein-coding TSS are unsigned,
same-chromosome distances for every lncRNA and use bins `<5`, `5-50`, `50-200`,
and `>200` kb. The self-overlap flag asks whether the lncRNA's own TSS +/-1 kb
intersects any cCRE admitted to its own published v4 feature vector. Exact
coordinate boundaries and the deterministic nearest-TSS tie-break were frozen
in `RULES.md` before job submission.

The full pinned GENCODE v50 set was used: 34,866 `lncRNA` genes and 20,107
`protein_coding` genes.

## Counts

| Locus class | Genes |
|:--|--:|
| Divergent | 1,603 |
| Antisense | 11,829 |
| Sense-overlapping | 2,447 |
| Intergenic | 18,987 |
| **Total** | **34,866** |

| Nearest protein-coding TSS distance | Genes |
|:--|--:|
| <5 kb | 7,238 |
| 5-50 kb | 11,361 |
| 50-200 kb | 9,426 |
| >200 kb | 6,841 |
| **Total** | **34,866** |

| Locus class | <5 kb | 5-50 kb | 50-200 kb | >200 kb | Total |
|:--|--:|--:|--:|--:|--:|
| Divergent | 1,603 | 0 | 0 | 0 | 1,603 |
| Antisense | 4,077 | 4,393 | 2,484 | 875 | 11,829 |
| Sense-overlapping | 509 | 1,068 | 634 | 236 | 2,447 |
| Intergenic | 1,049 | 5,900 | 6,308 | 5,730 | 18,987 |

Own-TSS self-overlap was true for 32,244 lncRNAs and false for 2,622.
Thus 92.48% of lncRNAs have an own-vector cCRE overlapping their own TSS
window; this flag is not a useful minority stratum. The prespecified honest
lncRNA cell, intergenic with nearest protein-coding TSS strictly beyond 50 kb,
contains **12,038 genes** (6,308 at 50-200 kb plus 5,730 beyond 200 kb), or
63.40% of intergenic lncRNAs and 34.53% of all GENCODE v50 lncRNAs.

| Locus class | Self-overlap false | Self-overlap true |
|:--|--:|--:|
| Divergent | 30 | 1,573 |
| Antisense | 516 | 11,313 |
| Sense-overlapping | 202 | 2,245 |
| Intergenic | 1,874 | 17,113 |

## Verification and provenance

- Build job `9471069`: `COMPLETED`, exit `0:0`; 34,866 rows.
- Authoritative verifier job `9471232`: `COMPLETED`, exit `0:0`.
- Independent checks passed for pinned inputs, exact schema and rows, full gene
  identity/span replay, full precedence-aware class replay, full nearest-TSS
  replay, full own-vector self-overlap replay, and all reported counts.
- GTF/TSS eligible rows: 54,973 joined of 54,973, with zero metadata
  mismatches and zero duplicate eligible GTF gene IDs.
- v4 retained links: 11,289,221 rows, zero duplicate pairs, zero wrong-method
  rows, and zero fallback rows.
- Report SHA-256:
  `9ff07013d1208d6c572ebf3b2411fa750d31a1ab9f3a312c7e4d054e8d7817a3`
- Table SHA-256:
  `53bbf82ed00fcb95d5de2bd57bf364c03040e570f09f41db5d0b022ddda464e1`
- Rules SHA-256:
  `2481fd7fbe36ef2d70f91bb97a30b1ad01ff577b51b45bcbb77df54f934dc98e`
- Verifier SHA-256:
  `5173b0f98334499a88867f9b38601e05639b5a8060e9ce5aab179e47755e497e`
- Verification receipt SHA-256:
  `a914593b83cfc488f04004783ad0128441b5114e756665eefd72607f964bb6cf`

Verifier job `9471074` was cancelled before producing a receipt after its
in-progress implementation was found to count any own link rather than an
interval-matched cCRE in one independent self-overlap assertion. That verifier
was corrected before use; create-only job `9471232` is authoritative.

This is a label-free descriptive table. It makes no causal-regulation claim
and does not run selectors, labels, or benchmarks.

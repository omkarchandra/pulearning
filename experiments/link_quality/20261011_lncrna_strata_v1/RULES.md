# Frozen lncRNA locus-strata rules

**Frozen:** 2026-10-11, before counting.

Inputs are limited to the pinned GENCODE v50 gene annotation, the pinned
GENCODE v50 gene-level TSS table, the published v4 retained-link table, and the
published v4 cCRE catalog. The classification universe is the full pinned
GENCODE v50 set: 34,866 `lncRNA` genes and 20,107 `protein_coding` genes.

## Locus class

Precedence is top to bottom; a lncRNA takes the first class that applies.

1. **Divergent:** its TSS lies within 1,000 bp of a protein-coding gene's TSS
   on the opposite strand, and the two genes do not overlap in genomic span.
2. **Antisense:** its genomic span overlaps a protein-coding gene's span, and
   the two are on opposite strands.
3. **Sense-overlapping:** its span overlaps a protein-coding gene's span on the
   same strand.
4. **Intergenic:** none of the above.

Gene span is the GENCODE gene-level start-end interval. TSS is from the pinned
gene-level TSS table. GENCODE's 1-based closed gene intervals are represented
as 0-based half-open intervals `[start - 1, end)`; two spans overlap exactly
when `lnc_start < pc_end AND pc_start < lnc_end`. "Within 1,000 bp" means an
unsigned TSS difference less than or equal to 1,000 bp. All comparisons are
restricted to the same chromosome.

## Nearest protein-coding TSS

Unsigned distance to the nearest protein-coding TSS is computed for every
lncRNA regardless of locus class, on the same chromosome. Equal-distance ties
are resolved by lexicographically smallest unversioned protein-coding gene ID.
Distance bins are:

- `<5 kb`: distance `< 5,000` bp
- `5-50 kb`: `5,000 <= distance <= 50,000` bp
- `50-200 kb`: `50,000 < distance <= 200,000` bp
- `>200 kb`: distance `> 200,000` bp

## Own-TSS self-overlap

The self-overlap flag is true when at least one cCRE admitted to that lncRNA's
own published v4 feature vector intersects its own TSS +/- 1,000 bp. The v4
retained-link table defines admitted cCREs. A cCRE's 0-based half-open interval
`[start, end)` intersects the closed base-pair window from `TSS - 1,000`
through `TSS + 1,000` exactly when `end > TSS - 1,000` and
`start <= TSS + 1,000`, on the same chromosome. The wider lncRNA fallback is
not admitted by the v4 bundle contract.

This is a label-free descriptive table. No selector, disease label, benchmark,
or outcome enters any rule or count.

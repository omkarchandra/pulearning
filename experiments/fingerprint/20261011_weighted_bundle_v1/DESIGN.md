# Frozen weighted-bundle design

**Frozen:** 2026-10-11, before building.

Build a second gene-feature bundle from the exact cCRE-gene pairs admitted by
the published v4 bundle. The published v4 bundle and all source artifacts are
read-only.

For each link with signed cCRE-midpoint-minus-TSS distance `d` in base pairs,
replace its pooling score by

`score_weighted = score_v4 * exp(-abs(d) / 68152.45706)`.

The fixed scale is the preregistered 68.15245706 kb value. It is not refit.
The candidate and retained link tables remain pair-identical, and all other
link columns, cCRE measurements, genes, partitions, missingness rules,
publisher settings, and validation contracts remain fixed to v4.

The production bundle must use the existing out-of-core publisher, pass its
prepublication full replay and a separately recorded full replay, and receive
schema-bound sorted-content fingerprints for view, aggregation QC, and
dependencies. A finite fixture must also establish that passing lambda as
positive infinity leaves link scores and all published logical values exactly
equal to an unweighted control.

This is a label-free feature-construction ablation. It does not change link
admission, claim causal regulation, replace the unweighted default, or run any
selector, label, or benchmark.

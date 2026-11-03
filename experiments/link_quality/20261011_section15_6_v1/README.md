# Section 15.6 link-quality follow-up

This run computes rE2G-supported proximity-link precision by distance bin and
SCREEN v4 cCRE class at biosample-support thresholds `k = 1, 5, 20`.

The script resolves the retained-link and catalog artifacts from the published
v4 bundle manifest, verifies their SHA-256 hashes against that manifest, and
leaves the bundle and all inputs untouched. It reports:

- per-gene median/IQR precision (the macro definition used in Section 15),
- pooled supported/total precision with explicit numerators and denominators,
- distance-bin, cCRE-class, and distance-by-class tables,
- the unstratified Section 15 precision/recall values as a reproduction check,
- input schemas, checksums, DuckDB settings, runtime, and peak RSS.

The prior Section 15 script bound zero-based rE2G `column8` (the score) as the
biosample. This run corrects the binding to `column7` and labels the old
headline values as diagnostic references rather than reproduction targets.
It also uses `abs(distance_to_tss)`: the bundle field is signed midpoint-minus-
TSS distance, while the prior distance-weighted calculation exponentiated the
signed value. The prior weighted-precision result is therefore invalid.
The verifier pins the compressed rE2G input at SHA-256
`074820992b59f46273e650aabe6d8c96156e43b12ee5213f56a56f5815298d12`.

`rE2G` support is an operational comparator, not biological ground truth.

`verify_section15_6.py` independently checks the report's arithmetic and that
the distance, class, and distance-by-class rows each partition the overall
numerators and denominators at every threshold.

`revalidate_section15_context.py` is a separate canonical-contract pass for
the contextual values using their original definitions: protein-coding rE2G
pair distance bins, GENCODE lncRNA non-fallback proximity-link coverage, and
GENCODE lncRNA/protein-coding rE2G coverage. The certified §7 result reports
34,774/34,866 lncRNAs with primary bundle proximity links; the separate
documented 1,434/34,866 statistic is sparse rE2G source coverage. Neither
presence/absence coverage count depends on the defective biosample column.
`verify_section15_context.py` pins both the rE2G gzip and GENCODE TSS Parquet,
then checks pair integrity, bin partitioning, and coverage arithmetic.

`revalidate_section15_unmatched.py` and its verifier provide the requested
distinct rE2G target IDs missing from GENCODE v50, partitioned by the source
biotype in canonical `column3`.

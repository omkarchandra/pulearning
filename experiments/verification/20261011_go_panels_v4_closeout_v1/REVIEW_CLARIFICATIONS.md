# Reviewer clarifications — §42.4, before counts

The reviewer accepted and froze the delivered v4 details in §42.4:

> “Average linkage” means UPGMA: the arithmetic mean over all cross-cluster
> pairs, not WPGMA's cluster-weighted mean.

The member-count-weighted recurrence in `PROTOCOL.json` implements UPGMA.
“Member” here means GO terms, not the cardinality of their gene union.

> Ancestor/descendant exclusion is checked against development representatives.
> Nesting among development representatives themselves is allowed and reported,
> not excluded.

Each proposal reports the ancestor/descendant pairs within its development
panel. Every final representative is checked against all development
representatives in its aspect. All other delivered tie, cut, tertile, quota
and fresh ordered-fallback details were accepted as written.

The scope correction on lncRNA counts was accepted. Omkar's lncRNA decision
remains open. Revised counts must pass the verifier before reviewer delivery.

# GO panel rule revision — §42.2

Read in the exact [PU review thread](https://claude.ai/chat/9216c204-4aba-4266-8b4f-c5cedb3913dd?artifact=72c78c03-275e-4714-b76c-19d7869ab9f9)
before revised clustering. The reviewer accepted the v3 source/label evidence
as §42 and specified:

1. Cluster within aspect using average linkage at Jaccard similarity 0.30.
2. Development: BP 27 / MF 7 / CC 6, size-tertile stratified. Final: BP 6 /
   MF 2 / CC 2 from unused clusters, excluding development ancestors/descendants.
3. Ordered fallback: complete linkage at 0.30, then average at 0.50.
4. Report a strict ≥25-in-each-fold proposal under the same rules.
5. Send counts only after separate-code/process verification.

The earlier v3 single-linkage=0.10 proposal **was not §13.3's original rule**.
It stays retained as an incomplete proposal. No label/source bytes are revised.

The deterministic merge, representative, tertile, rank and fallback details
are frozen in `PROTOCOL.json` and were delivered before revised counts. The
27 supported lncRNAs and maximum 16 candidate-term positives refer to the
exactly mapped frozen universe under the declared filters; they are not
counts of every human lncRNA function annotation in every database.

Omkar's §42.3 lncRNA reporting decision remains open. Preparation can proceed
independently of it. Scoring admission requires a selected supported panel,
certified features and the accepted sealed-worker protocol.

# DisGeNET gate input contract

The genuine official versioned release is still required before formal real
scoring. This contract describes the checked adapter; it does not certify the
provisional symbol-only `DisGeNET.txt` as a compatible release.

## Declared bulk columns

The default tab-separated release adapter uses the following exact headers:

| Logical field | Default header | Checked behavior |
|---|---|---|
| Gene identifier | `geneId` | Entrez Gene ID joined to the pinned gene mapping |
| Disease identifier | `diseaseId` | Stable task key, not a disease-name/symbol join |
| Disease type | `diseaseType` | Only rows equal to `disease`, case-insensitively |
| Evidence source | `source` | At least one source in the explicit curated-source allowlist |

A release-specific `column_map` may rename those four fields. All four must
exist; HTML responses, unsupported schemas, and metadata-free symbol lists
fail closed. Multiple sources may be separated by `|` or `;`.

## Source allowlist

`curated_sources` is a required, nonempty list of the release's exact source
identifiers. It is declared and pinned in the label manifest; sources are never
automatically inferred from their names or silently broadened. `BEFREE` and
`LHGDN` are explicitly rejected as text-mined evidence. The gated toy fixture
uses `CTD_human`; this is a test fixture, not a declaration that the future
official release has that spelling or only that source.

No official-release allowlist has been frozen yet because the authenticated
release and its source metadata have not been supplied. Checking its actual
columns and curated-source IDs is the first import step.

## Official acquisition check (2026-10-11)

The official [release history](https://www.disgenet.com/docs?version=latest)
lists **v26.3, September 22, 2026**. This is observed release metadata, not an
acquired or admitted dataset. The checked browser exposes a Login link and has
no authenticated DISGENET session. Accepting the license terms does not itself
provide account access.

The official [download guidance](https://support.disgenet.com/support/solutions/articles/202000089413-can-i-download-the-disgenet-database-)
says academic users obtain search-result exports through the web interface or
API rather than a full-database bulk download. The Free Academic plan provides
curated data. Any such export must record version, filters, complete acquisition
scope/pagination and hashes before it can serve as the release input for panel
construction. A partial search result cannot establish genome-wide task
eligibility. Account access or a supplied official export is still needed.

The acquisition observations are retained in
`experiments/20261011_eval_hardened_v5/disgenet_acquisition_status.json`.

Claude's **§39.1** acquisition decision is now explicit: use Omkar's API key in
a Quest environment variable to export curated associations across the full
declared disease scope, record request parameters and every page's hash, filter
to `disease_type = disease`, inspect the export's own source identifiers and
freeze the curated-source allowlist, and retain the versioned inputs under
`data/raw/disgenet/v26.3/`. Final eligibility is derived only after complete
coverage is established. If complete curated coverage cannot be retrieved,
retain the returned scope and stop before building production panels/scoring.

## Required input bindings

The label manifest supplies:

- `disgenet_version`, `disgenet_release_date`, `disgenet_sha256`, `license`.
- `gene_mapping_path`, `gene_mapping_sha256`: TSV columns
  `entrez_gene_id`, `gene_id`; the mapping must be pinned to the intended
  GENCODE/certified-universe release.
- `positives_path`, `positives_sha256`: JSON of disease IDs to gene-ID lists.
  The gate independently rederives these sets from filtered release rows.
- `panels_sha256`, `panel_hash`: the actual panel file is hashed, and its
  threshold/development/final membership hash is recomputed. A claimed hash
  field alone is insufficient. Panel IDs must exist in the rederived labels.
- `curated_sources` and, when necessary, `column_map`.

Separate real-label preparation requires the gate's authorization and matching
positive sets. A child verifier checks the prepared train/test labels against
the pinned release before the fitting worker starts. Test labels then use the
same full-key seal/load protocol as shuffled diagnostics. This establishes
content derivation and integrity from the supplied pinned source; provider
authenticity still comes from acquiring the official release.

Formal panel construction/support, the ≥25 frozen-test-positive requirement,
the declared scientific comparison protocol, and certified feature admission
remain prerequisites for reporting scientific results.

# Provisional Q1 preview plots

These scripts and assets are reusable visual previews only. They are not
inserted into the manuscript and do not overwrite
`paper/ViPragSent_NAACL.pdf`.

## Inputs and schema

### Q1a

`source_data/table2_q1a_baselines.csv` is the existing paper-local artifact
CSV. Required identity/provenance fields are `system`, `backbone`, `n`,
`seeds`, and `scope`; each metric has paired `<metric>_mean` and
`<metric>_sd` fields for implicit sentiment, sarcasm, irony, idiom/figurative,
code-switching, mocking, and macro-pragmatic F1. The plot preserves the exact
`system` and `backbone` strings and requires exactly one `scope=primary` row.

### Q1b

`source_data/q1b_external_provisional_subset.csv` is an explicitly provisional
three-row snapshot transcribed from the current manuscript's existing Q1b
subset. It is not a coverage ledger. Required identity fields are `model`,
`backbone`, `scope`, `record_status`, `n`, and `seeds`; each of UIT-VSFC,
UIT-VSMEC, AIVIVN, and ordinary F1 has paired `_mean` and `_sd` fields.
`record_status=provisional-current-manuscript` is retained to prevent this
preview input from being mistaken for exhaustive evidence.

## Identity and evidence rules

- Exactly one primary row is required in each input. The primary is
  ViPragSent XLM-R-large; comparison identities remain artifact-specific.
- GPT labels are never inferred or renamed. Vistral-backed records remain
  Vistral-backed and are not described as XLM-R.
- Blank mean/SD pairs remain missing; no values are imputed.
- Values are displayed in percentage points. Q1a performs only the recorded
  [0, 1] to [0, 100] conversion; Q1b uses the manuscript's already reported
  percentage-point values.
- Every run writes a `.metadata.json` sidecar with the input SHA-256, row
  count, primary identity, and provisional coverage status.
- The partial `revision_evidence/` handoff is consulted only for coverage
  planning; its candidate numbers are not imported into these inputs. The
  `revision_q2/` handoff, including unresolved split-specific ECE labeling, is
  outside this plotting scope and is not used.

## Regeneration commands

```powershell
python paper/figures/plot_q1a_per_head_model_bars.py
python paper/figures/plot_q1b_external_faceted_bars.py
```

After `paper/revision_evidence/` arrives, pass its verified Q1a/Q1b input
files with `--input` and a new `--output-stem`. Do not overwrite these preview
files until the evidence package has been reconciled and the parent approves
the integrated manuscript figures.

## Full row-inventory working templates

The working inputs `source_data/q1a_full_row_inventory_working.csv` and
`source_data/q1b_full_row_inventory_working.csv` lay out the requested 11 Q1a
rows and 8 Q1b rows. Known current-subset values are retained only where they
already exist in the paper-local inputs; all other cells are blank and render
as `UNVERIFIED` placeholders. The GPT rows use artifact-identity-pending
labels and are not silently called GPT-4o or GPT-4.1-mini. Vistral-backed
variants remain distinct from the sole primary XLM-R-large row.

Generate the parent-review working layouts with:

```powershell
python paper/figures/plot_q1a_per_head_model_bars.py --input paper/figures/source_data/q1a_full_row_inventory_working.csv --output-stem paper/figures/figure_q1a_full_row_inventory_working_preview
python paper/figures/plot_q1b_external_faceted_bars.py --input paper/figures/source_data/q1b_full_row_inventory_working.csv --output-stem paper/figures/figure_q1b_full_row_inventory_working_preview
```

These are layout previews only. They must be regenerated from reconciled
canonical evidence before any manuscript integration.

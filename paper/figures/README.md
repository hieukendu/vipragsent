# Figure package for the revised manuscript

The current main-paper set is the architecture schematic, the expanded Q1a
macro-pragmatic F1 bar chart, and the compact Q3 and Q4 figures. The Q1a chart
is generated from the same 12-row canonical inventory as the main results
table, so the expanded baseline coverage is visible without replacing the
table.
These figure assets and their LaTeX placement are part of the completed paper
artifact; the final bounded manuscript pass did not change the figure values.

## Provisional Q1 previews for parent review

The following assets are explicitly provisional and are not inserted into the
manuscript or copied over `paper/ViPragSent_NAACL.pdf`:

- `figure_q1a_per_head_model_bars_preview.png/.pdf`: faceted bars for the six
  pragmatic heads plus macro, using the existing six-system Q1a artifact CSV.
- `figure_q1b_external_faceted_bars_preview.png/.pdf`: horizontal faceted bars
  for UIT-VSFC, UIT-VSMEC, AIVIVN, and ordinary-F1 using the current three-row
  Q1b subset.
- `plot_q1a_per_head_model_bars.py` and
  `plot_q1b_external_faceted_bars.py`: reusable schema-validating generators;
  each writes a metadata sidecar with the input SHA-256 and provisional status.
- `q1_preview_provenance.md`: input schema, identity rules, and regeneration
  commands.

The partial Q1 evidence handoff is used only for coverage planning. Its
candidate numbers are not plotted. The Q2 handoff and its split-specific ECE
diagnostic are not used by either preview.

## Recommended set

### Architecture: `figure_architecture_schematic.png` (inserted)

The manuscript inserts the clean high-resolution PNG to avoid PDF-import stroke
artifacts. The source vector PDF is preserved alongside it. Both are generated
from the paper-local `evidence/method_evidence.json`.
It shows the XLM-R-large hidden sequence, first-nonpadding pooling, six
pragmatic heads plus polarity/emotion, the training-only two-layer rationale
decoder, and the eight-task uncertainty objective with beta 0.3.

### Q1a legacy comparison: `figure_q1a_macro_f1_baselines.pdf` (preserved, not inserted)

PNG companion: `figure_q1a_macro_f1_baselines.png`.

Suggested caption:

> **Standalone Q1a comparison figure: macro-pragmatic F1 on the comparable test artifact cohort.**
> Points show means and whiskers show sample SD over three logical runs
> (`n=3`; logical run labels 21, 22, and 23, corresponding to remote artifact
> seeds 20260521, 20260522, and 20260523). The ViPragSent XLM-R-large target
> is highlighted; the five complete-run systems are comparison baselines.

LaTeX inclusion:

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/figure_q1a_macro_f1_baselines.pdf}
  \caption{Q1a macro-pragmatic F1 on the comparable test artifact cohort.
  Points show means and whiskers show sample SD over three logical runs
  ($n=3$; logical run labels 21, 22, and 23, corresponding to remote artifact
  seeds 20260521, 20260522, and 20260523). The ViPragSent XLM-R-large target
  is highlighted; the five complete-run systems are comparison baselines.}
  \label{fig:q1a-macro-f1}
\end{figure*}
```

Source data: `source_data/table2_q1a_baselines.csv`, copied byte-for-byte from
the validated comparison package. The plotting script performs only the
recorded unit conversion from [0, 1] to percentage points:
`plot_q1a_macro_f1.py`.

### Q1a expanded inventory: `figure_q1a_table_macro_bars.pdf` (inserted as Figure 2)

PNG companion: `figure_q1a_table_macro_bars.png`.

This horizontal bar chart uses all 12 rows from
`../revision_q1a_extra/q1a_extra_canonical_rows.csv`, including the primary,
complete baseline families, complete variants, one-shot records, and the
provisional two-of-three COT row. Bars show macro-pragmatic F1 means in
percentage points; whiskers are drawn only where the canonical row contains a
sample SD. The sidecar metadata records the input SHA-256, row statuses, and
seed coverage. Generator: `plot_q1a_table_macro_bars.py`.

### Q3: `figure_q3_sarcasm_budget.png` (inserted as Figure 3)

Suggested caption:

> **Figure 3: Sarcasm F1 across pragmatic-label budgets.** Curves show the
> artifact-derived mean binary F1 at each positive-example budget; error bars
> show sample SD where the three-seed cohort is complete (`n=3`; logical run
> labels 21, 22, and 23, corresponding to remote artifact seeds 20260521,
> 20260522, and 20260523). The PhoBERT curve has gaps at 64, 512, and full.
> The target curve is shown with PhoBERT fine-tune and Vistral-7B SFT for
> context.

LaTeX inclusion:

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/figure_q3_sarcasm_budget.png}
  \caption{Sarcasm F1 across pragmatic-label budgets. Curves show the
  artifact-derived mean binary F1 at each positive-example budget; error bars
  show sample SD over three logical runs ($n=3$; logical run labels 21, 22,
  and 23, corresponding to remote artifact seeds 20260521, 20260522, and
  20260523). The target curve is shown with PhoBERT fine-tune and Vistral-7B
  SFT for context.}
  \label{fig:q3-sarcasm-budget}
\end{figure}
```

This figure is regenerated from the local table rather than copied from the
older package preview. The corresponding local table is
`source_data/table_q3_low_resource.csv`. The PhoBERT cohort audit is recorded
in `source_data/q3_phobert_cohort_audit.md/.json`; missing cells are left
blank in the source CSV and are not imputed in the plot.

### Q4: `figure_q4_reliability.png` (inserted as Figure 4)

Suggested caption:

> **Figure 4: Reliability diagrams for the Q4 calibration comparison.** Curves
> show empirical positive rates against mean confidence for ViPragSent
> XLM-R-large and Vistral-7B SFT; the dashed diagonal denotes perfect
> calibration. The artifact summary reports mean ECE plus sample SD over three
> logical runs (`n=3`; run labels 21, 22, and 23, corresponding to remote
> artifact seeds 20260521, 20260522, and 20260523). In the supplied summary,
> Vistral has the lower mean ECE.

LaTeX inclusion:

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/figure_q4_reliability.png}
  \caption{Reliability diagrams for the Q4 calibration comparison. Curves
  show empirical positive rates against mean confidence for ViPragSent
  XLM-R-large and Vistral-7B SFT; the dashed diagonal denotes perfect
  calibration. The artifact summary reports mean ECE plus sample SD over three
  logical runs ($n=3$; run labels 21, 22, and 23, corresponding to remote
  artifact seeds 20260521, 20260522, and 20260523). In the supplied summary,
  Vistral has the lower mean ECE.}
  \label{fig:q4-reliability}
\end{figure}
```

This is a direct copy of the validated package figure; no pixels or values
were regenerated. The corresponding local table is
`source_data/table_q4_calibration.csv`. The Q4 training-history figure is not
included in the recommended main-paper set because it is redundant for the
calibration question.

## Provenance and figure trace

Source package: `D:\vipragsent-pr\reports\hf_vipragsent_naacl_comparison_2026-09-15`.
The package records complete three-seed artifact coverage for these rows but no
independent training/inference rerun. The structure-reference PDF is not used
as a source for numbers.

| Artifact | Source or transformation | Local SHA-256 | Limitation |
|---|---|---|---|
| `figure_architecture_schematic.pdf` | Preserved source vector PDF generated from `evidence/method_evidence.json`; validation checks the primary XLM-R-large, pooling, heads, rationale-training-only path, and objective terms | `8F05496D68BAFE9644847F41BDD5AA7C26C945976811D024FA5093166291C304` | Architecture is method evidence, not an independent training claim |
| `figure_architecture_schematic.png` | High-resolution raster generated from the same validated schematic and inserted by the manuscript | `F8112E355D6B78EFA0415EBF6BE15B75849DC61FB4A98E89EAA4BE017F3CEDAC` | Architecture is method evidence, not an independent training claim |
| `figure_q1a_macro_f1_baselines.pdf` | Generated by `plot_q1a_macro_f1.py` from the local exact copy of `table2_q1a_baselines.csv`; values multiplied by 100 only for percent display | `906F60873D5D4954BAD8F5C60FD8D42932C678E6F53B86090E1D9788D8088D4F` | Artifact-level comparison; no independent rerun or significance test |
| `figure_q1a_macro_f1_baselines.png` | Same as PDF | `9575DAEA1A8909D51497D7D24B496F144010AA1C34E7B8A3E4F8F87FFE90D969` | Same limitation |
| `figure_q3_sarcasm_budget.png` | Regenerated from `source_data/table_q3_low_resource.csv`; incomplete PhoBERT cells are represented as gaps | `19355F9B70B4DC66667F235515C22F7A9117FBCF696E5BD5C3CCBAACBA00B85B` | Descriptive low-resource curve; no monotonicity or dominance claim |
| `figure_q4_reliability.png` | Byte-for-byte copy of source `figure7_q4_reliability_diagrams.png` | `04B623D35F1F109C7401E06202BC71B9CFAD575FE90215F64643E6C79248947D` | Descriptive calibration comparison; lower ECE belongs to Vistral in the supplied summary |

The earlier `figure2_q1a_baseline_grouped_bars.png` remains in the folder for
provenance continuity but is superseded for manuscript insertion by the compact
macro-F1 figure above.

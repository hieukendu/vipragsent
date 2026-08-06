# Private source archive

These three XLSX files are the original human annotation artifacts supplied by the project owner:

1. Human annotator 1
2. Human annotator 2
3. Human adjudicator / final gold master

The workbooks are preserved byte-for-byte for auditability. Some legacy metadata strings inside the
original workbook notes contain incorrect `AI_*` wording inserted during an earlier packaging step.
Those strings are not used by the experiment pipeline.

Use the cleaned CSV files in `01_clean_human_annotations/` and `02_vipragsent/` for training,
evaluation, IAA reporting, and release preparation.

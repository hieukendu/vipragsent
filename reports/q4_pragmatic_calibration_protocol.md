# Q4 pragmatic calibration protocol

Status: `RESOLVED`

Q4 is pragmatic calibration and learning dynamics for exactly `phobert_pragmatic_finetune`, `vistral_pragmatic_sft`, and `vipragsent_full_vistral`. Each system exposes the same six pragmatic positive-class sigmoid probabilities.

Calibration uses the frozen ViPragSent test split, ten equal-width bins, no temperature scaling, no thresholding, and no probability pooling across seeds. ECE is computed independently per seed and summarized by arithmetic mean and sample standard deviation (`ddof=1`). Learning curves use only frozen ViPragSent dev histories and dev macro pragmatic F1 by epoch.

Required tables, reliability backing data, learning curves, and PDF/PNG figures are listed in `configs/experiments/q4/pragmatic_calibration.yaml`.

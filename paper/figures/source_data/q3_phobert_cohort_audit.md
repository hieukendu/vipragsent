# Q3 PhoBERT cohort audit

The Q3 PhoBERT rows were checked against the current Hugging Face artifact
trees before finalizing Table 4. Two non-equivalent artifact cohorts exist.

The standard GPU40 cohort used by the existing table supports complete
three-seed test aggregates at budgets 32, 128, and 256. The live audit found
only seed 20260523 with a test metric at budget 64; seeds 20260521 and
20260522 are not complete test runs there. No complete standard-cohort test
aggregate was found for 512 or `full`.

A separate approved compact-final campaign is complete for 32, 128, 512, and
`full`. Its live test aggregates are:

| Budget | Macro-pragmatic F1 | Sarcasm F1 |
|---:|---:|---:|
| 32 | 0.7774219820 +/- 0.1081268039 | 0.6899713371 +/- 0.0174507349 |
| 128 | 0.8461866784 +/- 0.0037504633 | 0.7259523268 +/- 0.0114875792 |
| 512 | 0.7808768625 +/- 0.0913237592 | 0.7078561305 +/- 0.0275718127 |
| full | 0.8448127686 +/- 0.0010369229 | 0.7328306380 +/- 0.0083132085 |

The compact-final 32 and 128 values differ from the standard GPU40 values, and
the compact-final cohort has no 64 or 256 rows. It is therefore not merged
into Table 4. Table 4 retains the standard-cohort values at 32/128/256 and
prints `--` for 64/512/`full`; no value is imputed.

Representative live Hugging Face tree:

<https://huggingface.co/Thundergod2007/vipragsent-experiment-artifacts-overflow-002/tree/main/campaigns/vipragsent-v7-compact-20260820-190bb0f24a7e75a8/SHARD_H100_MIG20_COMPACT>

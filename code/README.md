# ViPragSent

Training and validation code aligned to the supplied paper PDF. Canonical data
are in `../paper/ARR_clean` in this workspace. After extracting the two anonymous
archives together, use sibling `code/` and `ARR_clean/` directories.

```bash
python -B prepare_inputs.py
python -B -m pytest tests -p no:cacheprovider --basetemp=.work/pytest_tmp
python -B build_submission.py
python -B verify_submission.py
```

`src/` implements models, losses, training and evaluation. `scripts/` contains
runners and analyses; `configs/` locks their recipes; `prompts/` holds protocols
and fixed demonstrations; `tests/` checks contracts. Prepared `data/` is local
runtime input, excluded from the software ZIP. Credentials, weights, downloads,
private recovery backups and generated execution outputs are excluded.

See `README_REPRODUCE.md` for commands and `docs/SUBMISSION_REAUDIT.md` for evidence
and limitations. Saved arithmetic validation passed; training, API inference and
bootstrap were not rerun. The PDF's 17 historical annotation notes remain unavailable;
the user waived their recovery. See the dataset's `AVAILABILITY.json` for this
documented limitation. Administrative submission requirements are not assessed.

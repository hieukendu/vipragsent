> Paste the entire contents of this file into Codex at the corresponding phase.
> Codex must perform only the current phase, run its tests, create the required handoff files, and then stop.
> From Phase 00 through Phase 14, downloading real model weights and running full experiments are prohibited.
> Model weights may be downloaded only in Phase 15. The complete experiment suite may be executed only through the one-click command in Phase 16.

# PHASE 00 — BOOTSTRAP THE REPOSITORY

Create the complete Python repository skeleton without ingesting datasets or downloading models.

Required work:

- Locate the V8 dataset ZIP.
- Initialize Git if needed.
- Create the repository structure from `20_EXPECTED_REPOSITORY_TREE.md`.
- Create `pyproject.toml`, dependency-lock strategy, `Makefile`, `README.md`, `.gitignore`, `.env.example`, and `PROJECT_STATE.json`.
- Create the Python package under `src/vipragsent`.
- Configure `pytest`, `ruff`, and reasonable type checking.
- Compute and record the SHA-256 checksum of the input ZIP.
- Create `scripts/check_environment.py`, `scripts/hash_artifacts.py`, and `scripts/validate_project_layout.py`.
- Implement the phase status and handoff framework.

Do not extract the dataset, download external datasets, download model weights, train models, or run full Azure jobs.

Acceptance criteria: package import succeeds, `pytest` runs, project-layout validation passes, the input checksum is recorded, and secrets/private artifacts are ignored by Git.


# BASE ENVIRONMENT TARGET

Target Python 3.11.

The setup must also detect:

- CUDA and NVIDIA driver;
- A100 versus A100 MIG profile;
- Java 17 LTS for VnCoreNLP;
- available system RAM and disk;
- network and authentication requirements for Hugging Face, Kaggle, and Azure.

Do not hard-code a dependency lock before compatibility resolution. Produce the final lock only after dummy tests
and Phase 15 model-load smoke tests have passed.

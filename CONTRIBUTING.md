# Contributing

Thanks for your interest in improving HeartScan AI!

## Getting set up

```bash
git clone https://github.com/IhebKhalfallaah/ECG-Signal-Prediction-.git
cd ECG-Signal-Prediction-
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

## Workflow

1. Open an issue describing the bug or improvement before large changes.
2. Create a branch from `main` (`feature/<short-name>` or `fix/<short-name>`).
3. Keep changes focused and add or update tests in `tests/`.
4. Make sure `pytest` passes, then open a pull request explaining **what** changed and **why**.

## Guidelines

- **Signal processing lives in `ecg/`**, the web layer in `app/`. Keep `ecg/` free of Flask imports.
- **Prediction behavior is pinned** by golden tests (`tests/fixtures/`). If a change intentionally alters features,
  filters or predictions, regenerate the fixtures in the same pull request and explain why.
- The model in `models/` was trained on features produced by `ecg/` with the current filter settings —
  changing the filters requires retraining (see `notebooks/`).
- Follow PEP 8 and keep functions short with descriptive names.
- Never commit patient data, raw datasets or credentials.

## Commit messages

Use the [Conventional Commits](https://www.conventionalcommits.org/) style, e.g. `fix: handle records without lead V4`.

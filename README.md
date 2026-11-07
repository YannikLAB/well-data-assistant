# Well Data Assistant

A question-answering assistant over Oklahoma water-well records. It answers with
numbers from the state's records, says how many wells each answer rests on, explains
well reports in plain language, and declines when the records can't answer.

Status: week 1. Data loading, data-quality checks and the first query function.

## Setup

You need Docker Desktop, Python 3.11 or newer, and the Oklahoma data folder.

```bash
cp .env.example .env        # then set WELL_DATA_DIR to the data folder
make install                # creates .venv (use PYTHON=python3.12 to pick a version)
make db                     # starts Postgres with PostGIS on port 5433
make load                   # loads the records (about 30 seconds)
make test                   # runs the tests
```

`make load` prints the data issues it found and writes them to `reports/data_issues.csv`.

## Data policy

- The data never goes in this repo. The loader reads it from `WELL_DATA_DIR`.
- Owner names and addresses are never read, so they can't reach the database or the repo.
- Tests use synthetic wells only.

## What's here

| Path | Purpose |
|---|---|
| `src/welldata/sources.py` | Reads the state table and the report extractions into tidy tables |
| `src/welldata/quality.py` | pandera checks; blanks impossible values, drops rows outside Oklahoma |
| `src/welldata/db.py`, `sql/schema.sql` | Creates the PostGIS tables and copies the data in |
| `src/welldata/queries.py` | Query functions the assistant will call (first: wells within a radius) |
| `tests/` | Unit tests, plus database tests that run in a throwaway schema |
| `.github/workflows/ci.yml` | Runs all tests against PostGIS on every push and pull request |

## Data notes

- "Drawdown depth" in the pump tests is the depth to water while pumping. Drawdown is that minus the start depth (`pump_tests.drawdown_ft`).
- Pump-test duration has no unit on the report.
- The Oklahoma check is a bounding box; a state outline would be stricter.
- OWRB's online well log application is disabled, so the report links currently go to the OWRB homepage.

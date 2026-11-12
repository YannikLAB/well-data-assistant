# Well Data Assistant

A question-answering assistant over Oklahoma water-well records. It answers with
numbers from the state's records, says how many wells each answer rests on, explains
well reports in plain language, and declines when the records can't answer.

Status: week 4. Data loading and checks, fields extracted from the saved reports, four
query tools, a 50-question evaluation set with gold answers, and an evaluation runner
comparing three systems.

## Setup

You need Docker Desktop, Python 3.11 or newer, and the Oklahoma data folder.

```bash
cp .env.example .env        # then set WELL_DATA_DIR to the data folder
make install                # creates .venv (use PYTHON=python3.12 to pick a version)
make db                     # starts Postgres with PostGIS on port 5433
.venv/bin/python -m welldata.extract_reports   # parses the saved reports (a few minutes)
make load                   # loads everything (about 30 seconds)
make test                   # runs the tests
```

`make load` prints the data issues it found and writes them to `reports/data_issues.csv`.
The extraction writes `report_fields.csv`, `lithology.csv` and `comments.csv` to
`WELL_DATA_DIR/data/derived`. Use `--start` and `--stop` to process a slice of the reports.

## Query tools

The assistant will call these instead of writing SQL. Each returns plain JSON-ready data,
says how many wells support each number, and raises `ToolInputError` with a fix when an
argument is wrong. Test holes are excluded.

| Tool | Answers |
|---|---|
| `summarize_area(latitude, longitude, radius_miles=0.5, use_class=None)` | Depth, first water, yield (incl. share reaching 10 gpm), screened interval and plugged wells nearby |
| `find_nearest_pump_test(latitude, longitude, use_class=None, max_miles=25)` | The closest pump test, its distance and drawdown |
| `get_water_level_trend(latitude, longitude, radius_miles=5, use_class=None)` | Median depth to water at drilling by decade, with the change between decades |
| `get_well_record(well_id)` | Everything known about one well, including lithology and comments, without owner details |

## Evaluation

`evals/` holds 25 dev and 25 frozen test questions with gold answers computed separately
from the tools, a reworded copy of the dev set, and 21 hard dev questions (no location,
swapped coordinates, comparisons, several parts, injected instructions, blank reports,
drillers' comments). See `evals/README.md`.

```bash
.venv/bin/python -m welldata.check_evalset
.venv/bin/python -m welldata.run_eval --system keyword --split dev_hard
.venv/bin/python -m welldata.run_eval --system assistant --split dev_paraphrased
.venv/bin/python -m welldata.rescore results/<run>.jsonl
.venv/bin/python -m welldata.faithfulness results/<assistant run>.jsonl --export-labels results/labels.csv
.venv/bin/python -m welldata.faithfulness --agreement results/labels.csv --judgments results/<assistant run>.faithfulness.jsonl
```

Systems: `keyword` (rules, no LLM), `no_tools` (Claude without the database) and
`assistant` (Claude calling the query tools). LLM systems need `ANTHROPIC_API_KEY` in `.env`
and default to `claude-haiku-5-5`. Every answer ends with a structured `submit_answer` call
listing the numbers it states, so `welldata.scoring` grades it without an LLM judge: it
checks the required numbers, that every stated number matches a tool output, and that the
text is plain. Explanations of a well record also get a sentence-by-sentence check by a
separate judge model (`claude-sonnet-5-5`), validated against hand labels. Responses are
cached in `.cache/llm`; `--repeat N` measures run-to-run variation.

Accuracy with 95% Wilson intervals:

| System | dev (templated) | dev_paraphrased | dev_hard |
|---|---|---|---|
| keyword | 100% (87–100%) | 40% (23–59%) | 33% (17–55%) |
| no_tools (Haiku) | not run | 12% (4–30%) | 19% (8–40%) |
| assistant (Haiku, prompt of Nov 12 a.m.) | not run | 92% (75–98%) | 86% (65–95%) |

In those assistant runs every unsupported claim was a labeling gap (a single well's yield
filed as a median, the yield count filed as `well_count`), not a made-up number; 7 of 46
first submissions leaked tool-call markup.

## Data policy

- The data never goes in this repo. The loader reads it from `WELL_DATA_DIR`.
- Owner names and addresses are never read, so they can't reach the database or the repo.
- Tests use synthetic wells and a synthetic report only.
- Driller comments can mention people or places. Review them before any public demo.

## What's here

| Path | Purpose |
|---|---|
| `src/welldata/sources.py` | Reads the state table, the report extractions and the extracted report fields |
| `src/welldata/quality.py` | pandera checks; blanks impossible values, drops rows outside Oklahoma |
| `src/welldata/reports.py`, `extract_reports.py` | Parses depth to water, lithology and comments from the saved reports |
| `src/welldata/db.py`, `sql/schema.sql` | Creates the PostGIS tables and copies the data in |
| `src/welldata/tools.py` | The query tools above |
| `src/welldata/gold.py`, `evalset.py`, `hard_questions.py`, `check_evalset.py` | Gold answers, question-set builders, and the tools-vs-gold check |
| `src/welldata/llm.py`, `agent.py` | Anthropic provider with a response cache, and the tool-use loop |
| `src/welldata/answers.py`, `scoring.py` | The structured answer format and the grader |
| `src/welldata/systems.py`, `run_eval.py` | The three systems and the evaluation runner |
| `src/welldata/faithfulness.py` | Sentence-level judge for record explanations, label export and judge-human agreement |
| `tests/` | Unit tests, plus database tests that run in a throwaway schema |
| `.github/workflows/ci.yml` | Runs all tests against PostGIS on every push and pull request |

## Data notes

- "Drawdown depth" in the pump tests is the depth to water while pumping. Drawdown is that minus the start depth (`pump_tests.drawdown_ft`).
- About 5% of pump tests with both depths show a negative drawdown, which points to swapped or mistyped values. The tools skip or flag them.
- Pump-test duration has no unit on the report.
- About half the reports record depth to water at the time of drilling.
- The Oklahoma check is a bounding box; a state outline would be stricter.
- OWRB's online well log application is disabled, so the report links currently go to the OWRB homepage.

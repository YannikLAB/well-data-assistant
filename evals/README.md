# Evaluation questions

50 customer-style questions with gold answers, built by `python -m welldata.evalset`
(seed 2026, data as of 2026-11-08).

| File | Use |
|---|---|
| `questions_dev.jsonl` (25) | Look at these, tune prompts and tools against them |
| `questions_test.jsonl` (25) | Frozen. Run only at milestones; never tune against it |
| `questions_test.sha256` | A test fails if the frozen file changes |

## Question types

| Type | Count | Expected tool |
|---|---|---|
| `area_depth` | 9 | `summarize_area` |
| `area_yield` | 8 | `summarize_area` |
| `area_screens` | 5 | `summarize_area` |
| `area_plugged` | 4 | `summarize_area` |
| `nearest_pump_test` | 5 | `find_nearest_pump_test` |
| `water_level_trend` | 5 | `get_water_level_trend` |
| `report_explanation` | 7 | `get_well_record` |
| `unanswerable` | 7 | none, or a tool that should return an error |

Each line holds `id`, `split`, `type`, `question`, `expected_tool`, `expected_args` and
`gold`. Unanswerable questions have `gold.behavior` set to `decline` (or
`decline_guarantee_give_evidence` when nearby yields should still be shown) and a
`gold.reason`.

## How gold answers are made

`welldata.gold` computes them with pandas and WGS84 geodesic distances, separately from
the SQL in `welldata.tools`. `python -m welldata.check_evalset` calls each expected tool
and compares; on 2026-11-08 all 50 matched. That check shows the best an assistant could
score by always choosing the right tool; it doesn't test an LLM.

## Privacy

Area questions use a point 0.1–0.3 miles from a real well, rounded to about 10 m, so no
question points at someone's home. Gold answers never include owner details or comment text.

## Adding questions

Add new questions to the dev set. Growing the test set to ~100 means rebuilding both files
on purpose with `--force` and committing the new checksum, before any results are reported.

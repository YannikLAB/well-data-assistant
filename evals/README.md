# Evaluation questions

50 customer-style questions with gold answers, built by `python -m welldata.evalset`
(seed 2026, data as of 2026-11-08).

| File | Use |
|---|---|
| `questions_dev.jsonl` (25) | Look at these, tune prompts and tools against them |
| `questions_test.jsonl` (25) | Frozen. Run only at milestones; never tune against it |
| `questions_test.sha256` | A test fails if the frozen file changes |
| `questions_dev_paraphrased.jsonl` (25) | The dev questions reworded (synonyms, coordinate formats like "N 35.68, W 95.48"), same gold answers |
| `questions_dev_hard.jsonl` (21) | Hard inputs the templates don't cover; built by `python -m welldata.hard_questions` |

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

## Why the paraphrased set exists

The generated questions follow templates, so simple keyword rules answer all of them: the
keyword baseline scores 100% on `questions_dev.jsonl` and 40% on the paraphrased version.
Measure language understanding on the paraphrased set, and replace the paraphrases with
real users' wording as interviews provide it. Paraphrases were written by hand to vary
wording only; review them before trusting results.

## The hard dev set

| Type | Count | What a good answer does |
|---|---|---|
| `missing_location`, `town_only` | 2, 2 | Declines and asks for coordinates or a well ID (there is no geocoding tool yet) |
| `swapped_coordinates`, `missing_minus` | 1, 1 | Fixes the coordinates and says so |
| `dms_coordinates`, `feet_radius` | 1, 1 | Converts degrees-minutes-seconds and feet |
| `radius_out_of_range` | 1 | Uses the 25-mile maximum and says so |
| `comparison` | 2 | Calls the tool once per place; claims carry `location_index` 1 or 2 |
| `multi_part` | 2 | Answers every part from its own call |
| `injection`, `injection_mixed` | 1, 1 | Never shares owner details; the mixed one still answers the depth |
| `blank_report` | 1 | States the depth and invents nothing else (`must_not_claim`) |
| `driller_comment`, `tds_safety` | 2, 1 | Reports the driller's TDS note; declines to call water safe and points to a lab test |
| `well_plugging`, `domestic_trend` | 1, 1 | Single-well plugging date; a trend for home wells only |

Each item lists `expected_calls` (one or more tool calls) and gives its gold answer as
explicit `gold.checks` (an `any_of` entry passes on any of its alternatives), plus optional
`text_must_include` (each entry is a list of acceptable words), `must_not_claim`,
`allowed_well_ids` and `decline_either_way`. The
oracle check covers them too: on 2026-11-12 all 71 questions matched. The DMS question
uses a point where a rounding-sized change (about 20 m) doesn't change which wells are in
range. Comment questions never include the comment text, only the words an answer needs.

## What the grader measures

- **Accuracy**: every required check passes (numbers within tolerance, the right decline,
  required words). The 95% interval is a Wilson score interval; with repeated runs it uses
  the number of distinct questions.
- **Claims supported**: every claim, required or not, matches a value in that run's tool
  outputs. A low rate means made-up or mislabeled numbers.
- **Unmatched numbers**: numbers in the answer text found in no tool input or output and not
  in the question (allowing rounding and percentages). A reading aid, not a score: derived
  numbers such as differences also show up.
- **Well-formed first try / plain-text answers**: whether the model's first `submit_answer`
  call was well formed (no tool-call markup in the text, `declined` present), and whether the
  final text is plain. The agent gives one retry; if the second try is malformed too, it cuts
  the markup from the text and marks the answer repaired.

Yield questions accept either count: all wells in the radius (`well_count`) or the wells
that report a yield (`yield_wells`), since the share rests on the latter. This was added on
2026-11-12 after the first Haiku runs labeled the yield count `well_count`, when no better
label existed. `python -m welldata.rescore <results file>` grades a saved run again with
the current grader, without calling a model.

## Adding questions

Add new questions to the dev set. Growing the test set to ~100 means rebuilding both files
on purpose with `--force` and committing the new checksum, before any results are reported.

"""The hard dev questions: helpers, and the shape of the committed file."""

import json
from pathlib import Path

from welldata.answers import QUANTITIES
from welldata.hard_questions import to_dms

HARD = Path(__file__).parent.parent / "evals" / "questions_dev_hard.jsonl"
TOOLS = {"summarize_area", "find_nearest_pump_test", "get_water_level_trend", "get_well_record"}


def test_to_dms_writes_degrees_minutes_seconds_and_returns_the_exact_point():
    text, lat, lon = to_dms(35.47, -97.518056)
    assert text == "35°28'12\"N, 97°31'05\"W"
    assert (lat, lon) == (35.47, -97.518056)


def test_every_hard_question_has_calls_and_gradable_gold():
    items = [json.loads(line) for line in HARD.read_text().splitlines() if line.strip()]
    assert len(items) >= 20 and len({i["id"] for i in items}) == len(items)
    for item in items:
        g = item["gold"]
        assert {c["tool"] for c in item["expected_calls"]} <= TOOLS
        assert all(c["quantity"] in QUANTITIES for c in g["checks"])
        # Every answerable question must check something, in numbers or in words.
        assert g.get("behavior", "").startswith("decline") or g["checks"] or g.get("text_must_include")

import hashlib
import random
from pathlib import Path

import pytest
from geographiclib.geodesic import Geodesic

from welldata import gold
from welldata.evalset import BuildSettings, build_questions, offset_point, split_questions

SMALL = BuildSettings(min_wells=1, min_decade_wells=1, min_decades=2, min_lithology_rows=2)
SPEC = {"area_depth": 2, "area_yield": 1, "area_screens": 1, "area_plugged": 1,
        "nearest_pump_test": 1, "water_level_trend": 1, "report_explanation": 1, "unanswerable": 7}
EVALS = Path(__file__).parents[1] / "evals"


def miles_between(a, b):
    return Geodesic.WGS84.Inverse(a[0], a[1], b[0], b[1])["s12"] / 1609.344


def test_offset_point_moves_a_short_distance_and_rounds():
    rng = random.Random(1)
    for _ in range(200):
        lat, lon = offset_point(35.5, -97.5, rng)
        assert 0.09 <= miles_between((35.5, -97.5), (lat, lon)) <= 0.31
        assert round(lat, 4) == lat and round(lon, 4) == lon


def test_area_gold_matches_the_reference_answer(sample_tables):
    items = build_questions(sample_tables, {"area_depth": 2, "area_yield": 1}, seed=3, settings=SMALL)
    for item in items:
        args = item["expected_args"]
        reference = gold.area_summary(sample_tables, args["latitude"], args["longitude"],
                                      args["radius_miles"], args["use_class"])
        assert item["expected_tool"] == "summarize_area"
        assert all(item["gold"][k] == reference[k] for k in item["gold"])


def test_every_type_is_built_with_question_text_and_gold(sample_tables):
    items = build_questions(sample_tables, SPEC, seed=3, settings=SMALL)
    assert sorted({i["type"] for i in items}) == sorted(SPEC)
    assert len(items) == sum(SPEC.values())
    assert all(i["question"] and i["gold"] for i in items)
    assert len({i["id"] for i in items}) == len(items)


def test_unanswerable_items_expect_a_decline(sample_tables):
    items = build_questions(sample_tables, SPEC, seed=3, settings=SMALL)
    declines = [i for i in items if i["type"] == "unanswerable"]
    assert len(declines) == 7
    assert all(i["gold"]["behavior"].startswith("decline") for i in declines)
    assert all(i["gold"]["reason"] for i in declines)


def test_same_seed_gives_same_questions(sample_tables):
    assert build_questions(sample_tables, SPEC, seed=5, settings=SMALL) == \
        build_questions(sample_tables, SPEC, seed=5, settings=SMALL)


def test_split_is_stratified_and_balanced(sample_tables):
    items = build_questions(sample_tables, SPEC, seed=3, settings=SMALL)
    dev, test = split_questions(items, seed=3)
    assert abs(len(dev) - len(test)) <= 1
    for kind, n in SPEC.items():
        if n >= 2:
            assert any(i["type"] == kind for i in dev) and any(i["type"] == kind for i in test)
    assert {i["id"] for i in dev}.isdisjoint({i["id"] for i in test})


def test_frozen_test_set_is_unchanged():
    path = EVALS / "questions_test.jsonl"
    if not path.exists():
        pytest.skip("Test set not generated yet.")
    recorded = (EVALS / "questions_test.sha256").read_text().split()[0]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == recorded, (
        "evals/questions_test.jsonl changed. The test set is frozen: add new questions to the "
        "dev set instead, or regenerate both files on purpose and record the new checksum.")

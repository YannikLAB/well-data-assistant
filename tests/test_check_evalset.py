"""The oracle check: call each question's expected tool with its expected arguments and
compare with the gold answer. On real data this should report no mismatches; any
mismatch means a tool or the gold code has a bug."""

import copy

import pytest

from welldata.check_evalset import check_items
from welldata.evalset import build_questions

pytestmark = pytest.mark.db
SPEC = {"area_depth": 2, "area_yield": 1, "area_screens": 1, "area_plugged": 1,
        "nearest_pump_test": 1, "water_level_trend": 1, "report_explanation": 1, "unanswerable": 7}


@pytest.fixture
def items(sample_tables):
    from welldata.evalset import BuildSettings
    return build_questions(sample_tables, SPEC, seed=3,
                           settings=BuildSettings(min_wells=1, min_decade_wells=1, min_decades=2,
                                                  min_lithology_rows=2))


def test_tools_match_gold_on_every_question(loaded_db, items):
    assert check_items(loaded_db, items) == []


def test_a_wrong_gold_value_is_reported(loaded_db, items):
    broken = copy.deepcopy(items)
    area = next(i for i in broken if i["type"] == "area_depth")
    area["gold"]["depth_ft"]["median"] = 12345.0
    problems = check_items(loaded_db, broken)
    assert len(problems) == 1
    assert problems[0]["id"] == area["id"] and "depth_ft" in problems[0]["detail"]


def test_declines_expect_a_tool_error(loaded_db, items):
    wichita = next(i for i in items if "Wichita" in i["question"])
    missing = next(i for i in items if "999999999" in i["question"])
    assert check_items(loaded_db, [wichita, missing]) == []


# Hard questions: explicit checks against the expected calls ------------------------------

from conftest import CENTER  # noqa: E402
from welldata import gold  # noqa: E402


def area_call(radius):
    return {"tool": "summarize_area", "args": {"latitude": CENTER[0], "longitude": CENTER[1],
                                               "radius_miles": radius, "use_class": None}}


def test_hard_checks_are_compared_with_the_call_for_their_location(loaded_db, sample_tables):
    wide = gold.area_summary(sample_tables, *CENTER, 5, None)["well_count"]
    near = gold.area_summary(sample_tables, *CENTER, 0.5, None)["well_count"]
    assert wide != near
    item = {"id": "h1", "type": "comparison", "expected_calls": [area_call(5), area_call(0.5)],
            "gold": {"checks": [{"quantity": "well_count", "value": wide, "location_index": 1},
                                {"quantity": "well_count", "value": near, "location_index": 2}]}}
    assert check_items(loaded_db, [item]) == []
    item["gold"]["checks"][0]["location_index"] = 2
    assert [p["id"] for p in check_items(loaded_db, [item])] == ["h1"]


def test_must_not_claim_is_checked_against_the_record(loaded_db, sample_tables):
    fields = sample_tables["report_fields"].dropna(subset=["depth_to_water_ft"])
    wid = int(fields["well_id"].iloc[0])
    item = {"id": "h2", "type": "blank_report",
            "expected_calls": [{"tool": "get_well_record", "args": {"well_id": wid}}],
            "gold": {"checks": [], "must_not_claim": ["depth_to_water_ft"]}}
    problems = check_items(loaded_db, [item])
    assert len(problems) == 1 and "depth_to_water_ft" in problems[0]["detail"]

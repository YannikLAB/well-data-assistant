"""Deterministic grading of an answer against a question's gold answer."""

import pytest

from welldata.answers import Answer
from welldata.scoring import score, summarize

AREA = {
    "id": "q1", "type": "area_depth", "expected_tool": "summarize_area",
    "gold": {"well_count": 14, "nearest_well_ids": [5, 6, 7],
             "depth_ft": {"wells": 14, "median": 132.5},
             "first_water_ft": {"wells": 12, "median": 60.0}},
}
TREND = {
    "id": "q2", "type": "water_level_trend", "expected_tool": "get_water_level_trend",
    "gold": {"decades": [
        {"decade": 1990, "wells": 7, "median_depth_to_water_ft": 20.0, "change_ft": None, "few_wells": False},
        {"decade": 2000, "wells": 2, "median_depth_to_water_ft": 31.0, "change_ft": 11.0, "few_wells": True},
        {"decade": 2010, "wells": 40, "median_depth_to_water_ft": 25.0, "change_ft": -6.0, "few_wells": False}]},
}
SAFETY = {"id": "q3", "type": "unanswerable", "expected_tool": None,
          "gold": {"behavior": "decline", "reason": "No water-quality data."}}
GUARANTEE = {"id": "q4", "type": "unanswerable", "expected_tool": "summarize_area",
             "gold": {"behavior": "decline_guarantee_give_evidence", "reason": "No guarantee.",
                      "evidence": {"well_count": 40, "yield_gpm": {"wells": 38, "median": 15.0,
                                                                   "share_at_least_10_gpm": 0.82}}}}


def answer(claims=(), declined=False, cited=()):
    return Answer(text="...", declined=declined, claims=list(claims), cited_well_ids=list(cited))


def claim(quantity, value, decade=None):
    c = {"quantity": quantity, "value": value}
    if decade is not None:
        c["decade"] = decade
    return c


def test_area_answer_with_all_numbers_is_correct():
    a = answer([claim("well_count", 14), claim("depth_median_ft", 130), claim("first_water_median_ft", 60)],
               cited=[5, 6])
    result = score(AREA, a)
    assert result["correct"] and result["checks_passed"] == 3
    assert result["citations_ok"] is True


def test_rounding_within_tolerance_passes_but_a_wrong_median_fails():
    ok = score(AREA, answer([claim("well_count", 14), claim("depth_median_ft", 135),
                             claim("first_water_median_ft", 60)]))
    wrong = score(AREA, answer([claim("well_count", 14), claim("depth_median_ft", 200),
                                claim("first_water_median_ft", 60)]))
    assert ok["correct"]
    assert not wrong["correct"]
    assert any("depth_median_ft" in f for f in wrong["failures"])


def test_missing_number_fails():
    result = score(AREA, answer([claim("well_count", 14), claim("depth_median_ft", 132.5)]))
    assert not result["correct"] and result["checks_passed"] == 2


def test_citing_a_well_outside_the_area_is_flagged():
    a = answer([claim("well_count", 14), claim("depth_median_ft", 132.5), claim("first_water_median_ft", 60)],
               cited=[5, 999])
    assert score(AREA, a)["citations_ok"] is False


def test_trend_checks_only_decades_with_enough_wells():
    a = answer([claim("decade_median_depth_to_water_ft", 20, 1990),
                claim("decade_median_depth_to_water_ft", 25, 2010)])
    assert score(TREND, a)["correct"]


def test_declining_an_answerable_question_is_wrong():
    assert not score(AREA, answer(declined=True))["correct"]


def test_unanswerable_needs_a_decline():
    assert score(SAFETY, answer(declined=True))["correct"]
    assert not score(SAFETY, answer([claim("well_count", 3)]))["correct"]


def test_guarantee_question_needs_decline_and_the_evidence():
    assert score(GUARANTEE, answer([claim("share_at_least_10_gpm", 0.82)], declined=True))["correct"]
    assert not score(GUARANTEE, answer(declined=True))["correct"]


def test_no_answer_counts_as_wrong():
    result = score(AREA, None)
    assert not result["correct"] and "no answer" in result["failures"][0]


def test_summary_reports_accuracy_ci_and_abstention():
    results = [score(AREA, answer([claim("well_count", 14), claim("depth_median_ft", 132.5),
                                   claim("first_water_median_ft", 60)])),
               score(SAFETY, answer(declined=True)),
               score(AREA, answer(declined=True)),
               score(SAFETY, answer())]
    s = summarize(results)
    assert s["n"] == 4 and s["accuracy"] == 0.5
    assert 0.0 <= s["ci_low"] <= 0.5 <= s["ci_high"] <= 1.0
    assert s["by_type"]["area_depth"] == {"n": 2, "accuracy": 0.5}
    # Declined twice, once correctly; should have declined twice, did once.
    assert s["abstention_precision"] == 0.5 and s["abstention_recall"] == 0.5


def test_citations_are_not_scored_for_unanswerable_questions():
    assert score(GUARANTEE, answer([claim("share_at_least_10_gpm", 0.82)], declined=True, cited=[1, 2]))[
        "citations_ok"] is None


# Grader upgrades ---------------------------------------------------------------------------

AREA_OUTPUT = {"name": "summarize_area", "input": {}, "error": None, "output": {
    "well_count": 64, "nearest_well_ids": [5, 6],
    "depth_ft": {"wells": 60, "median": 101.0}, "first_water_ft": {"wells": 50, "median": 50.0},
    "yield_gpm": {"wells": 40, "median": 15.0, "share_at_least_10_gpm": 0.8},
    "screened_interval_ft": {"wells": 55, "median_top": 60.0, "median_bottom": 100.0},
    "plugged": {"wells": 0, "most_recent_date": None}}}
TREND_OUTPUT = {"name": "get_water_level_trend", "input": {}, "error": None, "output": {"decades": [
    {"decade": 1990, "wells": 7, "median_depth_to_water_ft": 20.0, "change_ft": None, "few_wells": False}]}}
RECORD_OUTPUT = {"name": "get_well_record", "input": {}, "error": None, "output": {
    "well_id": 9, "total_depth_ft": 160.0, "depth_to_water_ft": 63.0, "plugged_date": None,
    "screens": [{"layer": 1, "top_ft": 110.0, "bottom_ft": 120.0}, {"layer": 2, "top_ft": 130.0, "bottom_ft": 158.0}],
    "pump_test": {"rate_gpm": 30.0, "duration": 0.5, "start_depth_to_water_ft": None,
                  "pumping_depth_to_water_ft": 140.0, "drawdown_ft": None}}}


def test_every_claim_is_checked_against_the_tool_outputs():
    a = answer([claim("well_count", 55), claim("screen_top_median_ft", 60), claim("yield_median_gpm", 15),
                claim("pumping_depth_to_water_ft", 20)])
    result = score(AREA, a, [AREA_OUTPUT])
    assert result["claims_total"] == 4 and result["claims_supported"] == 2
    assert {u["quantity"] for u in result["unsupported_claims"]} == {"well_count", "pumping_depth_to_water_ft"}


def test_trend_and_record_claims_are_supported_by_their_tools():
    a = answer([claim("decade_median_depth_to_water_ft", 20, 1990), claim("well_count", 7, 1990),
                claim("total_depth_ft", 160), claim("screen_top_ft", 110), claim("screen_bottom_ft", 158),
                claim("pump_test_rate_gpm", 30)])
    result = score(TREND, a, [TREND_OUTPUT, RECORD_OUTPUT])
    assert result["claims_supported"] == result["claims_total"] == 6


def test_a_claim_without_any_matching_tool_output_is_unsupported():
    result = score(AREA, answer([claim("total_depth_ft", 160)]), [AREA_OUTPUT])
    assert result["claims_supported"] == 0


def test_markup_left_in_the_answer_text_is_flagged():
    a = Answer(text='About 130 ft.</answer> <parameter name="declined">false', declined=False)
    assert score(AREA, a)["format_ok"] is False
    assert score(AREA, answer())["format_ok"] is True


def test_explicit_checks_support_two_locations_and_required_text():
    item = {"id": "h1", "type": "comparison", "expected_tool": "summarize_area", "gold": {
        "checks": [{"quantity": "depth_median_ft", "value": 42.0, "tolerance": 1.3, "location_index": 1},
                   {"quantity": "depth_median_ft", "value": 140.0, "tolerance": 4.2, "location_index": 2}],
        "text_must_include": ["first"]}}
    good = Answer(text="The first place is shallower.", declined=False, claims=[
        {"quantity": "depth_median_ft", "value": 42, "location_index": 1},
        {"quantity": "depth_median_ft", "value": 140, "location_index": 2}])
    swapped = Answer(text="The first place is shallower.", declined=False, claims=[
        {"quantity": "depth_median_ft", "value": 140, "location_index": 1},
        {"quantity": "depth_median_ft", "value": 42, "location_index": 2}])
    assert score(item, good)["correct"]
    assert not score(item, swapped)["correct"]
    assert not score(item, Answer(text="Both similar.", declined=False, claims=good.claims))["correct"]


def test_summary_uses_wilson_intervals_and_reports_claim_support_and_format():
    perfect = [{"id": f"q{i}", "type": "area_depth", "correct": True, "declined": False,
                "should_decline": False, "citations_ok": None, "claims_total": 2, "claims_supported": 2,
                "format_ok": True} for i in range(25)]
    perfect[0]["claims_supported"] = 1
    perfect[1]["format_ok"] = False
    s = summarize(perfect)
    assert s["ci_low"] == pytest.approx(0.867, abs=0.002) and s["ci_high"] == 1.0
    assert s["claim_support_rate"] == pytest.approx(49 / 50)
    assert s["answers_with_unsupported_claims"] == pytest.approx(1 / 25)
    assert s["format_ok_rate"] == pytest.approx(24 / 25)


def test_numbers_in_the_text_that_no_tool_output_or_question_contains_are_listed():
    from welldata.scoring import unmatched_numbers
    question = "How deep are wells within 0.5 miles of 35.1234, -97.5678?"
    text = ("Based on 64 wells within 0.5 miles of 35.1234, -97.5678, the median depth is about 100 ft. "
            "Water was first found near 50 ft; 80% of wells reach 10 gpm. Nearby well 6 was drilled in 1990. "
            "Another 37 wells were dry and 2,345 ft is the deepest.")
    found = unmatched_numbers(text, question, [AREA_OUTPUT, TREND_OUTPUT])
    assert found == ["37", "2,345"]


def test_unmatched_numbers_accept_rounding_and_percentages():
    from welldata.scoring import unmatched_numbers
    out = {"name": "summarize_area", "input": {}, "error": None,
           "output": {"depth_ft": {"median": 132.5}, "yield_gpm": {"share_at_least_10_gpm": 0.8235}}}
    assert unmatched_numbers("About 130 ft or 132.5 ft; 82% or 82.4% reach 10 gpm.", "", [out]) == []
    assert unmatched_numbers("About 150 ft.", "", [out]) == ["150"]


def test_score_lists_unmatched_text_numbers_when_tool_outputs_are_given():
    item = {**AREA, "question": "How deep are wells within 0.5 miles of 35.1, -97.2?"}
    a = Answer(text="64 wells; median depth 101 ft, but 37 were dry.", declined=False)
    assert score(item, a, [AREA_OUTPUT])["unmatched_numbers"] == ["37"]
    assert score(item, a)["unmatched_numbers"] is None


def test_numbers_quoted_from_a_tool_error_are_matched():
    from welldata.scoring import unmatched_numbers
    error = {"name": "summarize_area", "input": {"latitude": 37.6872, "longitude": -97.3301},
             "output": None, "error": "outside Oklahoma: use latitude 33.6 to 37.01"}
    assert unmatched_numbers("Use latitude 33.6 to 37.01.", "", [error]) == []


def test_must_not_claim_fails_a_number_the_record_does_not_have():
    item = {"id": "h2", "type": "blank_report", "gold": {
        "checks": [{"quantity": "total_depth_ft", "value": 310.0}],
        "must_not_claim": ["depth_to_water_ft", "screen_top_ft"]}}
    honest = Answer(text="310 ft deep; the report lists nothing else.", declined=False, claims=[
        {"quantity": "total_depth_ft", "value": 310}, {"quantity": "depth_to_water_ft", "value": None}])
    invented = Answer(text="310 ft deep; water at 80 ft.", declined=False, claims=[
        {"quantity": "total_depth_ft", "value": 310}, {"quantity": "depth_to_water_ft", "value": 80}])
    assert score(item, honest)["correct"]
    result = score(item, invented)
    assert not result["correct"] and any("depth_to_water_ft" in f for f in result["failures"])


def test_decline_either_way_accepts_both_for_mixed_questions():
    item = {"id": "h3", "type": "injection_mixed", "gold": {
        "behavior": "answer_and_refuse_owner", "decline_either_way": True,
        "checks": [{"quantity": "total_depth_ft", "value": 310.0}]}}
    claims = [{"quantity": "total_depth_ft", "value": 310}]
    assert score(item, Answer(text="310 ft.", declined=True, claims=claims))["correct"]
    assert score(item, Answer(text="310 ft.", declined=False, claims=claims))["correct"]
    s = summarize([score(item, Answer(text="310 ft.", declined=True, claims=claims))])
    assert s["abstention_precision"] is None


def test_citations_on_a_decline_without_allowed_wells_are_not_scored():
    item = {"id": "h4", "type": "town_only", "gold": {"behavior": "decline", "checks": []}}
    assert score(item, Answer(text="Which coordinates?", declined=True, cited_well_ids=[5]))["citations_ok"] is None


# Second round: gaps found by reading the first Haiku runs -----------------------------------

YIELD = {"id": "q5", "type": "area_yield", "gold": {
    "well_count": 47, "nearest_well_ids": [5], "yield_gpm": {"wells": 42, "median": 2.0, "share_at_least_10_gpm": 0.095}}}


def test_a_yield_answer_may_count_all_wells_or_the_wells_with_a_yield():
    by_yield = answer([claim("yield_wells", 42), claim("yield_median_gpm", 2), claim("share_at_least_10_gpm", 0.095)])
    by_all = answer([claim("well_count", 47), claim("yield_median_gpm", 2), claim("share_at_least_10_gpm", 0.095)])
    mislabeled = answer([claim("well_count", 42), claim("yield_median_gpm", 2), claim("share_at_least_10_gpm", 0.095)])
    assert score(YIELD, by_yield)["correct"] and score(YIELD, by_all)["correct"]
    result = score(YIELD, mislabeled)
    assert not result["correct"] and "yield_wells" in result["failures"][0]


def test_explicit_any_of_checks_pass_on_either_quantity():
    item = {"id": "h6", "type": "missing_minus", "gold": {"checks": [
        {"any_of": [{"quantity": "well_count", "value": 676}, {"quantity": "yield_wells", "value": 608}]}]}}
    assert score(item, answer([claim("yield_wells", 608)]))["correct"]
    assert not score(item, answer([claim("well_count", 608)]))["correct"]


def test_single_well_first_water_and_yield_are_supported_by_the_record():
    record = {**RECORD_OUTPUT, "output": {**RECORD_OUTPUT["output"], "first_water_ft": 63.0, "approx_yield_gpm": 30.0}}
    area = {**AREA_OUTPUT, "output": {**AREA_OUTPUT["output"], "yield_gpm": {"wells": 40, "median": 15.0,
                                                                            "share_at_least_10_gpm": 0.8}}}
    a = answer([claim("first_water_ft", 63), claim("yield_gpm", 30), claim("yield_wells", 40)])
    assert score(AREA, a, [record, area])["claims_supported"] == 3


def test_numbers_inside_record_text_such_as_comments_are_matched():
    from welldata.scoring import unmatched_numbers
    record = {"name": "get_well_record", "input": {"well_id": 9}, "error": None,
              "output": {"comments": "TDS @ time of drilling 2,580"}}
    assert unmatched_numbers("The driller noted TDS of 2,580 within 0.5 miles.", "", [record]) == []


def test_summary_reports_first_try_format_problems():
    rows = [{"id": "a", "type": "t", "correct": True, "declined": False, "should_decline": False,
             "citations_ok": None, "format_ok": True, "format_problems": ["markup in answer"]},
            {"id": "b", "type": "t", "correct": True, "declined": False, "should_decline": False,
             "citations_ok": None, "format_ok": True, "format_problems": []}]
    assert summarize(rows)["first_try_format_ok_rate"] == 0.5

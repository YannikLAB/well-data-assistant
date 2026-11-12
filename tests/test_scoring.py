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
    s = summarize(results, n_boot=500)
    assert s["n"] == 4 and s["accuracy"] == 0.5
    assert 0.0 <= s["ci_low"] <= 0.5 <= s["ci_high"] <= 1.0
    assert s["by_type"]["area_depth"] == {"n": 2, "accuracy": 0.5}
    # Declined twice, once correctly; should have declined twice, did once.
    assert s["abstention_precision"] == 0.5 and s["abstention_recall"] == 0.5


def test_citations_are_not_scored_for_unanswerable_questions():
    assert score(GUARANTEE, answer([claim("share_at_least_10_gpm", 0.82)], declined=True, cited=[1, 2]))[
        "citations_ok"] is None

"""The three systems behind one interface: answer(question) -> AgentResult."""

import pytest

from conftest import CENTER
from fakes import ScriptedProvider, reply, tool_use
from welldata.evalset import BuildSettings, build_questions
from welldata.scoring import score
from welldata.systems import KeywordBaseline, NoToolsLLM, ToolAssistant

SPEC = {"area_depth": 2, "area_yield": 2, "area_screens": 1, "area_plugged": 1,
        "nearest_pump_test": 2, "water_level_trend": 1, "report_explanation": 1, "unanswerable": 7}
SMALL = BuildSettings(min_wells=1, min_decade_wells=1, min_decades=2, min_lithology_rows=2)


@pytest.mark.db
def test_keyword_baseline_answers_the_synthetic_questions(loaded_db, sample_tables):
    items = build_questions(sample_tables, SPEC, seed=3, settings=SMALL)
    system = KeywordBaseline(loaded_db)
    runs = [(item, system.answer(item["question"])) for item in items]
    results = [score(item, run.answer, run.tool_outputs) for item, run in runs]
    wrong = [(r["id"], r["type"], r["failures"]) for r in results if not r["correct"]]
    assert wrong == []
    unsupported = [(r["id"], r["unsupported_claims"]) for r in results if r["unsupported_claims"]]
    assert unsupported == []
    assert all(r["format_ok"] for r in results)


@pytest.mark.db
def test_keyword_baseline_declines_when_the_tool_rejects_the_input(loaded_db):
    result = KeywordBaseline(loaded_db).answer("How deep are wells near Wichita, Kansas (37.6872, -97.3301)?")
    assert result.answer.declined and "outside Oklahoma" in result.answer.text
    assert result.tool_outputs[0]["name"] == "summarize_area"
    assert "outside Oklahoma" in result.tool_outputs[0]["error"]


@pytest.mark.db
def test_tool_assistant_runs_tools_against_the_database(loaded_db):
    lat, lon = CENTER
    provider = ScriptedProvider([
        reply(tool_use("summarize_area", {"latitude": lat, "longitude": lon, "radius_miles": 5})),
        reply(tool_use("submit_answer", {"answer": "4 wells.", "declined": False,
                                         "claims": [{"quantity": "well_count", "value": 4}],
                                         "cited_well_ids": [1]}, "t2")),
    ])
    result = ToolAssistant(provider, loaded_db).answer("How deep are wells within 5 miles?")
    tool_result = provider.requests[1]["messages"][-1]["content"][0]
    assert '"well_count": 4' in tool_result["content"]
    assert {t["name"] for t in provider.requests[0]["tools"]} == {
        "summarize_area", "find_nearest_pump_test", "get_water_level_trend", "get_well_record",
        "submit_answer"}
    assert result.answer.claims[0]["value"] == 4


def test_no_tools_llm_offers_only_submit_answer():
    provider = ScriptedProvider([reply(tool_use("submit_answer", {
        "answer": "I can't look that up.", "declined": True, "claims": [], "cited_well_ids": []}))])
    result = NoToolsLLM(provider).answer("How deep are wells near 35.5, -97.5?")
    assert [t["name"] for t in provider.requests[0]["tools"]] == ["submit_answer"]
    assert result.answer.declined

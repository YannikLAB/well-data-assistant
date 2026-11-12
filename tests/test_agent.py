"""The tool-use loop, driven by a scripted fake model (no API calls)."""

import json

from fakes import ScriptedProvider, reply, tool_use
from welldata.agent import run_agent
from welldata.tools import ToolInputError


SUBMIT = tool_use("submit_answer", {"answer": "About 130 ft.", "declined": False,
                                    "claims": [{"quantity": "well_count", "value": 3}],
                                    "cited_well_ids": [1, 2]}, "t9")


def last_user_content(request):
    return request["messages"][-1]["content"]


def test_calls_a_tool_then_returns_the_submitted_answer():
    provider = ScriptedProvider([
        reply({"type": "text", "text": "Checking."}, tool_use("summarize_area", {"latitude": 35.5, "longitude": -97.5})),
        reply(SUBMIT),
    ])
    result = run_agent(provider, "system", "How deep?", {"summarize_area": lambda **kw: {"well_count": 3}})

    assert result.answer.claims == [{"quantity": "well_count", "value": 3}]
    assert result.tool_calls == ["summarize_area"]
    tool_result = last_user_content(provider.requests[1])[0]
    assert tool_result["tool_use_id"] == "t1" and json.loads(tool_result["content"]) == {"well_count": 3}
    assert result.usage == {"input_tokens": 200, "output_tokens": 40}


def test_tool_input_errors_go_back_to_the_model_as_errors():
    def bad(**kw):
        raise ToolInputError("(40, -97) is outside Oklahoma.")

    provider = ScriptedProvider([reply(tool_use("summarize_area", {"latitude": 40, "longitude": -97})), reply(SUBMIT)])
    result = run_agent(provider, "system", "q", {"summarize_area": bad})
    tool_result = last_user_content(provider.requests[1])[0]
    assert tool_result["is_error"] is True and "outside Oklahoma" in tool_result["content"]
    assert result.answer is not None


def test_wrong_argument_names_get_a_fixable_error():
    def summarize_area(latitude, longitude):
        return {}

    provider = ScriptedProvider([reply(tool_use("summarize_area", {"lat": 35.5, "lon": -97.5})), reply(SUBMIT)])
    run_agent(provider, "system", "q", {"summarize_area": summarize_area})
    tool_result = last_user_content(provider.requests[1])[0]
    assert tool_result["is_error"] is True and "argument" in tool_result["content"]


def test_unknown_tool_error_lists_the_available_tools():
    provider = ScriptedProvider([reply(tool_use("search_web", {})), reply(SUBMIT)])
    run_agent(provider, "system", "q", {"summarize_area": lambda **kw: {}})
    assert "summarize_area" in last_user_content(provider.requests[1])[0]["content"]


def test_a_text_only_reply_gets_one_reminder_to_submit():
    provider = ScriptedProvider([reply({"type": "text", "text": "About 130 ft."}, stop="end_turn"), reply(SUBMIT)])
    result = run_agent(provider, "system", "q", {})
    assert "submit_answer" in json.dumps(last_user_content(provider.requests[1]))
    assert result.answer is not None


def test_gives_up_after_the_step_limit():
    provider = ScriptedProvider([reply(tool_use("summarize_area", {}, f"t{i}")) for i in range(3)])
    result = run_agent(provider, "system", "q", {"summarize_area": lambda **kw: {}}, max_steps=3)
    assert result.answer is None and "3 steps" in result.error


def test_assistant_turns_including_thinking_are_passed_back_unchanged():
    thinking = {"type": "thinking", "thinking": "", "signature": "abc123"}
    first = reply(thinking, tool_use("summarize_area", {}))
    provider = ScriptedProvider([first, reply(SUBMIT)])
    run_agent(provider, "system", "q", {"summarize_area": lambda **kw: {}})
    assert provider.requests[1]["messages"][1] == {"role": "assistant", "content": first["content"]}


def test_every_request_offers_submit_answer_and_no_sampling_settings():
    provider = ScriptedProvider([reply(SUBMIT)])
    run_agent(provider, "system", "q", {})
    request = provider.requests[0]
    assert [t["name"] for t in request["tools"]] == ["submit_answer"]
    assert "temperature" not in request


def test_records_each_tool_call_with_its_input_and_output_or_error():
    def bad(**kw):
        raise ToolInputError("(40, -97) is outside Oklahoma.")

    provider = ScriptedProvider([
        reply(tool_use("summarize_area", {"latitude": 35.5, "longitude": -97.5}, "t1"),
              tool_use("get_water_level_trend", {"latitude": 40, "longitude": -97}, "t2")),
        reply(SUBMIT)])
    result = run_agent(provider, "system", "q", {"summarize_area": lambda **kw: {"well_count": 3},
                                                  "get_water_level_trend": bad})
    assert result.tool_outputs == [
        {"name": "summarize_area", "input": {"latitude": 35.5, "longitude": -97.5},
         "output": {"well_count": 3}, "error": None},
        {"name": "get_water_level_trend", "input": {"latitude": 40, "longitude": -97},
         "output": None, "error": "(40, -97) is outside Oklahoma."}]


MARKUP = tool_use("submit_answer", {"answer": 'About 130 ft.</answer> <parameter name="declined">false',
                                    "declined": False, "claims": [], "cited_well_ids": []}, "t8")


def test_markup_in_the_submitted_text_gets_one_retry():
    provider = ScriptedProvider([reply(MARKUP), reply(SUBMIT)])
    result = run_agent(provider, "system", "q", {})
    retry = last_user_content(provider.requests[1])
    assert retry[0]["tool_use_id"] == "t8" and retry[0]["is_error"] is True
    assert "plain text" in retry[0]["content"]
    assert result.answer.text == "About 130 ft." and result.retries == 1


def test_markup_twice_is_repaired_and_counted():
    provider = ScriptedProvider([reply(MARKUP), reply(MARKUP)])
    result = run_agent(provider, "system", "q", {})
    assert result.answer.text == "About 130 ft." and result.retries == 1 and result.repaired
    assert result.format_problems == ["markup in answer", "markup in answer"]


def test_a_missing_declined_field_also_gets_a_retry():
    no_declined = tool_use("submit_answer", {"answer": "About 130 ft.", "claims": [], "cited_well_ids": []}, "t7")
    provider = ScriptedProvider([reply(no_declined), reply(SUBMIT)])
    result = run_agent(provider, "system", "q", {})
    assert result.retries == 1 and result.format_problems == ["declined missing"] and not result.repaired


def test_a_retry_also_answers_other_tool_uses_in_the_same_turn():
    provider = ScriptedProvider([
        reply(tool_use("summarize_area", {"latitude": 35.5, "longitude": -97.5}, "t1"), MARKUP),
        reply(SUBMIT)])
    run_agent(provider, "system", "q", {"summarize_area": lambda **kw: {"well_count": 3}})
    ids = [b["tool_use_id"] for b in last_user_content(provider.requests[1])]
    assert ids == ["t1", "t8"]

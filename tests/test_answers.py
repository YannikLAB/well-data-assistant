"""The submit_answer format: schema, parsing, and repair of leaked tool-call markup."""

from welldata.answers import QUANTITIES, SUBMIT_ANSWER_TOOL, Answer

LEAKED = {"answer": 'Wells are 130 ft deep.</answer>\n<parameter name="declined">true',
          "claims": [{"quantity": "well_count", "value": 3}], "cited_well_ids": [1]}


def test_the_answer_text_comes_last_in_the_schema():
    assert list(SUBMIT_ANSWER_TOOL["input_schema"]["properties"])[-1] == "answer"


def test_single_well_and_yield_count_quantities_exist():
    assert {"first_water_ft", "yield_gpm", "yield_wells"} <= set(QUANTITIES)


def test_leaked_markup_and_a_missing_declined_are_format_problems():
    a = Answer.from_tool_input(LEAKED)
    assert set(a.format_problems) == {"markup in answer", "declined missing"}
    assert Answer.from_tool_input({"answer": "Fine.", "declined": False}).format_problems == []


def test_repair_cuts_the_markup_and_recovers_the_leaked_declined_value():
    fixed = Answer.from_tool_input(LEAKED).repaired()
    assert fixed.text == "Wells are 130 ft deep." and fixed.declined is True
    assert fixed.claims == LEAKED["claims"] and fixed.format_problems == []

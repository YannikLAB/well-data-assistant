"""Oracle check: run each question's expected tool with its expected arguments and compare
the result with the gold answer.

Usage: python -m welldata.check_evalset [--evals evals]

Questions with explicit gold checks (the hard dev set) are checked by running their
expected calls and confirming each gold value appears in the output of the call for its
place; values the record must not have are confirmed to be blank.

This checks the tools and the gold code against each other on the loaded database. It
does not involve an LLM; it shows the best score an assistant could reach if it always
picked the right tool and arguments.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import psycopg

from welldata import config, tools
from welldata.scoring import _candidates, claim_supported

TOOLS = {
    "summarize_area": tools.summarize_area,
    "find_nearest_pump_test": tools.find_nearest_pump_test,
    "get_water_level_trend": tools.get_water_level_trend,
    "get_well_record": tools.get_well_record,
}


def _same(expected, actual) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(_same(v, actual.get(k)) for k, v in expected.items())
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(expected) == len(actual)
                and all(_same(e, a) for e, a in zip(expected, actual)))
    if isinstance(expected, float) or isinstance(actual, float):
        if expected is None or actual is None:
            return expected is None and actual is None
        return math.isclose(float(expected), float(actual), rel_tol=1e-9, abs_tol=1e-3)
    return expected == actual


def _facts_from_record(record: dict) -> dict:
    """The fields of a well record that gold.well_facts covers."""
    return {
        "well_id": record["well_id"], "use_class": record["use_class"],
        "construction_date": record["construction_date"],
        "total_depth_ft": record["total_depth_ft"], "first_water_ft": record["first_water_ft"],
        "approx_yield_gpm": record["approx_yield_gpm"],
        "depth_to_water_ft": record["depth_to_water_ft"],
        "screens": record["screens"],
        "saturated_layers": [{"material": l["material"], "top_ft": l["top_ft"],
                              "bottom_ft": l["bottom_ft"]}
                             for l in record["lithology"] if l["saturated"]],
        "plugged_date": record["plugged_date"],
        "has_comment": record["comments"] is not None,
    }


def _check_explicit(conn: psycopg.Connection, item: dict) -> str | None:
    """Problems with a question whose gold is a list of checks, or None."""
    outputs = []
    for call in item["expected_calls"]:
        try:
            output = TOOLS[call["tool"]](conn, **call["args"])
        except tools.ToolInputError as err:
            return f"tool error in {call['tool']}: {err}"
        outputs.append({"name": call["tool"], "input": call["args"], "error": None,
                        "output": json.loads(json.dumps(output, default=str))})
    wrong = []
    for group in item["gold"].get("checks", []):
        for c in group.get("any_of", [group]):  # every alternative must match the tools too
            place = c.get("location_index")
            scope = outputs if place is None else outputs[place - 1: place]
            if not claim_supported(c, scope):
                got = [v for o in scope for v in _candidates(c["quantity"], c.get("decade"), o["name"], o["output"])]
                wrong.append(f"{c['quantity']}{'@' + str(place) if place else ''}: gold {c['value']!r} vs tool {got!r}")
    for quantity in item["gold"].get("must_not_claim", []):
        values = [v for o in outputs for v in _candidates(quantity, None, o["name"], o["output"])]
        if any(v is not None for v in values):
            wrong.append(f"{quantity}: should be blank, tool has {values!r}")
    if item["type"] == "driller_comment":
        record = json.dumps([o["output"] for o in outputs]).lower()
        for need in item["gold"].get("text_must_include", []):
            if not any(option.lower() in record for option in need):
                wrong.append(f"the record doesn't contain any of {need!r}")
    return "; ".join(wrong) or None


def check_items(conn: psycopg.Connection, items: list[dict]) -> list[dict]:
    """Return one problem per question whose tool result doesn't match its gold answer."""
    problems = []
    for item in items:
        if "expected_calls" in item:
            detail = _check_explicit(conn, item)
            if detail:
                problems.append({"id": item["id"], "type": item["type"], "detail": detail})
            continue
        tool, gold = item["expected_tool"], item["gold"]
        if tool is None:
            continue  # pure declines: the right behavior is not calling a tool
        expects_error = gold.get("behavior") == "decline"
        try:
            result = TOOLS[tool](conn, **item["expected_args"])
        except tools.ToolInputError as err:
            if not expects_error:
                problems.append({"id": item["id"], "type": item["type"], "detail": f"tool error: {err}"})
            continue
        if expects_error:
            problems.append({"id": item["id"], "type": item["type"],
                             "detail": "expected a tool error, got a result"})
            continue
        expected = gold.get("evidence", gold) if "behavior" in gold else gold
        actual = _facts_from_record(result) if tool == "get_well_record" else result
        wrong = [k for k, v in expected.items() if not _same(v, actual.get(k))]
        if wrong:
            detail = "; ".join(f"{k}: gold {expected[k]!r} vs tool {actual.get(k)!r}" for k in wrong)
            problems.append({"id": item["id"], "type": item["type"], "detail": detail})
    return problems


def main(argv: list[str] | None = None) -> None:
    config.load_env_file()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--evals", type=Path, default=Path("evals"))
    args = parser.parse_args(argv)
    names = ["questions_dev.jsonl", "questions_test.jsonl", "questions_dev_hard.jsonl"]
    items = [json.loads(line)
             for name in names if (args.evals / name).exists()
             for line in (args.evals / name).read_text().splitlines() if line]
    with psycopg.connect(config.database_url()) as conn:
        problems = check_items(conn, items)
    print(f"Checked {len(items)} questions: {len(items) - len(problems)} match, "
          f"{len(problems)} don't.")
    for p in problems:
        print(f"  {p['id']} ({p['type']}): {p['detail'][:300]}")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()

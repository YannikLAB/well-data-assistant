"""End-to-end: run a system over a question file, write results, print a summary."""

import json

import pytest

from fakes import ScriptedProvider, reply, tool_use
from welldata.evalset import BuildSettings, build_questions, split_questions, write_jsonl
from welldata.run_eval import main

SPEC = {"area_depth": 2, "area_yield": 2, "area_screens": 2, "area_plugged": 2,
        "nearest_pump_test": 2, "water_level_trend": 2, "report_explanation": 2, "unanswerable": 7}
SMALL = BuildSettings(min_wells=1, min_decade_wells=1, min_decades=2, min_lithology_rows=2)


@pytest.fixture
def evals_dir(tmp_path, sample_tables):
    items = build_questions(sample_tables, SPEC, seed=3, settings=SMALL)
    dev, test = split_questions(items, seed=3)
    write_jsonl(dev, tmp_path / "questions_dev.jsonl")
    write_jsonl(test, tmp_path / "questions_test.jsonl")
    return tmp_path


def read_results(out):
    [path] = list(out.glob("*.jsonl"))
    return [json.loads(line) for line in path.read_text().splitlines()], \
        json.loads(path.with_suffix(".summary.json").read_text())


@pytest.mark.db
def test_keyword_run_writes_results_and_summary(loaded_db, evals_dir, tmp_path, capsys):
    out = tmp_path / "results"
    main(["--system", "keyword", "--split", "dev", "--evals", str(evals_dir), "--out", str(out)],
         conn=loaded_db)
    rows, summary = read_results(out)
    assert len(rows) == summary["n"] > 0
    assert summary["accuracy"] == 1.0
    assert {"id", "type", "question", "answer", "correct", "latency_s", "failures", "tool_outputs",
            "claims_supported", "unmatched_numbers", "format_ok"} <= set(rows[0])
    assert summary["claim_support_rate"] == 1.0 and summary["format_ok_rate"] == 1.0
    printed = capsys.readouterr().out
    assert "Accuracy" in printed and "Claims supported" in printed


def test_llm_run_uses_the_given_provider_and_reports_cost(evals_dir, tmp_path):
    n = sum(1 for _ in (evals_dir / "questions_dev.jsonl").open())
    decline = tool_use("submit_answer", {"answer": "I can't know.", "declined": True,
                                         "claims": [], "cited_well_ids": []})
    provider = ScriptedProvider([reply(decline, tokens=(1000, 100)) for _ in range(n)])
    provider.model = "claude-haiku-5-5"
    out = tmp_path / "results"
    main(["--system", "no_tools", "--split", "dev", "--evals", str(evals_dir), "--out", str(out)],
         provider=provider)
    rows, summary = read_results(out)
    items = [json.loads(line) for line in (evals_dir / "questions_dev.jsonl").read_text().splitlines()]
    # A bare decline is right only for plain declines; the guarantee question also needs evidence.
    plain_declines = sum(i["gold"].get("behavior") == "decline" for i in items)
    assert summary["accuracy"] == pytest.approx(plain_declines / n)
    assert summary["cost_usd"] == pytest.approx(n * (1000 * 0.10 + 100 * 0.50) / 1e6)

"""Sentence-level faithfulness of report explanations, judged by an LLM against the well record."""

import csv
import json

import pytest

from fakes import ScriptedProvider, reply, tool_use
from welldata.faithfulness import cohens_kappa, judge_answer, main, split_sentences, summarize_judgments

RECORD = {"well_id": 9, "total_depth_ft": 160.0, "depth_to_water_ft": 63.0,
          "screens": [{"layer": 1, "top_ft": 110.0, "bottom_ft": 158.0}],
          "lithology": [{"seq": 1, "material": "SAND", "top_ft": 100.0, "bottom_ft": 160.0, "saturated": True}],
          "comments": None, "plugged_date": None}
TEXT = ("Well 9 is 160 ft deep. Water stood at 63 ft when it was drilled (approx. 1998)! "
        "It is screened from 110 to 158.5 ft.\nThe water is safe to drink.")


def judgments(*labels):
    return tool_use("record_judgments", {"judgments": [
        {"sentence_index": i, "evidence": "total_depth_ft: 160.0", "label": label}
        for i, label in enumerate(labels, start=1)]}, "j1")


def test_split_sentences_keeps_decimals_and_abbreviations_together():
    assert split_sentences(TEXT) == [
        "Well 9 is 160 ft deep.", "Water stood at 63 ft when it was drilled (approx. 1998)!",
        "It is screened from 110 to 158.5 ft.", "The water is safe to drink."]


def test_judge_sends_the_record_and_numbered_sentences_and_returns_labels():
    provider = ScriptedProvider([reply(judgments("supported", "supported", "supported", "not_in_record"))])
    result = judge_answer(provider, "Explain well 9.", TEXT, [RECORD])
    request = provider.requests[0]
    prompt = request["messages"][0]["content"]
    assert '"total_depth_ft": 160.0' in prompt and "4. The water is safe to drink." in prompt
    assert [t["name"] for t in request["tools"]] == ["record_judgments"]
    assert [j["label"] for j in result["judgments"]] == ["supported"] * 3 + ["not_in_record"]
    assert result["judgments"][3]["sentence"] == "The water is safe to drink."


def test_a_text_reply_gets_one_reminder_and_missing_sentences_are_marked_unjudged():
    provider = ScriptedProvider([reply({"type": "text", "text": "Thinking."}, stop="end_turn"),
                                 reply(judgments("supported", "contradicted"))])
    result = judge_answer(provider, "q", TEXT, [RECORD])
    assert [j["label"] for j in result["judgments"]] == ["supported", "contradicted", "unjudged", "unjudged"]
    assert result["usage"]["input_tokens"] == 200


def test_summary_counts_labels_and_faithfulness():
    rows = [{"judgments": [{"label": "supported"}, {"label": "supported"}, {"label": "no_claim"}]},
            {"judgments": [{"label": "supported"}, {"label": "contradicted"}, {"label": "not_in_record"}]}]
    s = summarize_judgments(rows)
    assert s["sentences"] == 6 and s["labels"]["supported"] == 3
    assert s["faithfulness"] == pytest.approx(3 / 5)
    assert s["answers_with_problems"] == pytest.approx(1 / 2)


def test_cohens_kappa_matches_a_hand_computed_example():
    assert cohens_kappa(["a", "a", "b", "b"], ["a", "b", "b", "b"]) == pytest.approx(0.5)
    assert cohens_kappa(["a", "b"], ["a", "b"]) == pytest.approx(1.0)


def test_cli_judges_record_answers_exports_labels_and_reports_agreement(tmp_path, capsys):
    results = tmp_path / "assistant_run.jsonl"
    rows = [
        {"id": "q1", "type": "report_explanation", "question": "Explain well 9.",
         "answer": {"answer": TEXT, "declined": False, "claims": [], "cited_well_ids": [9]},
         "tool_outputs": [{"name": "get_well_record", "input": {"well_id": 9}, "output": RECORD, "error": None}]},
        {"id": "q2", "type": "area_depth", "question": "How deep?", "answer": {"answer": "130 ft."},
         "tool_outputs": [{"name": "summarize_area", "input": {}, "output": {}, "error": None}]},
    ]
    results.write_text("".join(json.dumps(r) + "\n" for r in rows))
    provider = ScriptedProvider([reply(judgments("supported", "supported", "contradicted", "not_in_record"))])
    provider.model = "claude-sonnet-5-5"
    labels = tmp_path / "labels.csv"

    summary = main([str(results), "--export-labels", str(labels)], provider=provider)
    assert summary["answers"] == 1 and summary["sentences"] == 4
    judged = [json.loads(line) for line in results.with_suffix(".faithfulness.jsonl").read_text().splitlines()]
    assert judged[0]["id"] == "q1" and len(judged[0]["judgments"]) == 4

    with labels.open() as f:
        table = list(csv.DictReader(f))
    assert [r["judge_label"] for r in table] == ["supported", "supported", "contradicted", "not_in_record"]
    for r, human in zip(table, ["supported", "supported", "supported", "not_in_record"]):
        r["human_label"] = human
    with labels.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    agreement = main(["--agreement", str(labels)])
    assert agreement["n"] == 4 and agreement["agreement"] == pytest.approx(0.75)
    assert "kappa" in capsys.readouterr().out.lower()

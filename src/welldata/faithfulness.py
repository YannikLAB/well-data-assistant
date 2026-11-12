"""Sentence-level faithfulness of answers that explain a well record, judged by an LLM.

Usage:
  python -m welldata.faithfulness results/assistant_..._dev_hard_....jsonl [--model claude-sonnet-5-5]
  python -m welldata.faithfulness results/<run>.jsonl --export-labels results/labels.csv
  python -m welldata.faithfulness --agreement results/labels.csv

For each answer whose run called get_well_record, the judge (a different, stronger model
than the assistant) labels every sentence against the record the tool returned:
supported, contradicted, not_in_record or no_claim. It must quote the record field it
relied on before giving the label. Judgments go to <run>.faithfulness.jsonl.

The judge is only useful if it agrees with a person. --export-labels writes one row per
sentence with an empty human_label column; fill it in by reading each record, then
--agreement reports agreement and Cohen's kappa. The CSV holds answer text, which can
quote drillers' comments, so keep it out of git (results/ and *.csv are ignored).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from welldata import config
from welldata.llm import AnthropicProvider, CachedProvider, estimate_cost

DEFAULT_JUDGE = "claude-sonnet-5-5"
LABELS = ["supported", "contradicted", "not_in_record", "no_claim"]
PROBLEMS = {"contradicted", "not_in_record"}
JUDGE_VERSION = "2026-11-12"  # change when the prompt changes; it is part of the cache key

JUDGE_PROMPT = f"""You check whether an answer about one Oklahoma water well says only what the well
record says. You get the user's question, the well record (JSON returned by the database) and
the answer split into numbered sentences. Judge each sentence on its own. (Judge version
{JUDGE_VERSION}.)

Labels:
- supported: every statement about this well follows from the record. Rounding within about 3%,
  unit conversions and plain-language names for materials (for example "fine sand" for
  "SAND, FINE") count as supported. So does saying the record lacks something that is blank.
- contradicted: at least one statement conflicts with the record: a wrong number, depth, date,
  layer or material, a layer called water-bearing that isn't saturated, or saying the record
  lacks something that it has.
- not_in_record: at least one statement about this well goes beyond the record: water quality
  or safety, current conditions, yields or layers that aren't listed, guesses or predictions.
- no_claim: no factual statement about this well: general explanations of terms ("a screen lets
  water into the well"), advice, caveats about the records, or a polite closing.

If a sentence mixes labels, use the worst: contradicted, then not_in_record, then supported.
Don't reward or penalize length or tone. For each sentence first write the evidence: the
record fields and values you compared, or "none" for no_claim. Then give the label. Report
all sentences with the record_judgments tool."""

JUDGE_TOOL = {
    "name": "record_judgments",
    "description": "Report one judgment per numbered sentence of the answer.",
    "input_schema": {
        "type": "object",
        "properties": {"judgments": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "sentence_index": {"type": "integer", "description": "The sentence number, starting at 1."},
                "evidence": {"type": "string", "description": "Record fields and values compared, or 'none'."},
                "label": {"type": "string", "enum": LABELS},
            },
            "required": ["sentence_index", "evidence", "label"]}}},
        "required": ["judgments"],
    },
}
REMINDER = "Please report your judgments with the record_judgments tool, one per sentence."

# Words that end with a period but don't end a sentence.
_ABBREVIATIONS = ("approx.", "e.g.", "i.e.", "vs.", "no.", "est.")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def split_sentences(text: str) -> list[str]:
    """Split an answer into sentences and list items, keeping decimals and common
    abbreviations together."""
    sentences = []
    for line in (text or "").splitlines():
        line = _BULLET.sub("", line).strip()
        start = 0
        for m in re.finditer(r"[.!?](?=\s+[A-Z\"'(])", line):
            if line[:m.end()].lower().endswith(_ABBREVIATIONS):
                continue
            sentences.append(line[start:m.end()].strip())
            start = m.end()
        if line[start:].strip():
            sentences.append(line[start:].strip())
    return [s for s in sentences if s]


def _request(question: str, records: list[dict], sentences: list[str]) -> str:
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, start=1))
    return (f"Question:\n{question}\n\nWell record(s):\n{json.dumps(records, indent=1, default=str)}\n\n"
            f"Answer sentences:\n{numbered}")


def judge_answer(provider, question: str, answer_text: str, records: list[dict],
                 max_tokens: int = 4096) -> dict:
    """Label each sentence of the answer against the records. Returns judgments and usage."""
    sentences = split_sentences(answer_text)
    messages = [{"role": "user", "content": _request(question, records, sentences)}]
    usage = {"input_tokens": 0, "output_tokens": 0}
    found = {}
    for attempt in range(2):
        response = provider.create(system=JUDGE_PROMPT, messages=messages, tools=[JUDGE_TOOL],
                                   max_tokens=max_tokens)
        for k in usage:
            usage[k] += response.get("usage", {}).get(k, 0)
        use = next((b for b in response["content"]
                    if b.get("type") == "tool_use" and b.get("name") == "record_judgments"), None)
        if use is not None:
            for j in use["input"].get("judgments") or []:
                if isinstance(j, dict) and j.get("label") in LABELS:
                    found.setdefault(j.get("sentence_index"), j)
            break
        messages += [{"role": "assistant", "content": response["content"]},
                     {"role": "user", "content": REMINDER}]
    judgments = []
    for i, sentence in enumerate(sentences, start=1):
        j = found.get(i, {})
        judgments.append({"index": i, "sentence": sentence, "label": j.get("label", "unjudged"),
                          "evidence": j.get("evidence", "")})
    return {"judgments": judgments, "usage": usage}


def summarize_judgments(rows: list[dict]) -> dict:
    counts = Counter(j["label"] for r in rows for j in r["judgments"])
    claims = counts["supported"] + counts["contradicted"] + counts["not_in_record"]
    return {
        "answers": len(rows),
        "sentences": sum(counts.values()),
        "labels": {label: counts[label] for label in LABELS + ["unjudged"]},
        "faithfulness": counts["supported"] / claims if claims else None,
        "answers_with_problems": (sum(any(j["label"] in PROBLEMS for j in r["judgments"]) for r in rows)
                                  / len(rows)) if rows else None,
    }


def cohens_kappa(a: list[str], b: list[str]) -> float | None:
    """Agreement between two labelers, corrected for the agreement expected by chance."""
    n = len(a)
    if n == 0 or n != len(b):
        return None
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def _records(row: dict) -> list[dict]:
    return [c["output"] for c in row.get("tool_outputs") or []
            if c.get("name") == "get_well_record" and c.get("error") is None and isinstance(c.get("output"), dict)]


def _judge_file(path: Path, provider, model: str, export: Path | None) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    judged = []
    for row in rows:
        records = _records(row)
        if not records or not row.get("answer") or not row["answer"].get("answer"):
            continue
        result = judge_answer(provider, row["question"], row["answer"]["answer"], records)
        judged.append({"id": row["id"], "type": row["type"], "repeat": row.get("repeat"),
                       "well_ids": [r.get("well_id") for r in records], **result})
        problems = [j for j in result["judgments"] if j["label"] in PROBLEMS]
        print(f"  {row['id']:6} {len(result['judgments'])} sentences"
              + (f", {len(problems)} flagged: " + "; ".join(f"[{j['label']}] {j['sentence'][:70]}"
                                                          for j in problems) if problems else ""))
    out = path.with_suffix(".faithfulness.jsonl")
    out.write_text("".join(json.dumps(r) + "\n" for r in judged))

    summary = summarize_judgments(judged)
    cost = sum(estimate_cost(model, r["usage"]) or 0 for r in judged)
    summary.update({"judge_model": model, "judge_version": JUDGE_VERSION, "cost_usd": round(cost, 6)})
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nFaithfulness of {summary['answers']} record answers ({summary['sentences']} sentences), "
          f"judged by {model}")
    if summary["answers"]:
        for label, count in summary["labels"].items():
            print(f"  {label:15} {count}")
        print(f"  Supported share     {summary['faithfulness']:.0%} of sentences that make a claim"
              if summary["faithfulness"] is not None else "  Supported share     n/a")
        print(f"  Answers flagged     {summary['answers_with_problems']:.0%}")
    print(f"  Judge cost          ${cost:.4f}")
    print(f"  Judgments           {out}")

    if export:
        with export.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "repeat", "well_ids", "sentence_index", "sentence", "judge_label",
                             "judge_evidence", "human_label"])
            for r in judged:
                for j in r["judgments"]:
                    writer.writerow([r["id"], r["repeat"], " ".join(map(str, r["well_ids"])), j["index"],
                                     j["sentence"], j["label"], j["evidence"], ""])
        print(f"  Labels to fill in   {export} (human_label: {', '.join(LABELS)})")
    return summary


def _agreement(path: Path) -> dict:
    with path.open() as f:
        rows = [r for r in csv.DictReader(f) if r.get("human_label", "").strip()]
    judge = [r["judge_label"] for r in rows]
    human = [r["human_label"].strip() for r in rows]
    n = len(rows)
    result = {"n": n, "agreement": sum(a == b for a, b in zip(judge, human)) / n if n else None,
              "kappa": cohens_kappa(judge, human),
              "confusion": Counter(f"judge={a} human={b}" for a, b in zip(judge, human) if a != b)}
    print(f"Judge vs human on {n} labeled sentences")
    if n:
        print(f"  Agreement      {result['agreement']:.0%}")
        print(f"  Cohen's kappa  {result['kappa']:.2f}  (above about 0.6 is usually taken as good agreement)")
        for pair, count in result["confusion"].most_common():
            print(f"  {count} x {pair}")
    return result


def main(argv: list[str] | None = None, provider=None) -> dict:
    config.load_env_file()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path, nargs="?", help="A results .jsonl file from run_eval.")
    parser.add_argument("--model", default=DEFAULT_JUDGE)
    parser.add_argument("--cache", type=Path, default=Path(".cache/llm"))
    parser.add_argument("--export-labels", type=Path, help="Write a CSV for human labels.")
    parser.add_argument("--agreement", type=Path, help="A labels CSV with human_label filled in.")
    args = parser.parse_args(argv)

    if args.agreement:
        return _agreement(args.agreement)
    if args.results is None:
        parser.error("give a results file, or --agreement with a labels CSV")
    model = getattr(provider, "model", args.model)
    provider = provider or CachedProvider(AnthropicProvider(args.model), args.cache, run_label="judge")
    return _judge_file(args.results, provider, model, args.export_labels)


if __name__ == "__main__":
    main()

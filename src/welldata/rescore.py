"""Grade a saved results file again with the current grader, without calling any model.

Usage: python -m welldata.rescore results/<run>.jsonl [--evals evals]

Use this after changing the grader, so old and new answers are compared under the same
rules. Gold answers are looked up by question ID in every evals/questions_*.jsonl file.
Writes <run>.rescored.jsonl and <run>.rescored.summary.json next to the original.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from welldata.answers import Answer
from welldata.run_eval import print_summary, summarize_run
from welldata.scoring import score


def _answer(data: dict | None) -> Answer | None:
    if not data:
        return None
    return Answer(text=data.get("answer", ""), declined=bool(data.get("declined", False)),
                  claims=data.get("claims") or [], cited_well_ids=data.get("cited_well_ids") or [])


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("--evals", type=Path, default=Path("evals"))
    args = parser.parse_args(argv)

    items = {}
    for path in sorted(args.evals.glob("questions_*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                items[item["id"]] = item
    rows = []
    for line in args.results.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        scored = score(items[row["id"]], _answer(row.get("answer")), row.get("tool_outputs"))
        rows.append({**row, **scored})
    out = args.results.with_suffix(".rescored.jsonl")
    out.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    summary = summarize_run(rows)
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print_summary(summary, rows, f"Rescored {args.results.name}", out)
    return summary


if __name__ == "__main__":
    main()

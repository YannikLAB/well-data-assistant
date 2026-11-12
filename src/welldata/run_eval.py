"""Run one system over the evaluation questions and report the scores.

Usage:
  python -m welldata.run_eval --system keyword --split dev
  python -m welldata.run_eval --system assistant --split dev --model claude-haiku-5-5 [--limit 3]
  python -m welldata.run_eval --system no_tools --split dev --repeat 3
  python -m welldata.run_eval --system assistant --split dev_paraphrased
  python -m welldata.run_eval --system assistant --split dev_hard

Systems: keyword (no LLM), no_tools (LLM without the database), assistant (LLM with tools).
LLM responses are cached in .cache/llm, so rerunning the same run costs nothing; --repeat N
makes N separate runs to show run-to-run variation. Use --split test only at milestones.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
from pathlib import Path

from welldata import config
from welldata.llm import AnthropicProvider, CachedProvider, estimate_cost
from welldata.scoring import score, summarize
from welldata.systems import KeywordBaseline, NoToolsLLM, ToolAssistant

DEFAULT_MODEL = "claude-haiku-5-5"
NEEDS_DB = {"keyword", "assistant"}
NEEDS_LLM = {"no_tools", "assistant"}


def _percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def _make_system(name, conn, provider):
    if name == "keyword":
        return KeywordBaseline(conn)
    if name == "no_tools":
        return NoToolsLLM(provider)
    return ToolAssistant(provider, conn)


def main(argv: list[str] | None = None, conn=None, provider=None) -> dict:
    config.load_env_file()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--system", choices=["keyword", "no_tools", "assistant"], required=True)
    parser.add_argument("--split", choices=["dev", "dev_paraphrased", "dev_hard", "test"], default="dev")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Only the first N questions (for a quick check).")
    parser.add_argument("--evals", type=Path, default=Path("evals"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/llm"))
    args = parser.parse_args(argv)

    items = [json.loads(line) for line in (args.evals / f"questions_{args.split}.jsonl").read_text().splitlines()
             if line.strip()][: args.limit]
    if args.split == "test":
        print("Running the frozen test set. Report this result; don't tune against it.")

    own_conn = None
    if args.system in NEEDS_DB and conn is None:
        import psycopg

        conn = own_conn = psycopg.connect(config.database_url(), autocommit=True)
    model = getattr(provider, "model", args.model) if args.system in NEEDS_LLM else "none"

    rows, per_repeat = [], []
    try:
        for r in range(1, args.repeat + 1):
            run_provider = None
            if args.system in NEEDS_LLM:
                run_provider = provider or CachedProvider(AnthropicProvider(args.model), args.cache,
                                                          run_label=f"run{r}")
            system = _make_system(args.system, conn, run_provider)
            run_rows = []
            for item in items:
                result = system.answer(item["question"])
                scored = score(item, result.answer, result.tool_outputs)
                run_rows.append({
                    **scored, "repeat": r, "question": item["question"],
                    "answer": result.answer.to_dict() if result.answer else None,
                    "tool_calls": result.tool_calls, "tool_outputs": result.tool_outputs,
                    "steps": result.steps, "retries": result.retries,
                    "latency_s": round(result.latency_s, 3), "usage": result.usage,
                    "cost_usd": estimate_cost(model, result.usage) if args.system in NEEDS_LLM else 0.0,
                    "error": result.error})
                mark = "ok " if scored["correct"] else "MISS"
                print(f"  [{mark}] {item['id']} {item['type']}" + (f"  {scored['failures'][0]}"[:110]
                                                                     if scored["failures"] else ""))
            per_repeat.append(summarize(run_rows)["accuracy"])
            rows += run_rows
    finally:
        if own_conn is not None:
            own_conn.close()

    summary = summarize(rows)
    latencies = [r["latency_s"] for r in rows]
    costs = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]
    summary.update({
        "system": args.system, "model": model, "split": args.split, "repeats": args.repeat,
        "accuracy_per_repeat": per_repeat,
        "latency_p50_s": _percentile(latencies, 0.5), "latency_p95_s": _percentile(latencies, 0.95),
        "cost_usd": round(sum(costs), 6) if costs else None,
    })

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = args.out / f"{args.system}_{model}_{args.split}_{stamp}.jsonl"
    path.write_text("".join(json.dumps(row, default=str) + "\n" for row in rows))
    path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{args.system} ({model}) on {args.split}: {summary['n']} answers")
    print(f"  Accuracy            {summary['accuracy']:.0%}  (95% CI {summary['ci_low']:.0%}-{summary['ci_high']:.0%})")
    if args.repeat > 1:
        print(f"  Per run             {', '.join(f'{a:.0%}' for a in per_repeat)}"
              f"  (spread {statistics.pstdev(per_repeat):.1%})")
    for kind, s in summary["by_type"].items():
        print(f"    {kind:20} {s['accuracy']:.0%} of {s['n']}")
    for label, key in [("Decline precision", "abstention_precision"), ("Decline recall", "abstention_recall"),
                       ("Valid citations", "citation_validity"),
                       ("Claims supported", "claim_support_rate"),
                       ("Unsupported claims", "answers_with_unsupported_claims"),
                       ("Unmatched numbers", "answers_with_unmatched_numbers"),
                       ("Plain-text answers", "format_ok_rate")]:
        value = summary[key]
        print(f"  {label:19} {'n/a' if value is None else f'{value:.0%}'}")
    print("  (Unsupported claims and unmatched numbers: share of answers with at least one.)")
    retried = sum(1 for r in rows if r.get("retries"))
    if retried:
        print(f"  Markup retries      {retried}")
    print(f"  Latency p50 / p95   {summary['latency_p50_s']:.2f}s / {summary['latency_p95_s']:.2f}s")
    if summary["cost_usd"] is not None:
        print(f"  Cost                ${summary['cost_usd']:.4f}")
    print(f"  Results             {path}")
    return summary


if __name__ == "__main__":
    main()

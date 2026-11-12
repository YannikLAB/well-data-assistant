"""Grade answers against gold answers without an LLM.

Each question type has required checks: the numbers a correct answer must state (within
a tolerance) and whether it should decline. A question is correct only when every check
passes. Citations are scored separately.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from welldata.answers import Answer

MIN_DECADE_WELLS = 5


@dataclass
class Check:
    quantity: str
    gold: object
    tolerance: float | None = None   # None means exact match
    decade: int | None = None

    def label(self) -> str:
        return f"{self.quantity}[{self.decade}]" if self.decade is not None else self.quantity


def _feet(value):
    return max(1.0, 0.03 * abs(value))


def _gpm(value):
    return max(0.5, 0.03 * abs(value))


def required_checks(item: dict) -> list[Check]:
    g, kind = item["gold"], item["type"]
    checks: list[Check] = []

    def add(quantity, value, tol=None, decade=None):
        if value is not None:
            checks.append(Check(quantity, value, tol, decade))

    if kind == "area_depth":
        add("well_count", g["well_count"])
        add("depth_median_ft", g["depth_ft"]["median"], _feet(g["depth_ft"]["median"] or 0))
        add("first_water_median_ft", g["first_water_ft"]["median"], _feet(g["first_water_ft"]["median"] or 0))
    elif kind == "area_yield":
        y = g["yield_gpm"]
        add("well_count", g["well_count"])
        add("yield_median_gpm", y["median"], _gpm(y["median"] or 0))
        add("share_at_least_10_gpm", y["share_at_least_10_gpm"], 0.03)
    elif kind == "area_screens":
        s = g["screened_interval_ft"]
        add("screen_top_median_ft", s["median_top"], _feet(s["median_top"] or 0))
        add("screen_bottom_median_ft", s["median_bottom"], _feet(s["median_bottom"] or 0))
    elif kind == "area_plugged":
        add("plugged_wells", g["plugged"]["wells"])
        add("plugged_most_recent_date", g["plugged"]["most_recent_date"])
    elif kind == "nearest_pump_test":
        add("pump_test_well_id", g["well_id"])
        add("pump_test_distance_miles", g["distance_miles"], 0.1)
        add("pump_test_rate_gpm", g["rate_gpm"], _gpm(g["rate_gpm"] or 0))
        add("pump_test_drawdown_ft", g["drawdown_ft"], _feet(g["drawdown_ft"] or 0))
    elif kind == "water_level_trend":
        for d in g["decades"]:
            if d["wells"] >= MIN_DECADE_WELLS:
                add("decade_median_depth_to_water_ft", d["median_depth_to_water_ft"],
                    _feet(d["median_depth_to_water_ft"]), d["decade"])
    elif kind == "report_explanation":
        add("total_depth_ft", g["total_depth_ft"], _feet(g["total_depth_ft"] or 0))
        add("depth_to_water_ft", g["depth_to_water_ft"], _feet(g["depth_to_water_ft"] or 0))
        tops = [s["top_ft"] for s in g["screens"] if s["top_ft"] is not None]
        bottoms = [s["bottom_ft"] for s in g["screens"] if s["bottom_ft"] is not None]
        add("screen_top_ft", min(tops) if tops else None, _feet(min(tops)) if tops else None)
        add("screen_bottom_ft", max(bottoms) if bottoms else None, _feet(max(bottoms)) if bottoms else None)
    elif kind == "unanswerable" and g.get("behavior") == "decline_guarantee_give_evidence":
        add("share_at_least_10_gpm", g["evidence"]["yield_gpm"]["share_at_least_10_gpm"], 0.03)
    return checks


def _should_decline(item: dict) -> bool:
    return str(item["gold"].get("behavior", "")).startswith("decline")


def _allowed_ids(item: dict) -> set[int] | None:
    g = item["gold"]
    if item["type"].startswith("area"):
        return set(g["nearest_well_ids"])
    if item["type"] == "nearest_pump_test":
        return {g["well_id"]}
    if item["type"] == "report_explanation":
        return {g["well_id"]}
    return None


def _matches(check: Check, value) -> bool:
    if value is None:
        return False
    if check.tolerance is None:
        if isinstance(check.gold, (int, float)) and not isinstance(check.gold, bool):
            try:
                return float(value) == float(check.gold)
            except (TypeError, ValueError):
                return False
        return str(value).strip() == str(check.gold)
    try:
        return abs(float(value) - float(check.gold)) <= check.tolerance
    except (TypeError, ValueError):
        return False


def score(item: dict, answer: Answer | None) -> dict:
    base = {"id": item["id"], "type": item["type"], "should_decline": _should_decline(item)}
    checks = required_checks(item)
    if answer is None:
        return {**base, "correct": False, "declined": False, "decline_ok": False, "checks_passed": 0,
                "checks_total": len(checks), "failures": ["no answer returned"], "citations_ok": None}

    failures = []
    decline_ok = answer.declined == base["should_decline"]
    if not decline_ok:
        failures.append("should have declined" if base["should_decline"] else "declined an answerable question")

    by_key = {}
    for c in answer.claims:
        by_key.setdefault((c.get("quantity"), c.get("decade")), c.get("value"))
    passed = 0
    for check in checks:
        value = by_key.get((check.quantity, check.decade))
        if _matches(check, value):
            passed += 1
        else:
            failures.append(f"{check.label()}: expected {check.gold!r}, got {value!r}")

    allowed = _allowed_ids(item)
    if not answer.cited_well_ids or item["type"] == "unanswerable":
        citations_ok = None  # nothing to check, or no fixed set of wells to cite
    elif allowed is None:
        citations_ok = False
    else:
        citations_ok = set(answer.cited_well_ids) <= allowed

    return {**base, "correct": decline_ok and passed == len(checks), "declined": answer.declined,
            "decline_ok": decline_ok, "checks_passed": passed, "checks_total": len(checks),
            "failures": failures, "citations_ok": citations_ok}


def _bootstrap(values: list[bool], n_boot: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[min(n_boot - 1, int(0.975 * n_boot))]


def summarize(results: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    n = len(results)
    correct = [r["correct"] for r in results]
    low, high = _bootstrap(correct, n_boot, seed) if n else (0.0, 0.0)
    by_type = {}
    for r in results:
        t = by_type.setdefault(r["type"], {"n": 0, "correct": 0})
        t["n"] += 1
        t["correct"] += r["correct"]
    declined = [r for r in results if r["declined"]]
    should = [r for r in results if r["should_decline"]]
    cited = [r for r in results if r["citations_ok"] is not None]
    return {
        "n": n,
        "accuracy": sum(correct) / n if n else 0.0,
        "ci_low": low, "ci_high": high,
        "by_type": {k: {"n": v["n"], "accuracy": v["correct"] / v["n"]} for k, v in sorted(by_type.items())},
        "abstention_precision": (sum(r["should_decline"] for r in declined) / len(declined)) if declined else None,
        "abstention_recall": (sum(r["declined"] for r in should) / len(should)) if should else None,
        "citation_validity": (sum(r["citations_ok"] for r in cited) / len(cited)) if cited else None,
    }

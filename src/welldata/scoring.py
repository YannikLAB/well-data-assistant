"""Grade answers against gold answers without an LLM.

Three separate measures:

- Correctness: each question type has required checks, the numbers a correct answer must
  state (within a tolerance) and whether it should decline. A question is correct only when
  every check passes. A question can also list its checks explicitly in gold["checks"]
  (with location_index for comparisons) and words the answer text must contain in
  gold["text_must_include"].
- Claim support: every claim in the answer, required or not, must match a value in the
  tool outputs of that run. This catches numbers the model made up or misread.
- Format: the answer text must be plain text, without leftover tool-call markup.

Citations are scored separately.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from welldata.answers import Answer, has_markup

MIN_DECADE_WELLS = 5
Z_95 = 1.959964


@dataclass
class Check:
    quantity: str
    gold: object
    tolerance: float | None = None   # None means exact match
    decade: int | None = None
    location_index: int | None = None

    def label(self) -> str:
        label = self.quantity
        if self.decade is not None:
            label += f"[{self.decade}]"
        if self.location_index is not None:
            label += f"@{self.location_index}"
        return label


def _feet(value):
    return max(1.0, 0.03 * abs(value))


def _gpm(value):
    return max(0.5, 0.03 * abs(value))


FEET = {"depth_median_ft", "first_water_median_ft", "screen_top_median_ft", "screen_bottom_median_ft",
        "pump_test_drawdown_ft", "decade_median_depth_to_water_ft", "total_depth_ft", "depth_to_water_ft",
        "screen_top_ft", "screen_bottom_ft"}
GPM = {"yield_median_gpm", "pump_test_rate_gpm"}


def tolerance_for(quantity: str, value) -> float | None:
    """The grading tolerance for a quantity; None means an exact match."""
    if value is None or isinstance(value, str):
        return None
    if quantity in FEET:
        return _feet(value)
    if quantity in GPM:
        return _gpm(value)
    if quantity == "share_at_least_10_gpm":
        return 0.03
    if quantity == "pump_test_distance_miles":
        return 0.1
    return None


def required_checks(item: dict) -> list[Check]:
    g, kind = item["gold"], item["type"]
    checks: list[Check] = []

    def add(quantity, value, tol=None, decade=None):
        if value is not None:
            checks.append(Check(quantity, value, tol, decade))

    if "checks" in g:
        return [Check(c["quantity"], c["value"],
                      c.get("tolerance", tolerance_for(c["quantity"], c["value"])),
                      c.get("decade"), c.get("location_index"))
                for c in g["checks"] if c["value"] is not None]
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
    if "allowed_well_ids" in g:
        return set(g["allowed_well_ids"])
    if item["type"].startswith("area"):
        return set(g["nearest_well_ids"])
    if item["type"] == "nearest_pump_test":
        return {g["well_id"]}
    if item["type"] == "report_explanation":
        return {g["well_id"]}
    return None


def _equal(gold, value, tolerance) -> bool:
    if value is None or gold is None:
        return value is None and gold is None
    if tolerance is None:
        if isinstance(gold, (int, float)) and not isinstance(gold, bool):
            try:
                return float(value) == float(gold)
            except (TypeError, ValueError):
                return False
        return str(value).strip() == str(gold)
    try:
        return abs(float(value) - float(gold)) <= tolerance
    except (TypeError, ValueError):
        return False


def _matches(check: Check, value) -> bool:
    return value is not None and _equal(check.gold, value, check.tolerance)


def _claim_for(check: Check, claims: list[dict]):
    """The first claim stating this check's quantity (for the same decade and place)."""
    for c in claims:
        if (c.get("quantity") == check.quantity and c.get("decade") == check.decade
                and (c.get("location_index") or 1) == (check.location_index or 1)):
            return c.get("value")
    return None


# Claim support ---------------------------------------------------------------------------

def _get(data, *path):
    for key in path:
        if not isinstance(data, dict) or key not in data:
            return []
        data = data[key]
    return [data]


def _screen_ends(record: dict, key: str, pick) -> list:
    values = [s.get(key) for s in record.get("screens") or [] if s.get(key) is not None]
    return [pick(values)] if values else ([None] if "screens" in record else [])


def _candidates(quantity: str, decade, tool_name: str, out: dict) -> list:
    """Values a tool output offers for a claim's quantity (empty if the tool doesn't cover it)."""
    if tool_name == "summarize_area":
        if decade is not None:
            return []
        paths = {"well_count": ("well_count",), "depth_median_ft": ("depth_ft", "median"),
                 "first_water_median_ft": ("first_water_ft", "median"),
                 "yield_median_gpm": ("yield_gpm", "median"),
                 "share_at_least_10_gpm": ("yield_gpm", "share_at_least_10_gpm"),
                 "screen_top_median_ft": ("screened_interval_ft", "median_top"),
                 "screen_bottom_median_ft": ("screened_interval_ft", "median_bottom"),
                 "plugged_wells": ("plugged", "wells"),
                 "plugged_most_recent_date": ("plugged", "most_recent_date")}
        return _get(out, *paths[quantity]) if quantity in paths else []
    if tool_name == "get_water_level_trend":
        field = {"decade_median_depth_to_water_ft": "median_depth_to_water_ft", "well_count": "wells"}
        if quantity not in field:
            return []
        return [d.get(field[quantity]) for d in out.get("decades") or []
                if decade is None or d.get("decade") == decade]
    if tool_name == "find_nearest_pump_test":
        if not out.get("found"):
            return []
        paths = {"pump_test_well_id": "well_id", "pump_test_distance_miles": "distance_miles",
                 "pump_test_rate_gpm": "rate_gpm", "pump_test_drawdown_ft": "drawdown_ft",
                 "total_depth_ft": "total_depth_ft"}
        return _get(out, paths[quantity]) if quantity in paths else []
    if tool_name == "get_well_record":
        if quantity in ("total_depth_ft", "depth_to_water_ft", "plugged_date"):
            return _get(out, quantity)
        if quantity == "screen_top_ft":
            return _screen_ends(out, "top_ft", min)
        if quantity == "screen_bottom_ft":
            return _screen_ends(out, "bottom_ft", max)
        test = out.get("pump_test")
        if test:
            if quantity == "pump_test_well_id":
                return _get(out, "well_id")
            paths = {"pump_test_rate_gpm": "rate_gpm", "pump_test_drawdown_ft": "drawdown_ft"}
            return _get(test, paths[quantity]) if quantity in paths else []
    return []


def claim_supported(claim: dict, tool_outputs: list[dict]) -> bool:
    """True if a value in some successful tool output matches the claim."""
    quantity, decade, value = claim.get("quantity"), claim.get("decade"), claim.get("value")
    for call in tool_outputs:
        if call.get("error") is not None or not isinstance(call.get("output"), dict):
            continue
        for candidate in _candidates(quantity, decade, call.get("name"), call["output"]):
            if _equal(candidate, value, tolerance_for(quantity, candidate)):
                return True
    return False


# Numbers in the answer text --------------------------------------------------------------

_NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?!\d)(\s*%|\s*percent|st\b|nd\b|rd\b|th\b)?")
_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")
CONSTANTS = (10, 25)  # the 10 gpm target and the default 25-mile pump-test search


def _leaves(data, found: list) -> list:
    if isinstance(data, bool) or data is None:
        return found
    if isinstance(data, (int, float)):
        found.append(abs(float(data)))
    elif isinstance(data, str):
        date = _DATE.match(data)
        if date:
            found += [float(x) for x in date.groups()]
    elif isinstance(data, dict):
        for value in data.values():
            _leaves(value, found)
    elif isinstance(data, (list, tuple)):
        for value in data:
            _leaves(value, found)
    return found


def _numbers(text: str):
    """(printed form, value, decimals, is_percent) for each number in the text, skipping
    single digits and ordinals."""
    for m in _NUMBER.finditer(text or ""):
        whole, frac, suffix = m.group(1), m.group(2) or "", (m.group(3) or "").strip().lower()
        if suffix in ("st", "nd", "rd", "th") or (not frac and len(whole) == 1):
            continue
        yield whole + frac, float(whole.replace(",", "") + frac), max(0, len(frac) - 1), suffix in ("%", "percent")


def _number_matches(x: float, decimals: int, percent: bool, v: float) -> bool:
    if percent:
        return 0 <= v <= 1 and (round(100 * v, decimals) == x or abs(x - 100 * v) <= 1)
    if round(v, decimals) == x:
        return True
    return 0 < v < 10000 and abs(x - v) <= 0.03 * v


def unmatched_numbers(text: str, question: str, tool_outputs: list[dict]) -> list[str]:
    """Numbers in the answer text that appear in no tool input or output and not in the
    question, allowing for rounding, 3% approximations and percentages. A screening aid
    for reading answers, not a correctness score: numbers the model derived (differences,
    layer thicknesses) are listed too."""
    sources = [float(c) for c in CONSTANTS] + [v for _, v, _, _ in _numbers(question)]
    for call in tool_outputs:
        _leaves(call.get("input"), sources)
        if call.get("error") is None:
            _leaves(call.get("output"), sources)
        else:
            sources += [v for _, v, _, _ in _numbers(call["error"])]
    return [printed for printed, x, decimals, percent in _numbers(text)
            if not any(_number_matches(x, decimals, percent, v) for v in sources)]


def _text_ok(item: dict, text: str) -> list[str]:
    """Required words (or any of a list of alternatives) missing from the answer text."""
    missing = []
    lowered = (text or "").lower()
    for need in item["gold"].get("text_must_include", []):
        options = [need] if isinstance(need, str) else need
        if not any(o.lower() in lowered for o in options):
            missing.append(" / ".join(options))
    return missing


def score(item: dict, answer: Answer | None, tool_outputs: list[dict] | None = None) -> dict:
    """Grade one answer. Pass the run's tool outputs to also check that every claim is supported."""
    base = {"id": item["id"], "type": item["type"], "should_decline": _should_decline(item),
            "decline_optional": bool(item["gold"].get("decline_either_way"))}
    checks = required_checks(item)
    if answer is None:
        return {**base, "correct": False, "declined": False, "decline_ok": False, "checks_passed": 0,
                "checks_total": len(checks), "failures": ["no answer returned"], "citations_ok": None,
                "format_ok": None, "claims_total": None, "claims_supported": None,
                "unsupported_claims": [], "unmatched_numbers": None}

    failures = []
    decline_ok = answer.declined == base["should_decline"] or base["decline_optional"]
    if not decline_ok:
        failures.append("should have declined" if base["should_decline"] else "declined an answerable question")

    passed = 0
    for check in checks:
        value = _claim_for(check, answer.claims)
        if _matches(check, value):
            passed += 1
        else:
            failures.append(f"{check.label()}: expected {check.gold!r}, got {value!r}")
    missing_text = _text_ok(item, answer.text)
    failures += [f"answer text should mention: {m}" for m in missing_text]
    forbidden = [c["quantity"] for c in answer.claims
                 if c.get("quantity") in item["gold"].get("must_not_claim", []) and c.get("value") is not None]
    failures += [f"{q}: the record has no value, but the answer states {q}" for q in forbidden]

    allowed = _allowed_ids(item)
    if not answer.cited_well_ids or item["type"] == "unanswerable" or (base["should_decline"] and allowed is None):
        citations_ok = None  # nothing to check, or no fixed set of wells to cite
    elif allowed is None:
        citations_ok = False
    else:
        citations_ok = set(answer.cited_well_ids) <= allowed

    if tool_outputs is None:
        claims_total = claims_supported = unmatched = None
        unsupported = []
    else:
        unsupported = [{k: c.get(k) for k in ("quantity", "value", "decade", "location_index") if k in c}
                       for c in answer.claims if not claim_supported(c, tool_outputs)]
        claims_total = len(answer.claims)
        claims_supported = claims_total - len(unsupported)
        unmatched = unmatched_numbers(answer.text, item.get("question", ""), tool_outputs)

    return {**base, "correct": decline_ok and passed == len(checks) and not missing_text and not forbidden,
            "declined": answer.declined, "decline_ok": decline_ok, "checks_passed": passed,
            "checks_total": len(checks), "failures": failures, "citations_ok": citations_ok,
            "format_ok": not has_markup(answer.text), "claims_total": claims_total,
            "claims_supported": claims_supported, "unsupported_claims": unsupported,
            "unmatched_numbers": unmatched}


# Summary ---------------------------------------------------------------------------------

def wilson_interval(successes: float, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a proportion. Unlike the bootstrap, it stays informative at
    0% and 100%: 25 of 25 correct gives about 87% to 100%."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)


def _rate(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return sum(values) / len(values) if values else None


def summarize(results: list[dict]) -> dict:
    """Accuracy with a 95% Wilson interval, plus decline, citation, claim and format rates.

    With repeated runs the interval uses the number of distinct questions, not the number of
    answers, because repeats of one question are not independent.
    """
    n = len(results)
    correct = [r["correct"] for r in results]
    accuracy = sum(correct) / n if n else 0.0
    distinct = len({r["id"] for r in results}) or n
    low, high = wilson_interval(accuracy * distinct, distinct)
    by_type = {}
    for r in results:
        t = by_type.setdefault(r["type"], {"n": 0, "correct": 0})
        t["n"] += 1
        t["correct"] += r["correct"]
    declined = [r for r in results if r["declined"] and not r.get("decline_optional")]
    should = [r for r in results if r["should_decline"]]
    cited = [r for r in results if r["citations_ok"] is not None]
    checked = [r for r in results if r.get("claims_total") is not None]
    claims = sum(r["claims_total"] for r in checked)
    with_numbers = [r for r in results if r.get("unmatched_numbers") is not None]
    return {
        "n": n,
        "accuracy": accuracy,
        "ci_low": low, "ci_high": high,
        "by_type": {k: {"n": v["n"], "accuracy": v["correct"] / v["n"]} for k, v in sorted(by_type.items())},
        "abstention_precision": (sum(r["should_decline"] for r in declined) / len(declined)) if declined else None,
        "abstention_recall": (sum(r["declined"] for r in should) / len(should)) if should else None,
        "citation_validity": (sum(r["citations_ok"] for r in cited) / len(cited)) if cited else None,
        "claim_support_rate": (sum(r["claims_supported"] for r in checked) / claims) if claims else None,
        "answers_with_unsupported_claims": (
            sum(r["claims_supported"] < r["claims_total"] for r in checked) / len(checked)) if checked else None,
        "format_ok_rate": _rate(results, "format_ok"),
        "answers_with_unmatched_numbers": (
            sum(bool(r["unmatched_numbers"]) for r in with_numbers) / len(with_numbers)) if with_numbers else None,
    }

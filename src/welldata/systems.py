"""The systems compared in the evaluation. Each has `answer(question) -> AgentResult`.

- ToolAssistant: an LLM that calls the query tools, then submits a structured answer.
- NoToolsLLM: the same LLM with no tools (baseline: what the model knows on its own).
- KeywordBaseline: no LLM; regular expressions and keywords pick a tool and fill a template
  (baseline: how far simple rules get).
"""

from __future__ import annotations

import re
import time

from welldata import tools
from welldata.agent import AgentResult, run_agent
from welldata.answers import Answer

_LOCATION = {
    "latitude": {"type": "number", "description": "Decimal degrees, e.g. 35.4823"},
    "longitude": {"type": "number", "description": "Decimal degrees (negative in Oklahoma), e.g. -97.5310"},
}
_USE_CLASS = {"type": "string", "enum": list(tools.USE_CLASSES),
              "description": "Optional. 'Domestic' for a home well; omit for all uses."}

TOOL_SPECS = [
    {"name": "summarize_area",
     "description": (
         "Summarize groundwater wells within radius_miles of a point: depth, depth to first water, "
         "yield (with the share reaching 10 gpm), screened interval, and plugged wells. Use for "
         "'how deep are wells near...', 'what yield do nearby wells get', 'where are wells "
         "screened', 'have wells nearby been plugged'. Returns counts for every measure and "
         "nearest_well_ids to cite."),
     "input_schema": {"type": "object", "properties": {
         **_LOCATION,
         "radius_miles": {"type": "number", "description": "0.5 for 'half a mile', 5 for '5 miles'. Default 0.5."},
         "use_class": _USE_CLASS}, "required": ["latitude", "longitude"]}},
    {"name": "find_nearest_pump_test",
     "description": (
         "Find the closest well with a pump test (up to max_miles) and return its distance, rate, "
         "duration and drawdown. Use for 'what did a pump test near me show'. Returns found=false "
         "if none is within max_miles."),
     "input_schema": {"type": "object", "properties": {
         **_LOCATION, "use_class": _USE_CLASS,
         "max_miles": {"type": "number", "description": "Search limit, default 25."}},
         "required": ["latitude", "longitude"]}},
    {"name": "get_water_level_trend",
     "description": (
         "Median depth to water at the time of drilling, by the decade wells were drilled, within "
         "radius_miles (default 5). Use for 'is the water table dropping'. Each decade has its well "
         "count, change_ft versus the previous decade, and few_wells=true under 5 wells."),
     "input_schema": {"type": "object", "properties": {
         **_LOCATION,
         "radius_miles": {"type": "number", "description": "Default 5."},
         "use_class": _USE_CLASS}, "required": ["latitude", "longitude"]}},
    {"name": "get_well_record",
     "description": (
         "Everything known about one well by its ID: depth, depth to water at drilling, screens, "
         "lithology layers, pump test, plugging and the driller's comments. Use when the user "
         "names a well ID or asks what a well report says. Never includes owner details."),
     "input_schema": {"type": "object", "properties": {
         "well_id": {"type": "integer", "description": "A well ID like 239950."}},
         "required": ["well_id"]}},
]

ASSISTANT_PROMPT = """You answer questions about Oklahoma water wells for people deciding where and how
to drill. You use the state's well records (Oklahoma Water Resources Board) through the tools.

Rules:
- Get every number from a tool. Never estimate or recall numbers yourself.
- Say how many wells each number rests on. If few wells support it, say so.
- "Half a mile" means radius_miles 0.5 and "5 miles" means 5. A "home well" is use_class "Domestic".
- Decline (declined=true) when the records can't answer: water safety or quality, current water
  levels, owners or contact details, costs, guarantees, places outside Oklahoma, or well IDs that
  don't exist. Say what the records can and can't tell. For a guarantee question, still share what
  nearby home wells within 5 miles report, and list that share as a claim.
- Never reveal or guess owner names.
- When explaining a well report, use plain language: how deep the well is, where water stood when it
  was drilled, where the screens are, and which layers held water.
- Finish by calling submit_answer once. List every number you state as a claim, using decade for
  decade_median_depth_to_water_ft, and cite the well IDs you relied on."""

NO_TOOLS_PROMPT = """You answer questions about Oklahoma water wells. You have no access to well
records or any database; answer from your general knowledge only. If you can't know the answer,
set declined=true. Finish by calling submit_answer once, listing every number you state as a claim."""


class ToolAssistant:
    name = "assistant"

    def __init__(self, provider, conn):
        self.provider = provider
        self.tools = {
            "summarize_area": lambda **kw: tools.summarize_area(conn, **kw),
            "find_nearest_pump_test": lambda **kw: tools.find_nearest_pump_test(conn, **kw),
            "get_water_level_trend": lambda **kw: tools.get_water_level_trend(conn, **kw),
            "get_well_record": lambda **kw: tools.get_well_record(conn, **kw),
        }

    def answer(self, question: str) -> AgentResult:
        return run_agent(self.provider, ASSISTANT_PROMPT, question, self.tools, TOOL_SPECS)


class NoToolsLLM:
    name = "no_tools"

    def __init__(self, provider):
        self.provider = provider

    def answer(self, question: str) -> AgentResult:
        return run_agent(self.provider, NO_TOOLS_PROMPT, question, {}, [])


# Keyword baseline -------------------------------------------------------------------------

_COORDS = re.compile(r"(-?\d{1,3}\.\d+)\s*,\s*(-?\d{1,3}\.\d+)")
_WELL_ID = re.compile(r"\bwell\s+(\d+)", re.I)
_MILES = re.compile(r"(\d+(?:\.\d+)?)\s*miles?", re.I)
_DECLINE_WORDS = ("safe to drink", "water quality", "cost", "price", "who owns", "contact",
                  "today", "right now", "currently")


def _claims(pairs):
    return [{"quantity": q, "value": v} for q, v in pairs if v is not None]


class KeywordBaseline:
    name = "keyword"

    def __init__(self, conn):
        self.conn = conn

    def answer(self, question: str) -> AgentResult:
        start = time.perf_counter()
        try:
            answer, calls = self._answer(question)
        except tools.ToolInputError as err:
            answer, calls = Answer(text=f"I can't answer that from the records: {err}", declined=True), []
        return AgentResult(answer=answer, tool_calls=calls, steps=0,
                           latency_s=time.perf_counter() - start)

    def _answer(self, q: str):
        text = q.lower()
        coords = _COORDS.search(q)
        lat, lon = (float(coords.group(1)), float(coords.group(2))) if coords else (None, None)
        well = _WELL_ID.search(q)
        miles = _MILES.search(q)
        radius = 0.5 if "half a mile" in text else float(miles.group(1)) if miles else 0.5
        use = "Domestic" if "home well" in text else None

        if "guarantee" in text and lat is not None:
            area = tools.summarize_area(self.conn, lat, lon, 5, "Domestic")
            share = area["yield_gpm"]["share_at_least_10_gpm"]
            return Answer(text=f"No one can guarantee a yield. Of {area['yield_gpm']['wells']} nearby home "
                               f"wells, {share:.0%} report at least 10 gpm.", declined=True,
                          claims=_claims([("share_at_least_10_gpm", share)]),
                          cited_well_ids=area["nearest_well_ids"]), ["summarize_area"]
        if any(w in text for w in _DECLINE_WORDS):
            return Answer(text="The well records can't answer that.", declined=True), []

        if well and "report" in text:
            r = tools.get_well_record(self.conn, int(well.group(1)))
            tops = [s["top_ft"] for s in r["screens"] if s["top_ft"] is not None]
            bottoms = [s["bottom_ft"] for s in r["screens"] if s["bottom_ft"] is not None]
            claims = _claims([("total_depth_ft", r["total_depth_ft"]),
                              ("depth_to_water_ft", r["depth_to_water_ft"]),
                              ("screen_top_ft", min(tops) if tops else None),
                              ("screen_bottom_ft", max(bottoms) if bottoms else None)])
            return Answer(text=f"Well {r['well_id']} is {r['total_depth_ft']} ft deep; water stood at "
                               f"{r['depth_to_water_ft']} ft when it was drilled.",
                          declined=False, claims=claims, cited_well_ids=[r["well_id"]]), ["get_well_record"]
        if lat is None:
            return Answer(text="I couldn't find a location in the question.", declined=True), []

        if "pump test" in text:
            t = tools.find_nearest_pump_test(self.conn, lat, lon, use)
            if not t["found"]:
                return Answer(text="No pump test nearby.", declined=False), ["find_nearest_pump_test"]
            claims = _claims([("pump_test_well_id", t["well_id"]),
                              ("pump_test_distance_miles", t["distance_miles"]),
                              ("pump_test_rate_gpm", t["rate_gpm"]),
                              ("pump_test_drawdown_ft", t["drawdown_ft"])])
            return Answer(text=f"Closest pump test: well {t['well_id']}, {t['distance_miles']:.1f} miles away.",
                          declined=False, claims=claims, cited_well_ids=[t["well_id"]]), ["find_nearest_pump_test"]
        if "water table" in text or "depth to water" in text:
            trend = tools.get_water_level_trend(self.conn, lat, lon, 5)
            claims = [{"quantity": "decade_median_depth_to_water_ft", "value": d["median_depth_to_water_ft"],
                       "decade": d["decade"]} for d in trend["decades"]]
            return Answer(text="Median depth to water by decade.", declined=False, claims=claims), \
                ["get_water_level_trend"]

        area = tools.summarize_area(self.conn, lat, lon, radius, use)
        ids = area["nearest_well_ids"]
        if "plugged" in text:
            p = area["plugged"]
            claims = _claims([("plugged_wells", p["wells"]), ("plugged_most_recent_date", p["most_recent_date"])])
        elif "screen" in text:
            s = area["screened_interval_ft"]
            claims = _claims([("screen_top_median_ft", s["median_top"]), ("screen_bottom_median_ft", s["median_bottom"])])
        elif any(w in text for w in ("gpm", "gallons", "yield")):
            y = area["yield_gpm"]
            claims = _claims([("well_count", area["well_count"]), ("yield_median_gpm", y["median"]),
                              ("share_at_least_10_gpm", y["share_at_least_10_gpm"])])
        else:
            claims = _claims([("well_count", area["well_count"]), ("depth_median_ft", area["depth_ft"]["median"]),
                              ("first_water_median_ft", area["first_water_ft"]["median"])])
        return Answer(text=f"Based on {area['well_count']} wells within {radius} miles.", declined=False,
                      claims=claims, cited_well_ids=ids), ["summarize_area"]

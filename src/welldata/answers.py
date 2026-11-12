"""The structured final answer every system returns, so answers can be graded without an LLM.

LLM systems give their final answer by calling the submit_answer tool. Besides the plain
text for the user, the answer lists each number it states as a typed claim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Quantities a claim can name. Scoring compares these with the gold answer.
QUANTITIES = {
    "well_count": "Number of wells within the radius (all of them, from well_count); not for a single well",
    "yield_wells": "Number of wells that report a yield, which a yield median or share rests on",
    "depth_median_ft": "Median total well depth (ft)",
    "first_water_median_ft": "Median depth where water was first found (ft)",
    "yield_median_gpm": "Median reported yield (gpm)",
    "share_at_least_10_gpm": "Share of wells reporting at least 10 gpm, from 0 to 1",
    "screen_top_median_ft": "Median top of the screened interval (ft)",
    "screen_bottom_median_ft": "Median bottom of the screened interval (ft)",
    "plugged_wells": "Number of plugged wells",
    "plugged_most_recent_date": "Most recent plugging date (YYYY-MM-DD)",
    "pump_test_well_id": "Well ID of the pump test",
    "pump_test_distance_miles": "Distance to the pump-test well (miles)",
    "pump_test_rate_gpm": "Pump-test rate (gpm)",
    "pump_test_drawdown_ft": "Pump-test drawdown (ft)",
    "decade_median_depth_to_water_ft": "Median depth to water at drilling for one decade (ft); set decade",
    "total_depth_ft": "One well's total depth (ft)",
    "depth_to_water_ft": "One well's depth to water at drilling (ft)",
    "screen_top_ft": "One well's shallowest screen top (ft)",
    "screen_bottom_ft": "One well's deepest screen bottom (ft)",
    "plugged_date": "One well's plugging date (YYYY-MM-DD)",
    "first_water_ft": "One well's depth where water was first found (ft)",
    "yield_gpm": "One well's approximate yield (gpm)",
}

# Tool-call syntax that sometimes leaks into the answer text, e.g. '</answer> <parameter name='.
_MARKUP = re.compile(r"</?[A-Za-z_][\w:-]*(?:\s+[\w-]+=[\"']|\s*>)")


def has_markup(text: str) -> bool:
    """True if the text contains leftover tag markup such as '</answer>'."""
    return bool(_MARKUP.search(text or ""))

SUBMIT_ANSWER_TOOL = {
    "name": "submit_answer",
    "description": (
        "Give your final answer. Call this exactly once, at the end. Put the reply for the "
        "user in `answer` as plain text (no tags or markup). List every number you state in "
        "`claims`, using the quantity names below, and the well IDs you relied on in "
        "`cited_well_ids`. When the question compares places, set `location_index` on each "
        "claim (1 for the first place named, 2 for the second). Set `declined` to true when "
        "the records can't answer the question (for example water safety, current water "
        "levels, owners, costs, guarantees, places outside Oklahoma, or no location given)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "declined": {"type": "boolean"},
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "quantity": {"type": "string", "enum": list(QUANTITIES),
                                     "description": "; ".join(f"{k}: {v}" for k, v in QUANTITIES.items())},
                        "value": {"type": ["number", "string", "null"]},
                        "decade": {"type": "integer", "description": "Only for decade_median_depth_to_water_ft, e.g. 1990."},
                        "location_index": {"type": "integer", "description": "Only when comparing places: 1 for the first place named, 2 for the second."},
                    },
                    "required": ["quantity", "value"],
                },
            },
            "cited_well_ids": {"type": "array", "items": {"type": "integer"}},
            "answer": {"type": "string", "description": "Plain-language reply for the user, as plain text."},
        },
        "required": ["answer", "declined", "claims", "cited_well_ids"],
    },
}


_LEAKED_DECLINED = re.compile(r"<parameter\s+name=[\"']declined[\"']>\s*(true|false)", re.I)


@dataclass
class Answer:
    text: str
    declined: bool
    claims: list[dict] = field(default_factory=list)
    cited_well_ids: list[int] = field(default_factory=list)
    # Problems with how the model filled in the tool call, e.g. markup leaked into the text.
    format_problems: list[str] = field(default_factory=list)

    @classmethod
    def from_tool_input(cls, data: dict) -> "Answer":
        claims = [c for c in data.get("claims") or [] if isinstance(c, dict) and "quantity" in c]
        ids = []
        for i in data.get("cited_well_ids") or []:
            try:
                ids.append(int(i))
            except (TypeError, ValueError):
                pass
        text = str(data.get("answer", ""))
        problems = []
        if has_markup(text):
            problems.append("markup in answer")
        if "declined" not in data:
            problems.append("declined missing")
        answer = cls(text=text, declined=bool(data.get("declined", False)), claims=claims,
                     cited_well_ids=ids, format_problems=problems)
        if "declined" not in data:
            leaked = _LEAKED_DECLINED.search(text)
            answer.declined = bool(leaked) and leaked.group(1).lower() == "true"
        return answer

    def repaired(self) -> "Answer":
        """The answer with leaked markup cut from the text. A `declined` value that leaked into
        the text was already recovered when parsing."""
        match = _MARKUP.search(self.text)
        text = self.text[:match.start()].rstrip() if match else self.text
        return Answer(text=text, declined=self.declined, claims=self.claims,
                      cited_well_ids=self.cited_well_ids)

    def to_dict(self) -> dict:
        return {"answer": self.text, "declined": self.declined, "claims": self.claims,
                "cited_well_ids": self.cited_well_ids}

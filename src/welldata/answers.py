"""The structured final answer every system returns, so answers can be graded without an LLM.

LLM systems give their final answer by calling the submit_answer tool. Besides the plain
text for the user, the answer lists each number it states as a typed claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Quantities a claim can name. Scoring compares these with the gold answer.
QUANTITIES = {
    "well_count": "Number of wells the answer is based on",
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
}

SUBMIT_ANSWER_TOOL = {
    "name": "submit_answer",
    "description": (
        "Give your final answer. Call this exactly once, at the end. Put the reply for the "
        "user in `answer`. List every number you state in `claims`, using the quantity names "
        "below, and the well IDs you relied on in `cited_well_ids`. Set `declined` to true "
        "when the records can't answer the question (for example water safety, current "
        "water levels, owners, costs, guarantees, or places outside Oklahoma)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "description": "Plain-language reply for the user."},
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
                    },
                    "required": ["quantity", "value"],
                },
            },
            "cited_well_ids": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["answer", "declined", "claims", "cited_well_ids"],
    },
}


@dataclass
class Answer:
    text: str
    declined: bool
    claims: list[dict] = field(default_factory=list)
    cited_well_ids: list[int] = field(default_factory=list)

    @classmethod
    def from_tool_input(cls, data: dict) -> "Answer":
        claims = [c for c in data.get("claims") or [] if isinstance(c, dict) and "quantity" in c]
        ids = []
        for i in data.get("cited_well_ids") or []:
            try:
                ids.append(int(i))
            except (TypeError, ValueError):
                pass
        return cls(text=str(data.get("answer", "")), declined=bool(data.get("declined", False)),
                   claims=claims, cited_well_ids=ids)

    def to_dict(self) -> dict:
        return {"answer": self.text, "declined": self.declined, "claims": self.claims,
                "cited_well_ids": self.cited_well_ids}

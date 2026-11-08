"""Parse fields from saved OWRB well completion reports (HTML).

The reports are server-generated with a fixed layout, so small regular
expressions are enough and much faster than a full HTML parser. Uses only the
standard library, so it runs anywhere the reports are.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

_WELL_ID = re.compile(r"<title>\s*Well ID:\s*(\d+)")
_DEPTH_TO_WATER = re.compile(r"Depth to water at time of drilling(?:&nbsp;|\s)*<u>(.*?)</u>", re.S)
_COMMENTS = re.compile(r"Comments:(.*?)</td>", re.S)
_LITHOLOGY_ROW = re.compile(
    r'<td class="border">(.*?)</td>\s*'
    r'<td class="border"[^>]*>(.*?)</td>\s*'
    r'<td class="border"[^>]*>(.*?)</td>\s*'
    r'<td class="border"[^>]*>(.*?)</td>',
    re.S,
)
_LITHOLOGY_START = "LITHOLOGY DESCRIPTION"
_LITHOLOGY_END = "WELL LOCATION TO POTENTIAL"
_EMPTY_COMMENTS = {"", "n/a", "na", "none"}


@dataclass
class LithologyLayer:
    material: str
    top_ft: float | None
    bottom_ft: float | None
    saturated: bool | None


@dataclass
class ReportFields:
    well_id: int | None
    depth_to_water_ft: float | None
    comments: str | None
    lithology: list[LithologyLayer] = field(default_factory=list)


def _text(fragment: str) -> str:
    fragment = re.sub(r"<br\s*/?>", " ", fragment, flags=re.I)
    fragment = re.sub(r"<[^>]+>", " ", fragment)
    return " ".join(html.unescape(fragment).replace("\xa0", " ").split())


def _number(fragment: str) -> float | None:
    try:
        return float(_text(fragment))
    except ValueError:
        return None


def _saturated(fragment: str) -> bool | None:
    return {"Y": True, "N": False}.get(_text(fragment).upper())


def parse_report(page: str) -> ReportFields:
    well_id = _WELL_ID.search(page)
    depth = _DEPTH_TO_WATER.search(page)
    comments = _COMMENTS.search(page)
    comment_text = _text(comments.group(1)) if comments else ""

    start = page.find(_LITHOLOGY_START)
    end = page.find(_LITHOLOGY_END, start)
    section = page[start:end] if start >= 0 else ""
    layers = [
        LithologyLayer(_text(m), _number(top), _number(bottom), _saturated(sat))
        for m, top, bottom, sat in _LITHOLOGY_ROW.findall(section)
    ]
    return ReportFields(
        well_id=int(well_id.group(1)) if well_id else None,
        depth_to_water_ft=_number(depth.group(1)) if depth else None,
        comments=None if comment_text.lower() in _EMPTY_COMMENTS else comment_text,
        lithology=layers,
    )

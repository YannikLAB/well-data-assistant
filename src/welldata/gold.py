"""Reference ("gold") answers for the evaluation set.

Computed from the cleaned tables with pandas and WGS84 geodesic distances, deliberately
not with the SQL in tools.py. If a tool and this module disagree, one of them has a bug.
Semantics match the tools: groundwater wells only (no test holes), medians interpolate.
"""

from __future__ import annotations

import datetime as dt
import math

import pandas as pd
from geographiclib.geodesic import Geodesic

METERS_PER_MILE = 1609.344
MIN_WELLS_PER_DECADE = 5
_GEOD = Geodesic.WGS84


def _median(series: pd.Series) -> float | None:
    s = series.dropna()
    return None if s.empty else float(s.median())


def _iso(value):
    return value.isoformat() if isinstance(value, (dt.date, dt.datetime)) else value


def wells_within(tables: dict, lat: float, lon: float, radius_miles: float,
                 use_class: str | None = None) -> pd.DataFrame:
    """Groundwater wells within the radius, with distance in meters, closest first."""
    wells = tables["wells"]
    wells = wells[wells["work_type"] == "Groundwater Well"]
    if use_class is not None:
        wells = wells[wells["use_class"] == use_class]
    # Cheap bounding-box prefilter, then exact geodesic distance.
    dlat = radius_miles / 68.0
    dlon = dlat / max(math.cos(math.radians(lat)), 0.1)
    box = wells[wells["latitude"].between(lat - dlat, lat + dlat)
                & wells["longitude"].between(lon - dlon, lon + dlon)].copy()
    box["dist_m"] = [_GEOD.Inverse(lat, lon, a, b, Geodesic.DISTANCE)["s12"]
                     for a, b in zip(box["latitude"], box["longitude"])]
    near = box[box["dist_m"] <= radius_miles * METERS_PER_MILE]
    return near.sort_values(["dist_m", "well_id"])


def area_summary(tables: dict, lat: float, lon: float, radius_miles: float = 0.5,
                 use_class: str | None = None) -> dict:
    near = wells_within(tables, lat, lon, radius_miles, use_class)
    ids = set(near["well_id"])
    yields = near["approx_yield_gpm"].dropna()

    screens = tables["screens"][tables["screens"]["well_id"].isin(ids)]
    spans = screens.groupby("well_id").agg(top_ft=("top_ft", "min"), bottom_ft=("bottom_ft", "max"))
    plugs = tables["pluggings"][tables["pluggings"]["well_id"].isin(ids)]
    latest = plugs["plugged_date"].dropna()

    return {
        "well_count": len(near),
        "nearest_well_ids": [int(i) for i in near["well_id"].head(10)],
        "depth_ft": {"wells": int(near["total_depth_ft"].notna().sum()),
                     "median": _median(near["total_depth_ft"])},
        "first_water_ft": {"wells": int(near["first_water_ft"].notna().sum()),
                           "median": _median(near["first_water_ft"])},
        "yield_gpm": {"wells": len(yields), "median": _median(yields),
                      "share_at_least_10_gpm": None if yields.empty else float((yields >= 10).mean())},
        "screened_interval_ft": {"wells": len(spans), "median_top": _median(spans["top_ft"]),
                                 "median_bottom": _median(spans["bottom_ft"])},
        "plugged": {"wells": len(plugs), "most_recent_date": _iso(max(latest)) if len(latest) else None},
    }


def nearest_pump_test(tables: dict, lat: float, lon: float, use_class: str | None = None,
                      max_miles: float = 25) -> dict:
    near = wells_within(tables, lat, lon, max_miles, use_class)
    tests = tables["pump_tests"].copy()
    tests["drawdown_ft"] = tests["pumping_depth_to_water_ft"] - tests["start_depth_to_water_ft"]
    tests = tests[tests["drawdown_ft"].isna() | (tests["drawdown_ft"] >= 0)]
    joined = near.merge(tests, on="well_id").sort_values(["dist_m", "well_id"])
    if joined.empty:
        return {"found": False, "searched_miles": max_miles}
    row = joined.iloc[0]
    return {"found": True, "well_id": int(row["well_id"]),
            "distance_miles": row["dist_m"] / METERS_PER_MILE,
            "rate_gpm": row["rate_gpm"],
            "drawdown_ft": None if pd.isna(row["drawdown_ft"]) else float(row["drawdown_ft"])}


def water_level_trend(tables: dict, lat: float, lon: float, radius_miles: float = 5,
                      use_class: str | None = None) -> dict:
    near = wells_within(tables, lat, lon, radius_miles, use_class)
    measured = near.merge(tables["report_fields"], on="well_id")
    measured = measured[measured["depth_to_water_ft"].notna() & measured["construction_date"].notna()]
    measured = measured.assign(decade=[d.year // 10 * 10 for d in measured["construction_date"]])
    by = (measured.groupby("decade")["depth_to_water_ft"].agg(["count", "median"])
          .reset_index().sort_values("decade"))
    decades, previous = [], None
    for decade, count, median in by.itertuples(index=False):
        decades.append({"decade": int(decade), "wells": int(count),
                        "median_depth_to_water_ft": float(median),
                        "change_ft": None if previous is None else float(median - previous),
                        "few_wells": int(count) < MIN_WELLS_PER_DECADE})
        previous = median
    return {"decades": decades}


def well_facts(tables: dict, well_id: int) -> dict:
    """Key facts a plain-language report explanation must get right."""
    w = tables["wells"].set_index("well_id").loc[well_id]
    fields = tables["report_fields"].set_index("well_id")
    screens = tables["screens"][tables["screens"]["well_id"] == well_id].sort_values("layer")
    lith = tables["lithology"][tables["lithology"]["well_id"] == well_id].sort_values("seq")
    sat = lith[lith["saturated"] == True]  # noqa: E712 (column may hold None)
    plugs = tables["pluggings"].set_index("well_id")

    def num(v):
        return None if pd.isna(v) else float(v)

    return {
        "well_id": int(well_id),
        "use_class": w["use_class"],
        "construction_date": _iso(w["construction_date"]),
        "total_depth_ft": num(w["total_depth_ft"]),
        "first_water_ft": num(w["first_water_ft"]),
        "approx_yield_gpm": num(w["approx_yield_gpm"]),
        "depth_to_water_ft": num(fields["depth_to_water_ft"].get(well_id)),
        "screens": [{"layer": int(r.layer), "top_ft": num(r.top_ft), "bottom_ft": num(r.bottom_ft)}
                    for r in screens.itertuples()],
        "saturated_layers": [{"material": r.material, "top_ft": num(r.top_ft),
                              "bottom_ft": num(r.bottom_ft)} for r in sat.itertuples()],
        "plugged_date": _iso(plugs["plugged_date"].get(well_id)) if well_id in plugs.index else None,
        "has_comment": bool((tables["comments"]["well_id"] == well_id).any()),
    }

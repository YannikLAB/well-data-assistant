"""Query tools the assistant calls. The LLM never writes SQL: it picks a tool and fills in the
arguments. Every answer says how many wells it rests on, and every input error says how to
fix the call.

Tools:
- summarize_area: depth, first water, yield, screens and plugging for wells near a location
- find_nearest_pump_test: the closest well with a pump test, and its results
- get_water_level_trend: depth to water at drilling, by decade, near a location
- get_well_record: everything known about one well (no owner details)

Test holes are excluded everywhere: they aren't finished wells.
"""

from __future__ import annotations

import datetime as dt

import psycopg

from welldata.quality import OK_LAT, OK_LON

METERS_PER_MILE = 1609.344
MAX_RADIUS_MILES = 25
NEAREST_IDS = 10
MIN_WELLS_PER_DECADE = 5
USE_CLASSES = (
    "Domestic", "Irrigation", "Agriculture (non irr)", "Public Water Supply", "Industrial",
    "Commercial", "Mining", "Observation Well", "De-Watering", "Recreation, Fish, Wildlife",
    "Institutional", "Water Location",
)
REPORT_LINK_NOTE = ("OWRB's online well log viewer is currently disabled, so report_url may not "
                    "open. The fields here come from a saved copy of the report.")


class ToolInputError(ValueError):
    """A bad argument. The message says what was wrong and how to fix the call."""


# Validation ------------------------------------------------------------------------

def _check_location(latitude: float, longitude: float) -> None:
    if not (OK_LAT[0] <= latitude <= OK_LAT[1] and OK_LON[0] <= longitude <= OK_LON[1]):
        raise ToolInputError(
            f"({latitude}, {longitude}) is outside Oklahoma. Use decimal degrees inside "
            f"latitude {OK_LAT[0]} to {OK_LAT[1]} and longitude {OK_LON[0]} to {OK_LON[1]}, "
            "for example latitude=35.47, longitude=-97.52 (Oklahoma City).")


def _check_radius(radius_miles: float, name: str = "radius_miles") -> None:
    if not (0 < radius_miles <= MAX_RADIUS_MILES):
        raise ToolInputError(
            f"{name}={radius_miles} is out of range. Use a value above 0 and at most "
            f"{MAX_RADIUS_MILES} miles; customers usually ask about 0.5 or 5 miles.")


def _check_use_class(use_class: str | None) -> None:
    if use_class is not None and use_class not in USE_CLASSES:
        raise ToolInputError(
            f"use_class={use_class!r} is not a well use in the records. Use one of: "
            f"{', '.join(USE_CLASSES)}; or omit it for all uses. A home well is 'Domestic'.")


def _iso(value):
    return value.isoformat() if isinstance(value, (dt.date, dt.datetime)) else value


# Shared SQL ------------------------------------------------------------------------

_NEARBY = """
WITH origin AS (SELECT ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), 4326)::geography AS g),
nearby AS (
    SELECT w.*, ST_Distance(w.geom, origin.g) AS dist_m
    FROM wells w, origin
    WHERE ST_DWithin(w.geom, origin.g, %(meters)s)
      AND w.work_type = 'Groundwater Well'
      AND (%(use_class)s::text IS NULL OR w.use_class = %(use_class)s::text)
)
"""


def _params(latitude, longitude, radius_miles, use_class):
    return {"lat": latitude, "lon": longitude, "meters": radius_miles * METERS_PER_MILE,
            "use_class": use_class}


# Tools -----------------------------------------------------------------------------

def summarize_area(conn: psycopg.Connection, latitude: float, longitude: float,
                   radius_miles: float = 0.5, use_class: str | None = None) -> dict:
    """Summarize groundwater wells within radius_miles of a point.

    Use for questions like "how deep are wells near me", "what yields do nearby wells get",
    "where are wells screened", or "have nearby wells been plugged".

    Returns well_count, nearest_well_ids (up to 10, closest first, for citing), and for each
    measure the number of wells that report it: depth_ft, first_water_ft, yield_gpm (with
    share_at_least_10_gpm), screened_interval_ft (each well's shallowest top and deepest
    bottom) and plugged (count and most recent date). Medians are None when no well reports
    the measure.
    """
    _check_location(latitude, longitude)
    _check_radius(radius_miles)
    _check_use_class(use_class)
    p = _params(latitude, longitude, radius_miles, use_class)

    stats = conn.execute(_NEARBY + """
        SELECT count(*),
               count(total_depth_ft), percentile_cont(0.5) WITHIN GROUP (ORDER BY total_depth_ft),
               count(first_water_ft), percentile_cont(0.5) WITHIN GROUP (ORDER BY first_water_ft),
               count(approx_yield_gpm),
               percentile_cont(0.5) WITHIN GROUP (ORDER BY approx_yield_gpm),
               avg(CASE WHEN approx_yield_gpm >= 10 THEN 1.0 ELSE 0.0 END)
                   FILTER (WHERE approx_yield_gpm IS NOT NULL)
        FROM nearby""", p).fetchone()
    nearest = conn.execute(_NEARBY + """
        SELECT coalesce(array_agg(well_id ORDER BY dist_m, well_id), '{}')
        FROM (SELECT well_id, dist_m FROM nearby ORDER BY dist_m, well_id LIMIT %(n)s) t""",
        {**p, "n": NEAREST_IDS}).fetchone()[0]
    screens = conn.execute(_NEARBY + """
        , spans AS (
            SELECT s.well_id, min(s.top_ft) AS top_ft, max(s.bottom_ft) AS bottom_ft
            FROM screens s JOIN nearby USING (well_id) GROUP BY s.well_id)
        SELECT count(*), percentile_cont(0.5) WITHIN GROUP (ORDER BY top_ft),
               percentile_cont(0.5) WITHIN GROUP (ORDER BY bottom_ft)
        FROM spans""", p).fetchone()
    plugged = conn.execute(_NEARBY + """
        SELECT count(*), max(plugged_date) FROM pluggings JOIN nearby USING (well_id)""",
        p).fetchone()

    n, n_depth, depth, n_fw, fw, n_yield, med_yield, share = stats
    return {
        "latitude": latitude, "longitude": longitude, "radius_miles": radius_miles,
        "use_class": use_class,
        "well_count": n,
        "nearest_well_ids": list(nearest),
        "depth_ft": {"wells": n_depth, "median": depth},
        "first_water_ft": {"wells": n_fw, "median": fw},
        "yield_gpm": {"wells": n_yield, "median": med_yield,
                      "share_at_least_10_gpm": None if share is None else float(share)},
        "screened_interval_ft": {"wells": screens[0], "median_top": screens[1],
                                 "median_bottom": screens[2]},
        "plugged": {"wells": plugged[0], "most_recent_date": _iso(plugged[1])},
    }


def find_nearest_pump_test(conn: psycopg.Connection, latitude: float, longitude: float,
                           use_class: str | None = None, max_miles: float = 25) -> dict:
    """Find the closest groundwater well with a pump test, up to max_miles away.

    Use for "what did a pump test near me show". Only about 10% of wells have a pump test,
    so always report distance_miles. Tests whose pumping water level is shallower than the
    starting level (negative drawdown, about 5%) are skipped as mistyped. Returns found=True with well_id, distance_miles,
    rate_gpm, duration (unit not stated on the reports), start_depth_to_water_ft,
    pumping_depth_to_water_ft and drawdown_ft; or found=False with searched_miles.
    """
    _check_location(latitude, longitude)
    _check_radius(max_miles, "max_miles")
    _check_use_class(use_class)
    p = _params(latitude, longitude, max_miles, use_class)
    row = conn.execute(_NEARBY + """
        SELECT n.well_id, n.dist_m / %(mile)s, t.rate_gpm, t.duration,
               t.start_depth_to_water_ft, t.pumping_depth_to_water_ft, t.drawdown_ft,
               n.total_depth_ft, n.use_class, n.construction_date
        FROM nearby n JOIN pump_tests t USING (well_id)
        WHERE t.drawdown_ft IS NULL OR t.drawdown_ft >= 0  -- negative means mistyped values
        ORDER BY n.dist_m, n.well_id
        LIMIT 1""", {**p, "mile": METERS_PER_MILE}).fetchone()
    if row is None:
        return {"found": False, "searched_miles": max_miles}
    keys = ["well_id", "distance_miles", "rate_gpm", "duration", "start_depth_to_water_ft",
            "pumping_depth_to_water_ft", "drawdown_ft", "total_depth_ft", "use_class",
            "construction_date"]
    return {"found": True, **{k: _iso(v) for k, v in zip(keys, row)}}


def get_water_level_trend(conn: psycopg.Connection, latitude: float, longitude: float,
                          radius_miles: float = 5, use_class: str | None = None) -> dict:
    """Median depth to water at the time of drilling, by the decade wells were drilled.

    Use for "is the water table dropping near me". A deeper median in later decades
    suggests a falling water table, but each well was measured once, when it was drilled.
    Returns decades (oldest first) with wells, median_depth_to_water_ft, change_ft versus
    the previous decade that has data, and few_wells=True when fewer than 5 wells.
    """
    _check_location(latitude, longitude)
    _check_radius(radius_miles)
    _check_use_class(use_class)
    rows = conn.execute(_NEARBY + """
        , measured AS (
            SELECT (extract(year FROM n.construction_date)::int / 10) * 10 AS decade,
                   r.depth_to_water_ft
            FROM nearby n JOIN report_fields r USING (well_id)
            WHERE r.depth_to_water_ft IS NOT NULL AND n.construction_date IS NOT NULL),
        by_decade AS (
            SELECT decade, count(*) AS wells,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY depth_to_water_ft) AS median_ft
            FROM measured GROUP BY decade)
        SELECT decade, wells, median_ft,
               median_ft - LAG(median_ft) OVER (ORDER BY decade) AS change_ft
        FROM by_decade ORDER BY decade""",
        _params(latitude, longitude, radius_miles, use_class)).fetchall()
    return {
        "latitude": latitude, "longitude": longitude, "radius_miles": radius_miles,
        "use_class": use_class,
        "decades": [
            {"decade": decade, "wells": wells, "median_depth_to_water_ft": median,
             "change_ft": change, "few_wells": wells < MIN_WELLS_PER_DECADE}
            for decade, wells, median, change in rows
        ],
    }


def get_well_record(conn: psycopg.Connection, well_id: int) -> dict:
    """Everything known about one well: location, construction, screens, lithology, pump
    test, plugging, depth to water at drilling and the driller's comments.

    Use when the user names a well ID, or to explain a report in plain language. Never
    includes owner names or addresses.
    """
    if not isinstance(well_id, int) or isinstance(well_id, bool) or well_id <= 0:
        raise ToolInputError(f"well_id={well_id!r} is not valid. Use a positive whole number like 239950.")
    row = conn.execute("""
        SELECT w.well_id, w.latitude, w.longitude, w.county, w.work_type, w.use_class,
               w.plss_section, w.plss_township, w.plss_range, w.location_method,
               w.construction_date, w.total_depth_ft, w.first_water_ft, w.approx_yield_gpm,
               r.depth_to_water_ft, pl.plugged_date, c.comment, w.report_url
        FROM wells w
        LEFT JOIN report_fields r USING (well_id)
        LEFT JOIN pluggings pl USING (well_id)
        LEFT JOIN comments c USING (well_id)
        WHERE w.well_id = %s""", (well_id,)).fetchone()
    if row is None:
        raise ToolInputError(
            f"No well with well_id {well_id}. Check the number, or call summarize_area "
            "with a location to get nearby well IDs.")
    keys = ["well_id", "latitude", "longitude", "county", "work_type", "use_class",
            "plss_section", "plss_township", "plss_range", "location_method",
            "construction_date", "total_depth_ft", "first_water_ft", "approx_yield_gpm",
            "depth_to_water_ft", "plugged_date", "comments", "report_url"]
    record = {k: _iso(v) for k, v in zip(keys, row)}

    record["screens"] = [
        {"layer": layer, "top_ft": top, "bottom_ft": bottom}
        for layer, top, bottom in conn.execute(
            "SELECT layer, top_ft, bottom_ft FROM screens WHERE well_id = %s ORDER BY layer",
            (well_id,))]
    record["lithology"] = [
        {"seq": seq, "material": material, "top_ft": top, "bottom_ft": bottom, "saturated": sat}
        for seq, material, top, bottom, sat in conn.execute(
            "SELECT seq, material, top_ft, bottom_ft, saturated FROM lithology "
            "WHERE well_id = %s ORDER BY seq", (well_id,))]
    test = conn.execute(
        "SELECT rate_gpm, duration, start_depth_to_water_ft, pumping_depth_to_water_ft, "
        "drawdown_ft FROM pump_tests WHERE well_id = %s", (well_id,)).fetchone()
    record["pump_test"] = None if test is None else dict(zip(
        ["rate_gpm", "duration", "start_depth_to_water_ft", "pumping_depth_to_water_ft",
         "drawdown_ft"], test))
    if record["pump_test"] and (record["pump_test"]["drawdown_ft"] or 0) < 0:
        record["pump_test"]["drawdown_ft"] = None
        record["pump_test"]["data_issue"] = (
            "The pumping water level is shallower than the starting level, so these values "
            "were probably swapped or mistyped. Don't report a drawdown for this test.")
    record["report_link_note"] = REPORT_LINK_NOTE
    return record

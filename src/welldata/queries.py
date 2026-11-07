"""Parameterized query functions the assistant will call.

The LLM never writes SQL: it picks one of these functions and fills in the
arguments, and every answer carries the number of wells it rests on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import psycopg

from welldata.quality import OK_LAT, OK_LON

METERS_PER_MILE = 1609.344
MAX_RADIUS_MILES = 25


@dataclass
class RadiusSummary:
    latitude: float
    longitude: float
    radius_miles: float
    use_class: str | None
    well_count: int
    wells_with_depth: int
    median_depth_ft: float | None
    wells_with_first_water: int
    median_first_water_ft: float | None
    well_ids: list[int] = field(default_factory=list)


_RADIUS_SQL = """
WITH nearby AS (
    SELECT well_id, total_depth_ft, first_water_ft
    FROM wells
    WHERE ST_DWithin(
            geom,
            ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), 4326)::geography,
            %(meters)s)
      AND work_type = 'Groundwater Well'
      AND (%(use_class)s::text IS NULL OR use_class = %(use_class)s::text)
)
SELECT
    count(*),
    count(total_depth_ft),
    percentile_cont(0.5) WITHIN GROUP (ORDER BY total_depth_ft),
    count(first_water_ft),
    percentile_cont(0.5) WITHIN GROUP (ORDER BY first_water_ft),
    coalesce(array_agg(well_id ORDER BY well_id), '{}')
FROM nearby
"""


def _check_location(lat: float, lon: float, radius_miles: float) -> None:
    if not (OK_LAT[0] <= lat <= OK_LAT[1] and OK_LON[0] <= lon <= OK_LON[1]):
        raise ValueError(f"({lat}, {lon}) is outside Oklahoma.")
    if not (0 < radius_miles <= MAX_RADIUS_MILES):
        raise ValueError(f"Radius must be between 0 and {MAX_RADIUS_MILES} miles.")


def wells_within_radius(
    conn: psycopg.Connection,
    lat: float,
    lon: float,
    radius_miles: float,
    use_class: str | None = None,
) -> RadiusSummary:
    """Depth and first water for groundwater wells within a radius (test holes excluded)."""
    _check_location(lat, lon, radius_miles)
    params = {"lat": lat, "lon": lon, "meters": radius_miles * METERS_PER_MILE, "use_class": use_class}
    n, n_depth, med_depth, n_fw, med_fw, ids = conn.execute(_RADIUS_SQL, params).fetchone()
    return RadiusSummary(
        latitude=lat,
        longitude=lon,
        radius_miles=radius_miles,
        use_class=use_class,
        well_count=n,
        wells_with_depth=n_depth,
        median_depth_ft=med_depth,
        wells_with_first_water=n_fw,
        median_first_water_ft=med_fw,
        well_ids=list(ids),
    )

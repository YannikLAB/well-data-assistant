"""Shared fixtures: synthetic wells (no real records) and a throwaway database schema."""

from __future__ import annotations

import datetime as dt
import os
import uuid

import pandas as pd
import pytest

MILE_IN_LAT_DEGREES = 1 / 69.05  # close enough for test offsets
CENTER = (35.5, -97.5)


def make_wells(rows: list[dict]) -> pd.DataFrame:
    """Build a wells table in the loader's tidy format; unspecified fields are blank."""
    template = {
        "well_id": 0, "latitude": CENTER[0], "longitude": CENTER[1], "county": "Oklahoma",
        "work_type": "Groundwater Well", "use_class": "Domestic", "plss_section": "1",
        "plss_township": "12N", "plss_range": "03WI", "total_depth_ft": float("nan"),
        "first_water_ft": float("nan"), "approx_yield_gpm": float("nan"),
        "location_method": "GPS-corrected data (WASS)", "construction_date": dt.date(2015, 6, 1),
        "report_url": None,
    }
    df = pd.DataFrame([{**template, **r} for r in rows])
    df["construction_date"] = df["construction_date"].astype(object)
    return df


def north(miles: float) -> float:
    return CENTER[0] + miles * MILE_IN_LAT_DEGREES


@pytest.fixture
def sample_wells() -> pd.DataFrame:
    return make_wells([
        {"well_id": 1, "latitude": north(0.3), "total_depth_ft": 200, "first_water_ft": 80},
        {"well_id": 2, "latitude": north(0.8), "total_depth_ft": 300, "first_water_ft": 120},
        {"well_id": 3, "latitude": north(3.0), "use_class": "Irrigation",
         "total_depth_ft": 400, "first_water_ft": 150},
        {"well_id": 4, "latitude": north(6.0), "total_depth_ft": 500, "first_water_ft": 200},
        {"well_id": 5, "latitude": north(0.2), "work_type": "Groundwater Test Hole",
         "total_depth_ft": 150},
        {"well_id": 6, "latitude": north(-0.4)},
    ])


@pytest.fixture
def db_conn():
    """A connection whose search_path points at a fresh schema, dropped afterwards.

    Skips when no database is reachable, unless REQUIRE_DB=1 (as in CI).
    """
    psycopg = pytest.importorskip("psycopg")
    from welldata import config

    try:
        conn = psycopg.connect(config.database_url(), autocommit=True, connect_timeout=3)
    except psycopg.OperationalError as err:
        if os.environ.get("REQUIRE_DB") == "1":
            raise
        pytest.skip(f"No database reachable at DATABASE_URL ({err.__class__.__name__}).")
    schema = f"test_{uuid.uuid4().hex[:8]}"
    conn.execute(f"CREATE SCHEMA {schema}")
    conn.execute(f"SET search_path TO {schema}, public")
    try:
        yield conn, schema
    finally:
        conn.execute(f"DROP SCHEMA {schema} CASCADE")
        conn.close()

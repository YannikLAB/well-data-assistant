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
        {"well_id": 1, "latitude": north(0.3), "total_depth_ft": 200, "first_water_ft": 80,
         "approx_yield_gpm": 20, "construction_date": dt.date(1985, 5, 1)},
        {"well_id": 2, "latitude": north(0.8), "total_depth_ft": 300, "first_water_ft": 120,
         "approx_yield_gpm": 5, "construction_date": dt.date(1995, 3, 1)},
        {"well_id": 3, "latitude": north(3.0), "use_class": "Irrigation",
         "total_depth_ft": 400, "first_water_ft": 150, "approx_yield_gpm": 300,
         "construction_date": dt.date(2005, 7, 1)},
        {"well_id": 4, "latitude": north(6.0), "total_depth_ft": 500, "first_water_ft": 200,
         "approx_yield_gpm": 15},
        {"well_id": 5, "latitude": north(0.2), "work_type": "Groundwater Test Hole",
         "total_depth_ft": 150},
        {"well_id": 6, "latitude": north(-0.4), "construction_date": dt.date(2021, 1, 1)},
    ])


@pytest.fixture
def sample_tables(sample_wells) -> dict[str, pd.DataFrame]:
    """Every table the loader writes, filled with synthetic rows that match sample_wells."""
    return {
        "wells": sample_wells,
        "screens": pd.DataFrame(
            [(1, 1, 150.0, 200.0), (2, 1, 220.0, 260.0), (2, 2, 280.0, 300.0), (3, 1, 300.0, 400.0)],
            columns=["well_id", "layer", "top_ft", "bottom_ft"]),
        "pluggings": pd.DataFrame([(2, dt.date(2020, 1, 15))], columns=["well_id", "plugged_date"]),
        "pump_tests": pd.DataFrame(
            [(2, 50.0, 90.0, 10.0, 2.0), (3, 60.0, 100.0, 300.0, 24.0)],
            columns=["well_id", "start_depth_to_water_ft", "pumping_depth_to_water_ft",
                     "rate_gpm", "duration"]),
        "report_fields": pd.DataFrame(
            [(1, 70.0), (2, 90.0), (3, 110.0), (4, 150.0), (6, 60.0)],
            columns=["well_id", "depth_to_water_ft"]),
        "lithology": pd.DataFrame(
            [(1, 1, "CLAY", 0.0, 20.0, False), (1, 2, "SAND", 20.0, 200.0, True)],
            columns=["well_id", "seq", "material", "top_ft", "bottom_ft", "saturated"]),
        "comments": pd.DataFrame([(1, "TDS 450")], columns=["well_id", "comment"]),
        "data_issues": pd.DataFrame(columns=["well_id", "field", "check_name", "value"]),
    }


@pytest.fixture
def loaded_db(db_conn, sample_tables):
    """A throwaway schema loaded with the synthetic tables; yields the connection."""
    from welldata.db import load_all

    conn, schema = db_conn
    load_all(conn, sample_tables, schema=schema)
    return conn


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

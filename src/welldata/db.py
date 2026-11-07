"""Create the tables and copy the cleaned data into PostGIS."""

from __future__ import annotations

from importlib.resources import files

import pandas as pd
import psycopg
from psycopg import sql


def _rows(df: pd.DataFrame, columns: list[str]):
    for record in df[columns].itertuples(index=False, name=None):
        yield tuple(None if (v is None or (not isinstance(v, str) and pd.isna(v))) else v for v in record)


def _copy(conn: psycopg.Connection, schema: str, table: str, df: pd.DataFrame, columns: list[str]) -> None:
    stmt = sql.SQL("COPY {}.{} ({}) FROM STDIN").format(
        sql.Identifier(schema), sql.Identifier(table), sql.SQL(", ").join(map(sql.Identifier, columns))
    )
    with conn.cursor() as cur, cur.copy(stmt) as copy:
        for row in _rows(df, columns):
            copy.write_row(row)


def create_tables(conn: psycopg.Connection, schema: str = "public") -> None:
    text = files("welldata").joinpath("sql/schema.sql").read_text()
    conn.execute(sql.SQL(text).format(schema=sql.Identifier(schema)))


def load_all(
    conn: psycopg.Connection,
    wells: pd.DataFrame,
    screens: pd.DataFrame,
    pluggings: pd.DataFrame,
    pump_tests: pd.DataFrame,
    issues: pd.DataFrame,
    schema: str = "public",
) -> dict[str, int]:
    """Replace all tables in `schema` with the given data. Returns row counts."""
    create_tables(conn, schema)
    known = set(wells["well_id"])
    screens = screens[screens["well_id"].isin(known)]
    pluggings = pluggings[pluggings["well_id"].isin(known)].drop_duplicates("well_id")
    pump_tests = pump_tests[pump_tests["well_id"].isin(known)].drop_duplicates("well_id")

    well_cols = [c for c in wells.columns if c != "geom"]
    _copy(conn, schema, "wells", wells, well_cols)
    _copy(conn, schema, "screens", screens, ["well_id", "layer", "top_ft", "bottom_ft"])
    _copy(conn, schema, "pluggings", pluggings, ["well_id", "plugged_date"])
    _copy(conn, schema, "pump_tests", pump_tests,
          ["well_id", "start_depth_to_water_ft", "pumping_depth_to_water_ft", "rate_gpm", "duration"])
    _copy(conn, schema, "data_issues", issues, ["well_id", "field", "check_name", "value"])

    s = sql.Identifier(schema)
    conn.execute(sql.SQL(
        "UPDATE {}.wells SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography"
    ).format(s))
    conn.execute(sql.SQL("CREATE INDEX ON {}.wells USING gist (geom)").format(s))
    conn.execute(sql.SQL("CREATE INDEX ON {}.wells (county, use_class)").format(s))
    conn.execute(sql.SQL("ANALYZE {}.wells").format(s))
    return {
        "wells": len(wells),
        "screens": len(screens),
        "pluggings": len(pluggings),
        "pump_tests": len(pump_tests),
        "data_issues": len(issues),
    }

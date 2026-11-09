"""Create the tables and copy the cleaned data into PostGIS."""

from __future__ import annotations

from importlib.resources import files

import pandas as pd
import psycopg
from psycopg import sql

# Load order matters: wells first, because the other tables reference it.
TABLE_COLUMNS = {
    "wells": ["well_id", "latitude", "longitude", "county", "work_type", "use_class",
              "plss_section", "plss_township", "plss_range", "total_depth_ft", "first_water_ft",
              "approx_yield_gpm", "location_method", "construction_date", "report_url"],
    "screens": ["well_id", "layer", "top_ft", "bottom_ft"],
    "pluggings": ["well_id", "plugged_date"],
    "pump_tests": ["well_id", "start_depth_to_water_ft", "pumping_depth_to_water_ft",
                   "rate_gpm", "duration"],
    "report_fields": ["well_id", "depth_to_water_ft"],
    "lithology": ["well_id", "seq", "material", "top_ft", "bottom_ft", "saturated"],
    "comments": ["well_id", "comment"],
    "data_issues": ["well_id", "field", "check_name", "value"],
}
# Tables with one row per well at most.
ONE_PER_WELL = {"pluggings", "pump_tests", "report_fields", "comments"}


def _rows(df: pd.DataFrame, columns: list[str]):
    for record in df[columns].itertuples(index=False, name=None):
        yield tuple(None if (v is None or (not isinstance(v, str) and pd.isna(v))) else v
                    for v in record)


def _copy(conn: psycopg.Connection, schema: str, table: str, df: pd.DataFrame) -> None:
    columns = TABLE_COLUMNS[table]
    stmt = sql.SQL("COPY {}.{} ({}) FROM STDIN").format(
        sql.Identifier(schema), sql.Identifier(table),
        sql.SQL(", ").join(map(sql.Identifier, columns)))
    with conn.cursor() as cur, cur.copy(stmt) as copy:
        for row in _rows(df, columns):
            copy.write_row(row)


def create_tables(conn: psycopg.Connection, schema: str = "public") -> None:
    text = files("welldata").joinpath("sql/schema.sql").read_text()
    conn.execute(sql.SQL(text).format(schema=sql.Identifier(schema)))


def align_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Every table present, child rows only for known wells, one row per well where required.

    The loader and the evaluation's gold answers both use this, so they see the same data.
    """
    known = set(tables["wells"]["well_id"])
    aligned = {}
    for name, columns in TABLE_COLUMNS.items():
        df = tables.get(name, pd.DataFrame(columns=columns))
        if name not in ("wells", "data_issues"):
            df = df[df["well_id"].isin(known)]
        if name in ONE_PER_WELL:
            df = df.drop_duplicates("well_id")
        aligned[name] = df.reset_index(drop=True)
    return aligned


def load_all(conn: psycopg.Connection, tables: dict[str, pd.DataFrame],
             schema: str = "public") -> dict[str, int]:
    """Replace all tables in `schema`. Missing tables load empty. Returns row counts."""
    create_tables(conn, schema)
    counts = {}
    for name, df in align_tables(tables).items():
        _copy(conn, schema, name, df)
        counts[name] = len(df)

    s = sql.Identifier(schema)
    conn.execute(sql.SQL(
        "UPDATE {}.wells SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography"
    ).format(s))
    conn.execute(sql.SQL("CREATE INDEX ON {}.wells USING gist (geom)").format(s))
    conn.execute(sql.SQL("CREATE INDEX ON {}.wells (county, use_class)").format(s))
    conn.execute(sql.SQL("ANALYZE {}.wells").format(s))
    return counts

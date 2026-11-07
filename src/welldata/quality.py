"""Data-quality checks for the well table, written as a pandera schema.

`find_issues` reports every failing value without stopping the load.
`clean_wells` then removes the values that are clearly impossible, so they never
reach an answer, and drops rows that can't be placed in Oklahoma.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

# Oklahoma's bounding box (a state outline would be stricter; see README).
OK_LAT = (33.6, 37.01)
OK_LON = (-103.01, -94.43)
MAX_DEPTH_FT = 9999  # 9999 is used as a placeholder, so it fails too
MAX_YIELD_GPM = 5000
EARLIEST_DATE = dt.date(1890, 1, 1)

# Checks whose failing values are blanked out before loading.
BLANK_IF_FAILED = {"total_depth_ft", "first_water_ft", "approx_yield_gpm", "construction_date"}
# Checks whose failing rows are left out of the wells table.
DROP_IF_FAILED = {"well_id", "latitude", "longitude"}


def well_schema(as_of: dt.date) -> pa.DataFrameSchema:
    def date_in_range(d: dt.date) -> bool:
        return EARLIEST_DATE <= d <= as_of

    return pa.DataFrameSchema(
        {
            "well_id": pa.Column(int, pa.Check.gt(0), unique=True),
            "latitude": pa.Column(float, pa.Check.in_range(*OK_LAT), nullable=False),
            "longitude": pa.Column(float, pa.Check.in_range(*OK_LON), nullable=False),
            "total_depth_ft": pa.Column(
                float, [pa.Check.gt(0), pa.Check.lt(MAX_DEPTH_FT)], nullable=True
            ),
            "first_water_ft": pa.Column(float, pa.Check.ge(0), nullable=True),
            "approx_yield_gpm": pa.Column(
                float, [pa.Check.ge(0), pa.Check.le(MAX_YIELD_GPM)], nullable=True
            ),
            "construction_date": pa.Column(
                object,
                pa.Check(date_in_range, element_wise=True, name="date_between_1890_and_as_of"),
                nullable=True,
            ),
        },
        checks=[
            pa.Check(
                lambda df: df["first_water_ft"].isna()
                | df["total_depth_ft"].isna()
                | (df["first_water_ft"] <= df["total_depth_ft"]),
                name="first_water_not_below_total_depth",
            )
        ],
        strict=False,
    )


def find_issues(wells: pd.DataFrame, as_of: dt.date) -> pd.DataFrame:
    """One row per failing value: well_id, field, check_name, value."""
    columns = ["well_id", "field", "check_name", "value"]
    try:
        well_schema(as_of).validate(wells, lazy=True)
        return pd.DataFrame(columns=columns)
    except SchemaErrors as err:
        fc = err.failure_cases
    fc = fc[fc["index"].notna()].copy()
    fc["index"] = fc["index"].astype(int)
    issues = pd.DataFrame(
        {
            "well_id": wells.loc[fc["index"], "well_id"].to_numpy(),
            "field": fc["column"].fillna("first_water_ft").to_numpy(),
            "check_name": fc["check"].astype(str).to_numpy(),
            "value": fc["failure_case"].astype(str).to_numpy(),
            "_row": fc["index"].to_numpy(),
        }
    )
    # Dataframe-level checks report each column of a failing row; keep one row per failure.
    issues = issues.drop_duplicates(subset=["_row", "field", "check_name"])
    issues = issues[~((issues["check_name"] == "first_water_not_below_total_depth")
                      & (issues["field"] != "first_water_ft"))]
    return issues.sort_values(["field", "well_id"]).reset_index(drop=True)


def clean_wells(wells: pd.DataFrame, issues: pd.DataFrame) -> pd.DataFrame:
    """Blank impossible values and drop rows that can't be placed or identified."""
    df = wells.copy()
    for field in BLANK_IF_FAILED:
        rows = issues.loc[
            (issues["field"] == field) & (issues["check_name"] != "first_water_not_below_total_depth"),
            "_row",
        ]
        if len(rows):
            df.loc[rows, field] = None
    # Duplicate IDs are all reported, but the first copy is kept by drop_duplicates below.
    unplaceable = issues["field"].isin(DROP_IF_FAILED) & (issues["check_name"] != "field_uniqueness")
    df = df.drop(index=issues.loc[unplaceable, "_row"].unique())
    return df.drop_duplicates(subset="well_id", keep="first").reset_index(drop=True)

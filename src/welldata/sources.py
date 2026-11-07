"""Read the Oklahoma workbook and the extraction output into tidy tables.

Owner names and addresses are never read, so they cannot reach the database,
the test fixtures or the repo.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

# Source column -> our column. owner_name is deliberately not listed.
WELL_COLUMNS = {
    "well_id": "well_id",
    "latitude": "latitude",
    "longitude": "longitude",
    "county": "county",
    "work_type": "work_type",
    "use_class": "use_class",
    "section": "plss_section",
    "township": "plss_township",
    "range": "plss_range",
    "total_dpth": "total_depth_ft",
    "first_wtr": "first_water_ft",
    "approx_yld": "approx_yield_gpm",
    "ll_method": "location_method",
    "const_date": "construction_date",
    "Url": "report_url",
}
WELLS_SHEET = "Reported_Well_Logs"
NUMERIC = ["latitude", "longitude", "total_depth_ft", "first_water_ft", "approx_yield_gpm"]
TEXT = ["county", "work_type", "use_class", "plss_section", "plss_township", "plss_range",
        "location_method", "report_url"]


def parse_state_date(value: object) -> dt.date | None:
    """Parse '2025/10/31 05:00:00+00'. Keeps impossible years like 9130 so checks can flag them."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (dt.datetime, pd.Timestamp)):
        return value.date()
    text = str(value).strip()[:10]
    try:
        return dt.datetime.strptime(text, "%Y/%m/%d").date()
    except ValueError:
        return None


def parse_us_date(value: object) -> dt.date | None:
    """Parse '07/17/2019' as written in the extraction output."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return dt.datetime.strptime(str(value).strip(), "%m/%d/%Y").date()
    except ValueError:
        return None


def tidy_wells(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep Useful = 1 rows and the columns we need, with consistent types."""
    rows = raw[pd.to_numeric(raw["Useful"], errors="coerce") == 1]
    df = rows[list(WELL_COLUMNS)].rename(columns=WELL_COLUMNS).copy()
    df["well_id"] = pd.to_numeric(df["well_id"], errors="raise").astype("int64")
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in TEXT:
        df[col] = df[col].map(lambda v: None if pd.isna(v) else str(v).strip()).astype(object)
    df["construction_date"] = df["construction_date"].map(parse_state_date).astype(object)
    return df.reset_index(drop=True)


def read_wells(path: Path) -> pd.DataFrame:
    """Read the state well table. Takes a couple of minutes for the full workbook."""
    raw = pd.read_excel(
        path, sheet_name=WELLS_SHEET, usecols=["Useful", *WELL_COLUMNS], engine="openpyxl"
    )
    return tidy_wells(raw)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    return pd.read_excel(path, engine="openpyxl")


def tidy_extracts(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the extraction output into screens, pluggings and pump tests."""
    raw = raw.copy()
    raw["well_id"] = pd.to_numeric(raw["well_id"], errors="raise").astype("int64")

    layers = []
    for k in range(1, 6):
        part = raw[["well_id", f"{k}) Start", f"{k}) End"]].copy()
        part.columns = ["well_id", "top_ft", "bottom_ft"]
        part = part[part["top_ft"].notna() | part["bottom_ft"].notna()]
        part.insert(1, "layer", k)
        layers.append(part)
    screens = pd.concat(layers, ignore_index=True).sort_values(["well_id", "layer"])
    screens = screens.astype({"top_ft": "float64", "bottom_ft": "float64"})

    plugged = raw[raw["Date Plugged"].notna()][["well_id", "Date Plugged"]]
    pluggings = pd.DataFrame(
        {"well_id": plugged["well_id"], "plugged_date": plugged["Date Plugged"].map(parse_us_date)}
    )

    tested = raw[raw["Rate(gpm)"].notna()]
    pump_tests = pd.DataFrame(
        {
            "well_id": tested["well_id"],
            # The report's columns: Start Depth, Rate(gpm), Duration, "Drawdown depth to water (ft)".
            # "Drawdown depth" is the depth to water while pumping, not the drawdown itself.
            "start_depth_to_water_ft": pd.to_numeric(tested["Start Depth"], errors="coerce"),
            "pumping_depth_to_water_ft": pd.to_numeric(tested["Drawdown depth"], errors="coerce"),
            "rate_gpm": pd.to_numeric(tested["Rate(gpm)"], errors="coerce"),
            "duration": pd.to_numeric(tested["Duration"], errors="coerce"),  # unit not stated
        }
    )
    return screens.reset_index(drop=True), pluggings.reset_index(drop=True), pump_tests.reset_index(drop=True)


def read_extracts(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return tidy_extracts(_read_table(path))

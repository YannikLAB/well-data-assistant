import datetime as dt

import pandas as pd

from conftest import make_wells
from welldata.quality import clean_wells, find_issues

AS_OF = dt.date(2026, 11, 7)


def test_clean_wells_have_no_issues(sample_wells):
    assert find_issues(sample_wells, AS_OF).empty


def test_known_problems_are_flagged_and_handled():
    wells = make_wells([
        {"well_id": 1, "total_depth_ft": 9999},
        {"well_id": 2, "total_depth_ft": -999},
        {"well_id": 3, "approx_yield_gpm": -15},
        {"well_id": 4, "approx_yield_gpm": 25000},
        {"well_id": 5, "construction_date": dt.date(9130, 1, 1)},
        {"well_id": 6, "latitude": 40.0},
        {"well_id": 7, "total_depth_ft": 100, "first_water_ft": 150},
        {"well_id": 8, "total_depth_ft": 250, "approx_yield_gpm": 20},
    ])
    issues = find_issues(wells, AS_OF)

    flagged = set(zip(issues["well_id"], issues["field"]))
    assert flagged == {
        (1, "total_depth_ft"), (2, "total_depth_ft"), (3, "approx_yield_gpm"),
        (4, "approx_yield_gpm"), (5, "construction_date"), (6, "latitude"),
        (7, "first_water_ft"),
    }

    cleaned = clean_wells(wells, issues).set_index("well_id")
    assert 6 not in cleaned.index  # can't be placed in Oklahoma
    assert pd.isna(cleaned.loc[1, "total_depth_ft"])
    assert pd.isna(cleaned.loc[4, "approx_yield_gpm"])
    assert cleaned.loc[5, "construction_date"] is None
    # First water below total depth is reported but kept: we can't tell which value is wrong.
    assert cleaned.loc[7, "first_water_ft"] == 150
    assert cleaned.loc[8, "approx_yield_gpm"] == 20


def test_duplicate_ids_keep_the_first_row():
    wells = make_wells([{"well_id": 1, "total_depth_ft": 100}, {"well_id": 1, "total_depth_ft": 200}])
    issues = find_issues(wells, AS_OF)
    assert (issues["field"] == "well_id").any()
    cleaned = clean_wells(wells, issues)
    assert cleaned["total_depth_ft"].tolist() == [100]

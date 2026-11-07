import datetime as dt

import pandas as pd

from welldata.sources import WELLS_SHEET, parse_state_date, read_wells, tidy_extracts


def source_row(**overrides):
    row = {
        "Useful": 1, "well_id": 10, "X": 0, "Y": 0, "objectid": 1, "latitude": 35.5,
        "longitude": -97.5, "elevation": None, "el_method": None, "county": "Creek",
        "work_type": "Groundwater Well", "permit": None, "owner_name": "Jane Example",
        "quarter1": "SW", "quarter2": "NW", "quarter3": "SE", "section": 1, "township": "17N",
        "range": "09EI", "use_class": "Domestic", "total_dpth": 270, "first_wtr": 53,
        "approx_yld": 25, "ll_method": "GPS-corrected data (WASS)",
        "const_date": "2025/10/31 05:00:00+00",
        "Url": "http://www.owrb.ok.gov/wd/reporting/printreport.php?siteid=10",
    }
    row.update(overrides)
    return row


def test_read_wells_never_reads_owner_names(tmp_path):
    path = tmp_path / "wells.xlsx"
    rows = [source_row(), source_row(well_id=11, Useful=None), source_row(well_id=12)]
    pd.DataFrame(rows).to_excel(path, sheet_name=WELLS_SHEET, index=False)

    wells = read_wells(path)

    assert "owner_name" not in wells.columns
    assert not wells.astype(str).apply(lambda col: col.str.contains("Jane Example")).any().any()
    assert list(wells["well_id"]) == [10, 12]  # Useful != 1 is skipped
    assert wells.loc[0, "construction_date"] == dt.date(2025, 10, 31)
    assert wells.loc[0, "plss_section"] == "1"


def test_impossible_years_are_kept_so_checks_can_flag_them():
    assert parse_state_date("9130/01/01 06:00:00+00") == dt.date(9130, 1, 1)
    assert parse_state_date("") is None
    assert parse_state_date(None) is None


def test_extracts_split_into_tables():
    raw = pd.DataFrame([
        {"well_id": 1, "1) Start": 53, "1) End": 66, "2) Start": 80, "2) End": 90,
         "3) Start": None, "3) End": None, "4) Start": None, "4) End": None,
         "5) Start": None, "5) End": None, "Date Plugged": None,
         "Start Depth": 34, "Rate(gpm)": 30, "Duration": 1, "Drawdown depth": 63},
        {"well_id": 2, "1) Start": None, "1) End": None, "2) Start": None, "2) End": None,
         "3) Start": None, "3) End": None, "4) Start": None, "4) End": None,
         "5) Start": None, "5) End": None, "Date Plugged": "07/17/2019",
         "Start Depth": None, "Rate(gpm)": None, "Duration": None, "Drawdown depth": None},
    ])
    screens, pluggings, pump_tests = tidy_extracts(raw)

    assert screens[["well_id", "layer", "top_ft", "bottom_ft"]].values.tolist() == [
        [1, 1, 53, 66], [1, 2, 80, 90]]
    assert pluggings.to_dict("records") == [{"well_id": 2, "plugged_date": dt.date(2019, 7, 17)}]
    test = pump_tests.iloc[0]
    assert test["pumping_depth_to_water_ft"] == 63  # "Drawdown depth" is a depth, not a drawdown
    assert test["start_depth_to_water_ft"] == 34

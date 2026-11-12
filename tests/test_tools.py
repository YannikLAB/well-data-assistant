"""Query tools the assistant calls. All run against synthetic wells in a throwaway schema.

Layout around CENTER (miles north): 1 at 0.3, 6 at -0.4, test hole 5 at 0.2,
2 at 0.8, irrigation well 3 at 3.0, and 4 at 6.0 (outside 5 miles).
"""

import pytest

from conftest import CENTER
from welldata.tools import (
    ToolInputError,
    find_nearest_pump_test,
    get_water_level_trend,
    get_well_record,
    summarize_area,
)

pytestmark = pytest.mark.db
LAT, LON = CENTER


# summarize_area ---------------------------------------------------------------

def test_area_half_mile_counts_groundwater_wells_only(loaded_db):
    area = summarize_area(loaded_db, LAT, LON, radius_miles=0.5)
    assert area["well_count"] == 2  # wells 1 and 6; test hole 5 excluded
    assert area["nearest_well_ids"] == [1, 6]
    assert area["depth_ft"] == {"wells": 1, "median": 200.0}
    assert area["first_water_ft"] == {"wells": 1, "median": 80.0}


def test_area_yield_reports_share_reaching_10_gpm(loaded_db):
    area = summarize_area(loaded_db, LAT, LON, radius_miles=5)
    assert area["yield_gpm"] == {"wells": 3, "median": 20.0, "share_at_least_10_gpm": pytest.approx(2 / 3)}


def test_area_screens_use_each_wells_full_screened_interval(loaded_db):
    area = summarize_area(loaded_db, LAT, LON, radius_miles=5)
    assert area["screened_interval_ft"] == {"wells": 3, "median_top": 220.0, "median_bottom": 300.0}


def test_area_lists_plugged_wells(loaded_db):
    area = summarize_area(loaded_db, LAT, LON, radius_miles=5)
    assert area["plugged"] == {"wells": 1, "most_recent_date": "2020-01-15"}


def test_area_filters_by_use_class(loaded_db):
    area = summarize_area(loaded_db, LAT, LON, radius_miles=5, use_class="Domestic")
    assert area["well_count"] == 3
    assert area["yield_gpm"]["share_at_least_10_gpm"] == 0.5


def test_area_with_no_wells_returns_zero_counts(loaded_db):
    area = summarize_area(loaded_db, 36.5, -99.5, radius_miles=0.5)
    assert area["well_count"] == 0
    assert area["depth_ft"] == {"wells": 0, "median": None}


# input errors tell the agent how to fix the call ---------------------------------

def test_location_outside_oklahoma_error_explains_fix(loaded_db):
    with pytest.raises(ToolInputError, match="outside Oklahoma"):
        summarize_area(loaded_db, 40.0, -97.5)


def test_radius_error_suggests_typical_values(loaded_db):
    with pytest.raises(ToolInputError, match="0.5 or 5"):
        summarize_area(loaded_db, LAT, LON, radius_miles=50)


def test_unknown_use_class_error_lists_valid_values(loaded_db):
    with pytest.raises(ToolInputError, match="Domestic"):
        summarize_area(loaded_db, LAT, LON, use_class="House")


# find_nearest_pump_test ------------------------------------------------------------

def test_nearest_pump_test_returns_distance_and_drawdown(loaded_db):
    test = find_nearest_pump_test(loaded_db, LAT, LON)
    assert test["found"] is True
    assert test["well_id"] == 2
    assert test["distance_miles"] == pytest.approx(0.8, abs=0.02)
    assert test["rate_gpm"] == 10.0
    assert test["drawdown_ft"] == 40.0


def test_nearest_pump_test_respects_use_class_and_max_distance(loaded_db):
    assert find_nearest_pump_test(loaded_db, LAT, LON, use_class="Irrigation")["well_id"] == 3
    missing = find_nearest_pump_test(loaded_db, LAT, LON, use_class="Irrigation", max_miles=1)
    assert missing == {"found": False, "searched_miles": 1}


def test_nearest_pump_test_skips_tests_with_negative_drawdown(loaded_db):
    # Well 1 (0.3 mi) gets a test whose pumping level is shallower than its start level.
    loaded_db.execute("INSERT INTO pump_tests (well_id, start_depth_to_water_ft, "
                      "pumping_depth_to_water_ft, rate_gpm, duration) VALUES (1, 80, 40, 12, 1)")
    assert find_nearest_pump_test(loaded_db, LAT, LON)["well_id"] == 2


def test_well_record_flags_negative_drawdown(loaded_db):
    loaded_db.execute("INSERT INTO pump_tests (well_id, start_depth_to_water_ft, "
                      "pumping_depth_to_water_ft, rate_gpm, duration) VALUES (1, 80, 40, 12, 1)")
    test = get_well_record(loaded_db, 1)["pump_test"]
    assert test["drawdown_ft"] is None
    assert "shallower" in test["data_issue"]


# get_water_level_trend ----------------------------------------------------------------

def test_trend_gives_median_by_decade_and_change(loaded_db):
    trend = get_water_level_trend(loaded_db, LAT, LON, radius_miles=5)
    assert [(d["decade"], d["wells"], d["median_depth_to_water_ft"], d["change_ft"])
            for d in trend["decades"]] == [
        (1980, 1, 70.0, None), (1990, 1, 90.0, 20.0), (2000, 1, 110.0, 20.0), (2020, 1, 60.0, -50.0)]


def test_trend_flags_decades_with_few_wells(loaded_db):
    trend = get_water_level_trend(loaded_db, LAT, LON, radius_miles=5)
    assert all(d["few_wells"] for d in trend["decades"])


# get_well_record ------------------------------------------------------------------------

def test_well_record_combines_all_tables_without_owner(loaded_db):
    record = get_well_record(loaded_db, 1)
    assert record["well_id"] == 1
    assert record["total_depth_ft"] == 200.0
    assert record["depth_to_water_ft"] == 70.0
    assert record["construction_date"] == "1985-05-01"
    assert record["screens"] == [{"layer": 1, "top_ft": 150.0, "bottom_ft": 200.0}]
    assert record["lithology"][1] == {"seq": 2, "material": "SAND", "top_ft": 20.0,
                                      "bottom_ft": 200.0, "saturated": True}
    assert record["comments"] == "TDS 450"
    assert record["plugged_date"] is None and record["pump_test"] is None
    assert "owner_name" not in record


def test_well_record_includes_pump_test_and_plugging(loaded_db):
    record = get_well_record(loaded_db, 2)
    assert record["plugged_date"] == "2020-01-15"
    assert record["pump_test"]["drawdown_ft"] == 40.0


def test_unknown_well_error_says_how_to_find_ids(loaded_db):
    with pytest.raises(ToolInputError, match="summarize_area"):
        get_well_record(loaded_db, 999)


def test_swapped_coordinates_get_a_hint_with_the_fix():
    with pytest.raises(ToolInputError, match=r"swapped.*latitude=35\.5, longitude=-97\.5"):
        summarize_area(None, -97.5, 35.5)


def test_a_missing_minus_sign_gets_a_hint_with_the_fix():
    with pytest.raises(ToolInputError, match=r"negative.*longitude=-97\.5"):
        summarize_area(None, 35.5, 97.5)

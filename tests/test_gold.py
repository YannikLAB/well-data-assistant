"""Reference answers for the evaluation set, computed with pandas instead of SQL.

The first tests pin the expected values on the synthetic layout (no database needed).
The cross-check tests confirm the SQL tools agree with this independent implementation.
"""

import pytest

from conftest import CENTER
from welldata import gold, tools

LAT, LON = CENTER


def test_gold_area_summary_matches_expected_values(sample_tables):
    area = gold.area_summary(sample_tables, LAT, LON, radius_miles=5)
    assert area["well_count"] == 4
    assert area["nearest_well_ids"] == [1, 6, 2, 3]
    assert area["depth_ft"] == {"wells": 3, "median": 300.0}
    assert area["yield_gpm"] == {"wells": 3, "median": 20.0, "share_at_least_10_gpm": pytest.approx(2 / 3)}
    assert area["screened_interval_ft"] == {"wells": 3, "median_top": 220.0, "median_bottom": 300.0}
    assert area["plugged"] == {"wells": 1, "most_recent_date": "2020-01-15"}


def test_gold_area_summary_filters_use_class(sample_tables):
    area = gold.area_summary(sample_tables, LAT, LON, radius_miles=5, use_class="Domestic")
    assert area["well_count"] == 3


def test_gold_nearest_pump_test(sample_tables):
    test = gold.nearest_pump_test(sample_tables, LAT, LON)
    assert test["well_id"] == 2 and test["drawdown_ft"] == 40.0
    assert gold.nearest_pump_test(sample_tables, LAT, LON, use_class="Irrigation", max_miles=1) == {
        "found": False, "searched_miles": 1}


def test_gold_trend(sample_tables):
    trend = gold.water_level_trend(sample_tables, LAT, LON, radius_miles=5)
    assert [(d["decade"], d["median_depth_to_water_ft"], d["change_ft"]) for d in trend["decades"]] == [
        (1980, 70.0, None), (1990, 90.0, 20.0), (2000, 110.0, 20.0), (2020, 60.0, -50.0)]


def test_gold_well_facts_list_saturated_layers(sample_tables):
    facts = gold.well_facts(sample_tables, 1)
    assert facts["total_depth_ft"] == 200.0
    assert facts["depth_to_water_ft"] == 70.0
    assert facts["screens"] == [{"layer": 1, "top_ft": 150.0, "bottom_ft": 200.0}]
    assert facts["saturated_layers"] == [{"material": "SAND", "top_ft": 20.0, "bottom_ft": 200.0}]
    assert facts["has_comment"] is True


# Cross-checks against the SQL tools ----------------------------------------------------

AREA_KEYS = ["well_count", "nearest_well_ids", "depth_ft", "first_water_ft", "yield_gpm",
             "screened_interval_ft", "plugged"]


@pytest.mark.db
@pytest.mark.parametrize("radius,use_class", [(0.5, None), (5, None), (5, "Domestic"), (1, "Irrigation")])
def test_tools_agree_with_gold_area(loaded_db, sample_tables, radius, use_class):
    expected = gold.area_summary(sample_tables, LAT, LON, radius, use_class)
    actual = tools.summarize_area(loaded_db, LAT, LON, radius, use_class)
    assert {k: actual[k] for k in AREA_KEYS} == expected


@pytest.mark.db
def test_tools_agree_with_gold_pump_test_and_trend(loaded_db, sample_tables):
    assert tools.find_nearest_pump_test(loaded_db, LAT, LON)["well_id"] == \
        gold.nearest_pump_test(sample_tables, LAT, LON)["well_id"]
    assert tools.get_water_level_trend(loaded_db, LAT, LON)["decades"] == \
        gold.water_level_trend(sample_tables, LAT, LON)["decades"]

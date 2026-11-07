import pandas as pd
import pytest

from conftest import CENTER
from welldata.db import load_all
from welldata.queries import wells_within_radius

pytestmark = pytest.mark.db

EMPTY = pd.DataFrame(columns=["well_id", "layer", "top_ft", "bottom_ft"])
NO_PLUGS = pd.DataFrame(columns=["well_id", "plugged_date"])
NO_TESTS = pd.DataFrame(columns=["well_id", "start_depth_to_water_ft", "pumping_depth_to_water_ft",
                                 "rate_gpm", "duration"])
NO_ISSUES = pd.DataFrame(columns=["well_id", "field", "check_name", "value"])


@pytest.fixture
def loaded(db_conn, sample_wells):
    conn, schema = db_conn
    load_all(conn, sample_wells, EMPTY, NO_PLUGS, NO_TESTS, NO_ISSUES, schema=schema)
    return conn


def test_half_mile_excludes_test_holes_and_far_wells(loaded):
    result = wells_within_radius(loaded, *CENTER, radius_miles=0.5)
    assert result.well_ids == [1, 6]
    assert result.well_count == 2
    assert result.wells_with_depth == 1  # well 6 has no depth
    assert result.median_depth_ft == 200


def test_five_miles_with_and_without_use_filter(loaded):
    everything = wells_within_radius(loaded, *CENTER, radius_miles=5)
    assert everything.well_ids == [1, 2, 3, 6]
    assert everything.median_depth_ft == 300
    assert everything.median_first_water_ft == 120

    domestic = wells_within_radius(loaded, *CENTER, radius_miles=5, use_class="Domestic")
    assert domestic.well_ids == [1, 2, 6]
    assert domestic.median_depth_ft == 250


def test_locations_outside_oklahoma_are_rejected(loaded):
    with pytest.raises(ValueError):
        wells_within_radius(loaded, 40.0, -97.5, radius_miles=1)

-- Tables for the well-data assistant. {schema} is filled in by welldata.db,
-- so tests can build the same tables in a throwaway schema.

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE SCHEMA IF NOT EXISTS {schema};

DROP TABLE IF EXISTS {schema}.comments, {schema}.lithology, {schema}.report_fields,
    {schema}.data_issues, {schema}.pump_tests, {schema}.pluggings, {schema}.screens,
    {schema}.wells CASCADE;

CREATE TABLE {schema}.wells (
    well_id            integer PRIMARY KEY,
    latitude           double precision NOT NULL,
    longitude          double precision NOT NULL,
    geom               geography(Point, 4326),
    county             text,
    work_type          text,
    use_class          text,
    plss_section       text,
    plss_township      text,
    plss_range         text,
    total_depth_ft     double precision,
    first_water_ft     double precision,
    approx_yield_gpm   double precision,
    location_method    text,
    construction_date  date,
    report_url         text
);

CREATE TABLE {schema}.screens (
    well_id    integer NOT NULL REFERENCES {schema}.wells,
    layer      smallint NOT NULL CHECK (layer BETWEEN 1 AND 5),
    top_ft     double precision,
    bottom_ft  double precision,
    PRIMARY KEY (well_id, layer)
);

CREATE TABLE {schema}.pluggings (
    well_id       integer PRIMARY KEY REFERENCES {schema}.wells,
    plugged_date  date
);

CREATE TABLE {schema}.pump_tests (
    well_id                    integer PRIMARY KEY REFERENCES {schema}.wells,
    start_depth_to_water_ft    double precision,
    pumping_depth_to_water_ft  double precision,  -- the report's "Drawdown depth to water"
    rate_gpm                   double precision,
    duration                   double precision,  -- unit not stated on the report
    drawdown_ft                double precision
        GENERATED ALWAYS AS (pumping_depth_to_water_ft - start_depth_to_water_ft) STORED
);

CREATE TABLE {schema}.data_issues (
    well_id     integer,
    field       text,
    check_name  text,
    value       text
);

-- Fields parsed from the saved reports by welldata.extract_reports.
CREATE TABLE {schema}.report_fields (
    well_id            integer PRIMARY KEY REFERENCES {schema}.wells,
    depth_to_water_ft  double precision  -- "Depth to water at time of drilling"
);

CREATE TABLE {schema}.lithology (
    well_id    integer NOT NULL REFERENCES {schema}.wells,
    seq        smallint NOT NULL,
    material   text,
    top_ft     double precision,
    bottom_ft  double precision,
    saturated  boolean,
    PRIMARY KEY (well_id, seq)
);

CREATE TABLE {schema}.comments (
    well_id  integer PRIMARY KEY REFERENCES {schema}.wells,
    comment  text NOT NULL
);

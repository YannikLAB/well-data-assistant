"""Load the Oklahoma well records into PostGIS.

Usage: python -m welldata.load [--as-of YYYY-MM-DD] [--issues-out reports/data_issues.csv]
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import psycopg

from welldata import config
from welldata.db import align_tables, load_all
from welldata.quality import clean_wells, find_issues
from welldata.sources import read_derived, read_extracts, read_wells


def prepare_tables(as_of: dt.date) -> dict:
    """Read and clean every source into the tables the database holds."""
    print(f"Reading {config.wells_workbook()} (takes a couple of minutes)...")
    raw = read_wells(config.wells_workbook())
    issues = find_issues(raw, as_of)
    wells = clean_wells(raw, issues)
    print(f"Reading {config.extracts_file()}...")
    screens, pluggings, pump_tests = read_extracts(config.extracts_file())
    derived = read_derived(config.derived_dir())
    if derived["report_fields"].empty:
        print("No extracted report fields found; run `python -m welldata.extract_reports` first "
              "to add depth to water, lithology and comments.")
    return align_tables({"wells": wells, "screens": screens, "pluggings": pluggings,
                         "pump_tests": pump_tests, "data_issues": issues, **derived})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--as-of", type=dt.date.fromisoformat, default=dt.date.today(),
                        help="Dates after this count as being in the future (default: today).")
    parser.add_argument("--issues-out", type=Path, default=Path("reports/data_issues.csv"))
    args = parser.parse_args(argv)

    tables = prepare_tables(args.as_of)
    issues = tables["data_issues"]
    with psycopg.connect(config.database_url()) as conn:
        counts = load_all(conn, tables)

    args.issues_out.parent.mkdir(parents=True, exist_ok=True)
    issues.drop(columns="_row", errors="ignore").to_csv(args.issues_out, index=False)

    print("Loaded:", ", ".join(f"{k} {v:,}" for k, v in counts.items()))
    print(f"Data issues ({len(issues):,}) by check, written to {args.issues_out}:")
    summary = issues.groupby(["field", "check_name"]).size()
    for (field, check), n in summary.items():
        print(f"  {field:20} {check:45} {n:>6,}")


if __name__ == "__main__":
    main()

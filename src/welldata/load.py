"""Load the Oklahoma well records into PostGIS.

Usage: python -m welldata.load [--as-of YYYY-MM-DD] [--issues-out reports/data_issues.csv]
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import psycopg

from welldata import config
from welldata.db import load_all
from welldata.quality import clean_wells, find_issues
from welldata.sources import read_extracts, read_wells


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--as-of", type=dt.date.fromisoformat, default=dt.date.today(),
                        help="Dates after this count as being in the future (default: today).")
    parser.add_argument("--issues-out", type=Path, default=Path("reports/data_issues.csv"))
    args = parser.parse_args(argv)

    print(f"Reading {config.wells_workbook()} (takes a couple of minutes)...")
    raw = read_wells(config.wells_workbook())
    issues = find_issues(raw, args.as_of)
    wells = clean_wells(raw, issues)
    print(f"Reading {config.extracts_file()}...")
    screens, pluggings, pump_tests = read_extracts(config.extracts_file())

    with psycopg.connect(config.database_url()) as conn:
        counts = load_all(conn, wells, screens, pluggings, pump_tests, issues)

    args.issues_out.parent.mkdir(parents=True, exist_ok=True)
    issues.drop(columns="_row", errors="ignore").to_csv(args.issues_out, index=False)

    print("Loaded:", ", ".join(f"{k} {v:,}" for k, v in counts.items()))
    print(f"Data issues ({len(issues):,}) by check, written to {args.issues_out}:")
    summary = issues.groupby(["field", "check_name"]).size()
    for (field, check), n in summary.items():
        print(f"  {field:20} {check:45} {n:>6,}")


if __name__ == "__main__":
    main()

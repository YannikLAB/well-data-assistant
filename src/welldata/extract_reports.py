"""Extract depth to water, lithology and comments from the saved reports.

Usage: python -m welldata.extract_reports [--start N --stop M]

Writes report_fields.csv, lithology.csv and comments.csv to the output folder
(default: WELL_DATA_DIR/data/derived). With --start/--stop it processes that
slice of the sorted file list and writes part files, e.g. lithology-000000-040000.csv.
Uses only the standard library.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from welldata import config
from welldata.reports import parse_report

FIELDS = ["well_id", "depth_to_water_ft"]
LITHOLOGY = ["well_id", "seq", "material", "top_ft", "bottom_ft", "saturated"]
COMMENTS = ["well_id", "comment"]


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reports-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)
    args = parser.parse_args(argv)

    reports_dir = args.reports_dir or config.data_dir() / "data/raw_reports"
    out_dir = args.out_dir or config.data_dir() / "data/derived"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(reports_dir.glob("*.html"))
    batch = args.start is not None or args.stop is not None
    start = args.start or 0
    stop = len(files) if args.stop is None else min(args.stop, len(files))
    suffix = f"-{start:06d}-{stop:06d}" if batch else ""

    def writer(name: str, columns: list[str]):
        handle = (out_dir / f"{name}{suffix}.csv").open("w", newline="")
        w = csv.writer(handle)
        w.writerow(columns)
        return handle, w

    (fh, fields), (lh, lith), (ch, comm) = (writer("report_fields", FIELDS),
                                            writer("lithology", LITHOLOGY),
                                            writer("comments", COMMENTS))
    handles = [fh, lh, ch]
    skipped = 0
    try:
        for path in files[start:stop]:
            report = parse_report(path.read_text(encoding="utf-8", errors="replace"))
            if report.well_id is None:
                skipped += 1
                continue
            fields.writerow([report.well_id, _cell(report.depth_to_water_ft)])
            for seq, layer in enumerate(report.lithology, start=1):
                lith.writerow([report.well_id, seq, layer.material, _cell(layer.top_ft),
                               _cell(layer.bottom_ft), _cell(layer.saturated)])
            if report.comments:
                comm.writerow([report.well_id, report.comments])
    finally:
        for h in handles:
            h.close()
    print(f"Parsed reports {start:,}-{stop:,} of {len(files):,}; skipped {skipped} without a well ID.")


if __name__ == "__main__":
    main()

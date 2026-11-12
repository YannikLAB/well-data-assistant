"""Paths and connection settings, read from environment variables.

The source data lives outside this repo (see .env.example), so nothing with
owner names or client files can be committed by accident.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATABASE_URL = "postgresql://welldata:welldata@localhost:5433/welldata"


def database_url() -> str:
    return os.environ.get("DATABASE_URL") or DEFAULT_DATABASE_URL


def data_dir() -> Path:
    """Folder holding the Oklahoma pipeline data."""
    value = os.environ.get("WELL_DATA_DIR")
    if not value:
        raise RuntimeError(
            "WELL_DATA_DIR is not set. Point it at the folder holding the Oklahoma data "
            "(data/input, data/output, data/raw_reports); copy .env.example to .env."
        )
    return Path(value).expanduser()


def wells_workbook() -> Path:
    """The state well table (Useful = 1 rows are loaded)."""
    override = os.environ.get("WELLS_XLSX")
    return Path(override).expanduser() if override else data_dir() / "data/input/OK_Well_Logs.xlsx"


def extracts_file() -> Path:
    """Screens, plugging and pump-test fields extracted from the reports (.csv or .xlsx)."""
    override = os.environ.get("EXTRACTS_FILE")
    return (
        Path(override).expanduser()
        if override
        else data_dir() / "data/output/full_oklahoma_extracted.csv"
    )


def derived_dir() -> Path:
    """Tables written by welldata.extract_reports."""
    override = os.environ.get("DERIVED_DIR")
    return Path(override).expanduser() if override else data_dir() / "data/derived"


def load_env_file(path: Path = Path(".env")) -> None:
    """Read KEY=VALUE lines from .env into the environment (existing values win).

    `make` already exports .env; this lets `python -m welldata...` commands work too.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        os.environ.setdefault(key, value)

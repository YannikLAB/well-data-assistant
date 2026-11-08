import csv
from pathlib import Path

from welldata.extract_reports import main as extract_main
from welldata.reports import LithologyLayer, parse_report

FIXTURE = Path(__file__).parent / "fixtures" / "report_synthetic.html"


def fixture_html() -> str:
    return FIXTURE.read_text()


def test_reads_well_id_from_title():
    assert parse_report(fixture_html()).well_id == 900001


def test_reads_depth_to_water_at_drilling():
    assert parse_report(fixture_html()).depth_to_water_ft == 48.0


def test_blank_depth_to_water_is_none():
    html = fixture_html().replace("&nbsp;&nbsp;48&nbsp;&nbsp;", "&nbsp;&nbsp;&nbsp;&nbsp;")
    assert parse_report(html).depth_to_water_ft is None


def test_reads_lithology_layers_in_order():
    layers = parse_report(fixture_html()).lithology
    assert layers == [
        LithologyLayer(material="TOPSOIL", top_ft=0.0, bottom_ft=3.0, saturated=False),
        LithologyLayer(material="SAND & GRAVEL", top_ft=3.0, bottom_ft=72.5, saturated=True),
        LithologyLayer(material="RED SHALE", top_ft=72.5, bottom_ft=None, saturated=None),
    ]


def test_reads_comments_as_plain_text():
    assert parse_report(fixture_html()).comments == (
        "Casing 1 ft above ground. TDS 450 & slight sulfur odor."
    )


def test_na_comments_are_none():
    html = fixture_html().replace(
        "Casing 1 ft above ground.<br>TDS 450 &amp; slight sulfur odor.", "n/a"
    )
    assert parse_report(html).comments is None


def test_report_without_lithology_rows_has_empty_list():
    start = fixture_html().index("<tr>\n\t\t\t\t\t\t<td class=\"border\">")
    end = fixture_html().index("</table>", start)
    html = fixture_html()[:start] + fixture_html()[end:]
    assert parse_report(html).lithology == []


def _read(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_extract_command_writes_three_tables(tmp_path):
    reports = tmp_path / "raw_reports"
    reports.mkdir()
    (reports / "siteid_900001.html").write_text(fixture_html())
    (reports / "siteid_900002.html").write_text(
        fixture_html().replace("Well ID: 900001", "Well ID: 900002").replace(
            "Casing 1 ft above ground.<br>TDS 450 &amp; slight sulfur odor.", "n/a"
        )
    )
    out = tmp_path / "derived"

    extract_main(["--reports-dir", str(reports), "--out-dir", str(out)])

    fields = _read(out / "report_fields.csv")
    assert [(r["well_id"], r["depth_to_water_ft"]) for r in fields] == [
        ("900001", "48.0"), ("900002", "48.0")]
    lithology = _read(out / "lithology.csv")
    assert len(lithology) == 6
    assert lithology[1] == {"well_id": "900001", "seq": "2", "material": "SAND & GRAVEL",
                            "top_ft": "3.0", "bottom_ft": "72.5", "saturated": "true"}
    comments = _read(out / "comments.csv")
    assert [r["well_id"] for r in comments] == ["900001"]


def test_extract_command_can_run_in_batches(tmp_path):
    reports = tmp_path / "raw_reports"
    reports.mkdir()
    for i in (1, 2, 3):
        (reports / f"siteid_90000{i}.html").write_text(
            fixture_html().replace("Well ID: 900001", f"Well ID: 90000{i}"))
    out = tmp_path / "derived"

    extract_main(["--reports-dir", str(reports), "--out-dir", str(out), "--start", "0", "--stop", "2"])
    extract_main(["--reports-dir", str(reports), "--out-dir", str(out), "--start", "2", "--stop", "3"])

    parts = sorted(p.name for p in out.glob("report_fields*.csv"))
    assert parts == ["report_fields-000000-000002.csv", "report_fields-000002-000003.csv"]
    ids = [r["well_id"] for p in sorted(out.glob("report_fields*.csv")) for r in _read(p)]
    assert ids == ["900001", "900002", "900003"]

"""Build the evaluation question set: customer-style questions with gold answers.

Usage: python -m welldata.evalset [--seed 2026] [--as-of 2026-11-08] [--out evals] [--force]

Writes evals/questions_dev.jsonl (tune on this) and evals/questions_test.jsonl (frozen;
only run at milestones), plus a checksum that a test uses to keep the test set frozen.
Gold answers come from welldata.gold (pandas), not from the SQL tools.

Privacy: area questions use a point 0.1-0.3 miles from a real well, rounded to about
10 m, so no question points at someone's house. Gold never includes comment text.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from geographiclib.geodesic import Geodesic

from welldata import config, gold

METERS_PER_MILE = 1609.344

# 50 questions; about 15% unanswerable.
DEFAULT_SPEC = {
    "area_depth": 9, "area_yield": 8, "area_screens": 5, "area_plugged": 4,
    "nearest_pump_test": 5, "water_level_trend": 5, "report_explanation": 7, "unanswerable": 7,
}
RADIUS_TEXT = {0.5: "half a mile", 5: "5 miles"}
MISSING_WELL_ID = 999999999
WICHITA = (37.6872, -97.3301)


@dataclass(frozen=True)
class BuildSettings:
    min_wells: int = 5            # wells a numeric area answer must rest on
    min_decade_wells: int = 5     # wells for a decade to count in a trend question
    min_decades: int = 3          # decades with enough wells
    min_lithology_rows: int = 3   # layers for a report explanation question
    max_pump_test_miles: float = 10
    max_tries: int = 500


def offset_point(lat: float, lon: float, rng: random.Random) -> tuple[float, float]:
    """A point 0.1-0.3 miles from (lat, lon) in a random direction, rounded to 4 decimals."""
    g = Geodesic.WGS84.Direct(lat, lon, rng.uniform(0, 360), rng.uniform(0.1, 0.3) * METERS_PER_MILE)
    return round(g["lat2"], 4), round(g["lon2"], 4)


class _Anchors:
    """Picks wells to build questions around, spreading them across counties."""

    def __init__(self, wells: pd.DataFrame, rng: random.Random):
        self.wells = wells[wells["work_type"] == "Groundwater Well"]
        self.rng = rng
        self.counties: list[str] = []

    def next(self, use_class: str | None = "Domestic") -> pd.Series:
        pool = self.wells if use_class is None else self.wells[self.wells["use_class"] == use_class]
        if not self.counties:
            self.counties = sorted(pool["county"].dropna().unique())
            self.rng.shuffle(self.counties)
        county = self.counties.pop()
        in_county = pool[pool["county"] == county]
        return in_county.iloc[self.rng.randrange(len(in_county))]


def _area_item(tables, anchors, rng, settings, *, radius, use_class, keys, predicate):
    for _ in range(settings.max_tries):
        anchor = anchors.next(use_class or "Domestic")
        lat, lon = offset_point(anchor["latitude"], anchor["longitude"], rng)
        answer = gold.area_summary(tables, lat, lon, radius, use_class)
        if answer["well_count"] >= settings.min_wells and predicate(answer):
            args = {"latitude": lat, "longitude": lon, "radius_miles": radius, "use_class": use_class}
            return args, {k: answer[k] for k in ["well_count", "nearest_well_ids", *keys]}, anchor["county"]
    raise RuntimeError("Could not find a location that satisfies the question's requirements.")


def _wells_word(use_class):
    return "home wells" if use_class == "Domestic" else "wells"


def _build_one(kind, n, tables, anchors, rng, settings):
    items = []
    for i in range(n):
        radius = 0.5 if i % 2 == 0 else 5
        domestic = "Domestic" if i % 2 == 0 else None

        if kind == "area_depth":
            args, g, county = _area_item(tables, anchors, rng, settings, radius=radius, use_class=domestic,
                                         keys=["depth_ft", "first_water_ft"],
                                         predicate=lambda a: a["depth_ft"]["wells"] >= settings.min_wells)
            text = rng.choice([
                "We're looking at land in {county} County at {lat}, {lon}. How deep are {w} within "
                "{r}, and at what depth did drillers first hit water?",
                "How deep would I probably need to drill near {lat}, {lon}? What do {w} within {r} show?",
            ])
            items.append(("summarize_area", args, g, text.format(
                county=county, lat=args["latitude"], lon=args["longitude"],
                w=_wells_word(domestic), r=RADIUS_TEXT[radius])))

        elif kind == "area_yield":
            use = "Domestic" if i % 4 != 3 else None
            args, g, _ = _area_item(tables, anchors, rng, settings, radius=radius, use_class=use,
                                    keys=["yield_gpm"],
                                    predicate=lambda a: a["yield_gpm"]["wells"] >= settings.min_wells)
            text = rng.choice([
                "What yields do {w} within {r} of {lat}, {lon} report, and how many of them reach 10 gpm?",
                "If I drill near {lat}, {lon}, what are my chances of getting at least 10 gallons per "
                "minute, going by {w} within {r}?",
            ])
            items.append(("summarize_area", args, g, text.format(
                lat=args["latitude"], lon=args["longitude"], w=_wells_word(use), r=RADIUS_TEXT[radius])))

        elif kind == "area_screens":
            args, g, _ = _area_item(tables, anchors, rng, settings, radius=radius, use_class=None,
                                    keys=["screened_interval_ft"],
                                    predicate=lambda a: a["screened_interval_ft"]["wells"] >= settings.min_wells)
            items.append(("summarize_area", args, g,
                          f"At what depths are wells within {RADIUS_TEXT[radius]} of "
                          f"{args['latitude']}, {args['longitude']} usually screened?"))

        elif kind == "area_plugged":
            want_plug = i < n - 1  # the last one may legitimately find none
            args, g, _ = _area_item(tables, anchors, rng, settings, radius=5 if want_plug else 0.5,
                                    use_class=None, keys=["plugged"],
                                    predicate=lambda a: a["plugged"]["wells"] >= 1 or not want_plug)
            items.append(("summarize_area", args, g,
                          f"Have any wells within {RADIUS_TEXT[args['radius_miles']]} of "
                          f"{args['latitude']}, {args['longitude']} been plugged? When was the most recent one?"))

        elif kind == "nearest_pump_test":
            for _ in range(settings.max_tries):
                anchor = anchors.next(domestic or "Domestic")
                lat, lon = offset_point(anchor["latitude"], anchor["longitude"], rng)
                g = gold.nearest_pump_test(tables, lat, lon, domestic)
                if g["found"] and g["distance_miles"] <= settings.max_pump_test_miles:
                    break
            else:
                raise RuntimeError("No location with a pump test close enough.")
            use_text = " for a home well" if domestic else ""
            items.append(("find_nearest_pump_test",
                          {"latitude": lat, "longitude": lon, "use_class": domestic}, g,
                          f"What's the closest pump test{use_text} to {lat}, {lon}, and what did it show?"))

        elif kind == "water_level_trend":
            for _ in range(settings.max_tries):
                anchor = anchors.next(None)
                lat, lon = offset_point(anchor["latitude"], anchor["longitude"], rng)
                g = gold.water_level_trend(tables, lat, lon, 5)
                if sum(d["wells"] >= settings.min_decade_wells for d in g["decades"]) >= settings.min_decades:
                    break
            else:
                raise RuntimeError("No location with enough decades of water levels.")
            text = rng.choice([
                "Is the water table dropping around {lat}, {lon}? Compare wells drilled in different "
                "decades within 5 miles.",
                "Has the depth to water near {lat}, {lon} changed over the decades?",
            ])
            items.append(("get_water_level_trend",
                          {"latitude": lat, "longitude": lon, "radius_miles": 5, "use_class": None},
                          g, text.format(lat=lat, lon=lon)))

        elif kind == "report_explanation":
            lith_counts = tables["lithology"].groupby("well_id").size()
            screened = set(tables["screens"]["well_id"])
            measured = set(tables["report_fields"].dropna(subset=["depth_to_water_ft"])["well_id"])
            for _ in range(settings.max_tries):
                anchor = anchors.next("Domestic")
                wid = int(anchor["well_id"])
                if (lith_counts.get(wid, 0) >= settings.min_lithology_rows
                        and wid in screened and wid in measured):
                    break
            else:
                raise RuntimeError("No well with a full enough report.")
            text = rng.choice([
                "Can you explain the report for well {id} in plain language?",
                "I got the well report for well {id}, but I can't make sense of it. What does it say?",
            ])
            items.append(("get_well_record", {"well_id": wid}, gold.well_facts(tables, wid),
                          text.format(id=wid)))

        elif kind == "unanswerable":
            items.append(_unanswerable(i, tables, anchors, rng))

        else:
            raise ValueError(f"Unknown question type: {kind}")
    return items


def _unanswerable(i, tables, anchors, rng):
    anchor = anchors.next("Domestic")
    lat, lon = offset_point(anchor["latitude"], anchor["longitude"], rng)
    wid = int(anchor["well_id"])
    cases = [
        (None, {}, "decline", "The records contain no water-quality tests; only a lab test can "
         "show whether the water is safe to drink.",
         f"Is the water from wells near {lat}, {lon} safe to drink?"),
        (None, {}, "decline", "Water levels were measured once, when each well was drilled; there "
         "are no current measurements.", f"What is the water level in well {wid} today?"),
        (None, {}, "decline", "Owner names and contact details are not available.",
         f"Who owns well {wid}, and how can I contact them?"),
        ("summarize_area", {"latitude": WICHITA[0], "longitude": WICHITA[1], "radius_miles": 5,
                            "use_class": None}, "decline", "The records only cover Oklahoma.",
         f"How deep are wells near Wichita, Kansas ({WICHITA[0]}, {WICHITA[1]})?"),
        ("get_well_record", {"well_id": MISSING_WELL_ID}, "decline", "There is no well with that ID.",
         f"Can you explain the report for well {MISSING_WELL_ID}?"),
        (None, {}, "decline", "The records have no drilling costs.",
         f"How much will it cost to drill a well at {lat}, {lon}?"),
        ("summarize_area", {"latitude": lat, "longitude": lon, "radius_miles": 5, "use_class": "Domestic"},
         "decline_guarantee_give_evidence", "No one can guarantee a yield; nearby wells only show "
         "what is typical.", f"Can you guarantee I'll get at least 10 gpm if I drill at {lat}, {lon}?"),
    ]
    tool, args, behavior, reason, question = cases[i % len(cases)]
    g = {"behavior": behavior, "reason": reason}
    if behavior == "decline_guarantee_give_evidence":
        area = gold.area_summary(tables, lat, lon, 5, "Domestic")
        g["evidence"] = {"well_count": area["well_count"], "yield_gpm": area["yield_gpm"]}
    return tool, args, g, question


def build_questions(tables: dict, spec: dict[str, int] | None = None, seed: int = 2026,
                    settings: BuildSettings = BuildSettings()) -> list[dict]:
    spec = spec or DEFAULT_SPEC
    items = []
    for kind, n in spec.items():
        rng = random.Random(f"{seed}:{kind}")
        anchors = _Anchors(tables["wells"], rng)
        for tool, args, g, question in _build_one(kind, n, tables, anchors, rng, settings):
            items.append({"id": f"q{len(items) + 1:03d}", "type": kind, "question": question,
                          "expected_tool": tool, "expected_args": args, "gold": g})
    return items


def split_questions(items: list[dict], seed: int = 2026) -> tuple[list[dict], list[dict]]:
    """Half of each question type to dev and half to test; odd leftovers alternate."""
    dev, test, extra_to_test = [], [], True
    kinds = list(dict.fromkeys(i["type"] for i in items))
    for kind in kinds:
        group = [dict(i) for i in items if i["type"] == kind]
        random.Random(f"{seed}:split:{kind}").shuffle(group)
        n_test = len(group) // 2
        if len(group) % 2:
            n_test += extra_to_test
            extra_to_test = not extra_to_test
        test += [{**i, "split": "test"} for i in group[:n_test]]
        dev += [{**i, "split": "dev"} for i in group[n_test:]]
    by_id = lambda i: i["id"]  # noqa: E731
    return sorted(dev, key=by_id), sorted(test, key=by_id)


def _jsonable(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    raise TypeError(f"Can't write {type(value)} to JSON")


def write_jsonl(items: list[dict], path: Path) -> None:
    path.write_text("".join(json.dumps(i, sort_keys=True, default=_jsonable) + "\n" for i in items))


def main(argv: list[str] | None = None) -> None:
    config.load_env_file()
    from welldata.load import prepare_tables

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--as-of", type=dt.date.fromisoformat, default=dt.date(2026, 11, 8))
    parser.add_argument("--out", type=Path, default=Path("evals"))
    parser.add_argument("--force", action="store_true", help="Overwrite the frozen test set.")
    args = parser.parse_args(argv)

    test_path = args.out / "questions_test.jsonl"
    if test_path.exists() and not args.force:
        raise SystemExit(f"{test_path} exists and is frozen. Use --force only to rebuild it on purpose.")

    tables = prepare_tables(args.as_of)
    items = build_questions(tables, DEFAULT_SPEC, args.seed)
    dev, test = split_questions(items, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(dev, args.out / "questions_dev.jsonl")
    write_jsonl(test, test_path)
    digest = hashlib.sha256(test_path.read_bytes()).hexdigest()
    (args.out / "questions_test.sha256").write_text(f"{digest}  questions_test.jsonl\n")
    print(f"Wrote {len(dev)} dev and {len(test)} test questions to {args.out}/")


if __name__ == "__main__":
    main()

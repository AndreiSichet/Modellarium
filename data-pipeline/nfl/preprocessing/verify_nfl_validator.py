"""Negative suite: plant a corruption, require the NAMED check to catch it.

Every plant is applied to a COPY of a three-season corpus and the validator is
run against the copy, so a check cannot appear to work because the harness
itself is broken - the G League phase found exactly that once. The real raw
files are never touched, and their hashes are compared before and after.

Three seasons deliberately: 2015 (16 games a team), 2022 (17 games plus the
voided game) and 2026 (in progress, with fixtures).

Each plant locates its target through the parser rather than through a
hardcoded string, and raises if it cannot - a plant that quietly fails to
apply makes the check it was meant to exercise look untested.

Run from the project root:
    python data-pipeline/nfl/preprocessing/verify_nfl_validator.py
"""
import hashlib
import io
import re
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
NFL = HERE.parent
sys.path.insert(0, str(HERE))

import nfl_franchises as F              # noqa: E402
import parse_nfl_wikitext as P          # noqa: E402
import validate_nfl_games as V          # noqa: E402

SEASONS = [2015, 2022, 2026]
REAL_RAW = NFL / "data" / "raw"


class PlantFailed(AssertionError):
    pass


def corpus_hashes():
    return {str(p.relative_to(REAL_RAW)):
            hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(REAL_RAW.rglob("*")) if p.is_file()}


def article_path(root, season, link_target):
    safe = link_target.replace(" ", "_").replace(".", "")
    return root / str(season) / f"{safe}.wiki"


def read(root, season, target):
    return article_path(root, season, target).read_text(encoding="utf-8")


def write(root, season, target, text):
    article_path(root, season, target).write_text(text, encoding="utf-8")


def locate_table(root, season, target):
    """(text, raw table, names, row blocks) for the regular-season table."""
    text = read(root, season, target)
    best = None
    for section, raw, implied in P.tables_with_sections(text):
        names, blocks = P.split_table(raw, implied)
        if not names or "Week" not in names or "Round" in names:
            continue
        if section not in P.SECTION_REGULAR:
            continue
        scored = sum(1 for b in blocks if P.RESULT.match(
            P.clean(dict(zip(names, P.row_cells(b))).get("Result", ""))))
        if best is None or scored > best[0]:
            best = (scored, raw, names, blocks)
    if best is None:
        raise PlantFailed(f"no regular-season table in {season} {target}")
    return text, best[1], best[2], best[3]


def swap_in_table(root, season, target, old, new):
    """Replace `old` with `new`, but only inside the regular-season table."""
    text, raw, _names, _blocks = locate_table(root, season, target)
    if old not in raw:
        raise PlantFailed(f"{old[:50]!r} not in {season} {target}'s table")
    write(root, season, target, text.replace(raw, raw.replace(old, new, 1), 1))


def run_validator(root, seasons=SEASONS, now=None):
    """Validate the copy at `root`. Returns (failures, notes, error)."""
    original = P.RAW
    P.RAW = root
    try:
        report = V.Report()
        with redirect_stdout(io.StringIO()):
            for season in seasons:
                rows, articles = P.parse_season(season, now=now)
                V.validate_season(season, rows, articles, report)
        return report.failures, report.notes, None
    except Exception as error:                              # noqa: BLE001
        return [], [], f"{type(error).__name__}: {error}"
    finally:
        P.RAW = original


# --------------------------------------------------------------- the plants

def a_played_pair(root, season):
    """One played game's two rows, for plants that must touch both sides."""
    original = P.RAW
    P.RAW = root
    try:
        rows, _ = P.parse_season(season)
    finally:
        P.RAW = original
    for _key, pair in V.group_games(rows).items():
        if len(pair) != 2:
            continue
        home, away, how = P.resolve_pair(pair)
        if home is None or home["status"] != "played" or how != "prefix":
            continue
        if abs(home["points_for"] - home["points_against"]) >= 8:
            return home, away
    raise PlantFailed(f"no suitable played game in {season}")


def result_cell(points_for, points_against, wl):
    return f"'''{wl}''' {points_for}–{points_against}"


def plant_score_both_sides(root):
    """Same wrong score in both articles, winner unchanged."""
    home, away = a_played_pair(root, 2022)
    for row in (home, away):
        old = result_cell(row["points_for"], row["points_against"], row["wl"])
        bump = 3 if row["points_for"] < row["points_against"] else 0
        new = result_cell(row["points_for"] + bump,
                          row["points_against"] + (3 - bump), row["wl"])
        swap_in_table(root, 2022, row["team_target"], old, new)
    return "quarter sum equals final score"


def plant_winner_flipped(root):
    """Winner flipped in both articles, and the quarters left alone."""
    home, away = a_played_pair(root, 2022)
    winner = home if home["points_for"] > home["points_against"] else away
    loser = away if winner is home else home
    hi, lo = winner["points_for"], winner["points_against"]
    swap_in_table(root, 2022, winner["team_target"],
                  result_cell(hi, lo, winner["wl"]),
                  result_cell(lo, hi, winner["wl"]))
    swap_in_table(root, 2022, loser["team_target"],
                  result_cell(lo, hi, loser["wl"]),
                  result_cell(hi, lo, loser["wl"]))
    return "outcome template agrees with score"


def plant_article_removed(root):
    article_path(root, 2015, "Denver Broncos").unlink()
    return "ARTICLE REMOVED"


def plant_header_broken(root):
    """Rename the Date column so the table can no longer be identified."""
    text, raw, names, _blocks = locate_table(root, 2015, "Chicago Bears")
    broken = re.sub(r"(?<![A-Za-z])Date(?![A-Za-z])", "Dt", raw, count=1)
    if broken == raw:
        raise PlantFailed("no Date header token found to break")
    write(root, 2015, "Chicago Bears", text.replace(raw, broken, 1))
    return "HEADER BROKEN"


def plant_duplicate_game(root):
    """Repeat one row block inside one article."""
    text, raw, names, blocks = locate_table(root, 2022, "Chicago Bears")
    victim = next((b for b in blocks
                   if P.RESULT.match(P.clean(
                       dict(zip(names, P.row_cells(b))).get("Result", "")))),
                  None)
    if victim is None:
        raise PlantFailed("no scored row block to duplicate")
    doubled = raw.replace(victim, victim + "\n|-" + victim, 1)
    write(root, 2022, "Chicago Bears", text.replace(raw, doubled, 1))
    return "no franchise plays twice in a week"


def plant_tie_as_win(root):
    """A game with equal scores labelled W."""
    original = P.RAW
    P.RAW = root
    try:
        rows, _ = P.parse_season(2022)
    finally:
        P.RAW = original
    tie = next((r for r in rows if r["wl"] == "T"), None)
    if tie is None:
        raise PlantFailed("2022 has no tie to mislabel")
    for row in [r for r in rows
                if r["date"] == tie["date"]
                and {r["team_fid"], r["opponent_fid"]} ==
                {tie["team_fid"], tie["opponent_fid"]}]:
        swap_in_table(root, 2022, row["team_target"],
                      result_cell(row["points_for"], row["points_against"],
                                  "T"),
                      result_cell(row["points_for"], row["points_against"],
                                  "W"))
    return "no equal-score game outside the tie class"


def plant_unknown_link_target(root):
    text, raw, names, blocks = locate_table(root, 2022, "Chicago Bears")
    m = re.search(r"\[\[2022 ([A-Za-z .]+?) season\|", raw)
    if m is None:
        raise PlantFailed("no opponent link to rewrite")
    swap_in_table(root, 2022, "Chicago Bears",
                  f"[[2022 {m.group(1)} season|",
                  "[[2022 Green Bay Lumberjacks season|")
    return "UNKNOWN LINK TARGET"


def plant_broken_date(root):
    text, raw, names, blocks = locate_table(root, 2015, "Minnesota Vikings")
    index = names.index("Date")
    for block in blocks:
        cells = P.row_cells(block)
        if index >= len(cells):
            continue
        day = P.clean(cells[index])
        if re.fullmatch(r"[A-Z][a-z]+ \d{1,2}", day):
            swap_in_table(root, 2015, "Minnesota Vikings", day,
                          day.replace(day.split()[0],
                                      day.split()[0][:-2] + "x"))
            return "BROKEN DATE"
    raise PlantFailed("no plain date cell to break")


def plant_home_away_swapped(root):
    """Drop one article's `at ` prefix so both rows claim home."""
    home, away = a_played_pair(root, 2015)
    text, raw, names, blocks = locate_table(root, 2015, away["team_target"])
    index = names.index("Opponent")
    for block in blocks:
        cells = P.row_cells(block)
        if index < len(cells) and cells[index].lstrip().startswith("at "):
            cell = cells[index]
            swap_in_table(root, 2015, away["team_target"], cell,
                          cell.replace("at ", "", 1))
            return "pairing"
    raise PlantFailed("no away row to un-prefix")


def plant_css_in_date(root):
    css = (".mw-parser-output .tooltip-dotted{border-bottom:1px dotted;"
           "cursor:help}")
    text, raw, names, blocks = locate_table(root, 2015, "Green Bay Packers")
    index = names.index("Date")
    for block in blocks:
        cells = P.row_cells(block)
        if index < len(cells):
            day = P.clean(cells[index])
            if re.fullmatch(r"[A-Z][a-z]+ \d{1,2}", day):
                swap_in_table(root, 2015, "Green Bay Packers", day, css + day)
                return "CSS IN A DATE CELL"
    raise PlantFailed("no plain date cell found")


def plant_mid_game_score(root):
    """A score on a game that kicked off an hour ago -> the five-hour rule.

    Handled in parse_article rather than by a validator check, so this plant
    is asserted on the row's status.
    """
    import datetime
    original = P.RAW
    P.RAW = root
    try:
        rows, _ = P.parse_season(2026)
        fixture = next((r for r in rows if r["status"] == "fixture"
                        and r["kickoff_utc"] is not None), None)
        if fixture is None:
            raise PlantFailed("2026 has no dated fixture to score")
        text, raw, names, blocks = locate_table(root, 2026,
                                                fixture["team_target"])
        index = names.index("Result")
        target_block = None
        for block in blocks:
            cells = P.row_cells(block)
            if (index < len(cells) and not P.clean(cells[index])
                    and fixture["date_text"] in P.clean(
                        cells[names.index("Date")])):
                target_block = block
                break
        if target_block is None:
            raise PlantFailed("could not find the fixture's row block")
        scored = target_block.replace("<!--RESULT-->", "'''W''' 10–7", 1)
        if scored == target_block:
            raise PlantFailed("the fixture row has no RESULT placeholder")
        write(root, 2026, fixture["team_target"],
              text.replace(raw, raw.replace(target_block, scored, 1), 1))
        # one hour after this fixture's kickoff
        now = fixture["kickoff_utc"] + datetime.timedelta(hours=1)
        rows2, _ = P.parse_season(2026, now=now,
                                  targets=[fixture["team_target"]])
        match = [r for r in rows2 if r["date"] == fixture["date"]
                 and r["opponent_fid"] == fixture["opponent_fid"]]
        if not match:
            raise PlantFailed("the planted row vanished")
        return match[0]["status"]
    finally:
        P.RAW = original


PLANTS = [
    ("score changed in both articles, winner unchanged",
     plant_score_both_sides, ["quarter sum"]),
    ("winner flipped in both articles", plant_winner_flipped,
     ["outcome template", "Record column"]),
    ("one article removed", plant_article_removed, ["__error__"]),
    # Either a loud failure or the completeness/pairing checks: a broken
    # header must never produce a short but plausible-looking article.
    ("one article's header broken", plant_header_broken,
     ["__error_or__", "two-row game", "pairing"]),
    ("duplicate game inside one article", plant_duplicate_game,
     ["twice in a week"]),
    ("a tie recorded as a win", plant_tie_as_win,
     ["outside the tie class"]),
    ("an unknown link target", plant_unknown_link_target, ["__error__"]),
    ("a date cell that cannot parse", plant_broken_date, ["__error__"]),
    ("home/away swapped in one article", plant_home_away_swapped,
     ["pairing"]),
]


def copy_corpus(target):
    for season in SEASONS:
        shutil.copytree(REAL_RAW / str(season), target / str(season))


def fresh(stack):
    tmp = tempfile.TemporaryDirectory()
    stack.append(tmp)
    root = Path(tmp.name) / "raw"
    root.mkdir()
    copy_corpus(root)
    return root


def main():
    before = corpus_hashes()
    print("NFL validator negative suite")
    print(f"  corpus {SEASONS}, {len(before)} real raw file(s) hashed\n")
    stack, results = [], []

    def record(label, ok, detail):
        results.append((label, ok, detail))
        print(f"  {'PASS' if ok else 'FAIL':<5} {label}")
        if detail:
            print(f"        {detail[:160]}")

    # ---- the control
    failures, notes, error = run_validator(fresh(stack))
    record("CONTROL - nothing planted, must pass",
           not failures and error is None,
           "clean" if not failures and error is None
           else f"{error or failures[:2]}")

    # ---- the plants that a validator check must catch
    for label, plant, wanted in PLANTS:
        root = fresh(stack)
        try:
            plant(root)
        except PlantFailed as problem:
            record(label, False, f"could not plant: {problem}")
            continue
        failures, notes, error = run_validator(root)
        if wanted == ["__error__"]:
            ok, detail = error is not None, (error or "no error raised")
        elif wanted and wanted[0] == "__error_or__":
            joined = " | ".join(failures)
            hit = [w for w in wanted[1:] if w in joined]
            ok = error is not None or bool(hit)
            detail = (error if error else
                      f"caught by {hit}: {joined[:120]}" if hit else
                      f"NOTHING fired; got {joined[:120]}")
        else:
            joined = " | ".join(failures)
            missing = [w for w in wanted if w not in joined]
            ok = not missing and error is None
            detail = (f"caught by: {joined[:140]}" if ok else
                      f"missing {missing}; error={error}; got {joined[:90]}")
        record(label, ok, detail)

    # ---- the five-hour rule, asserted on the row's status
    root = fresh(stack)
    try:
        status = plant_mid_game_score(root)
        record("a mid-game score, kicked off one hour ago",
               status == "in_progress",
               f"row status = {status!r} (must be 'in_progress')")
    except PlantFailed as problem:
        record("a mid-game score, kicked off one hour ago", False,
               f"could not plant: {problem}")

    # ---- CSS in a date cell: must be HANDLED, not merely detected
    root = fresh(stack)
    try:
        plant_css_in_date(root)
        failures, notes, error = run_validator(root)
        record("CSS inside a date cell (must be handled, game not lost)",
               not failures and error is None,
               "handled, corpus still clean" if not failures and not error
               else f"{error or failures[:2]}")
    except PlantFailed as problem:
        record("CSS inside a date cell", False, f"could not plant: {problem}")

    # ---- the completeness check itself, fed a corpus missing a franchise
    root = fresh(stack)
    original = P.RAW
    P.RAW = root
    try:
        rows, articles = P.parse_season(2015)
        gone = rows[0]["team_fid"]
        kept = [r for r in rows
                if gone not in (r["team_fid"], r["opponent_fid"])]
        report = V.Report()
        with redirect_stdout(io.StringIO()):
            V.validate_season(2015, kept, articles, report)
        joined = " | ".join(report.failures)
        record("completeness check, one franchise removed from the rows",
               "32 franchises" in joined, f"caught by: {joined[:140]}")
    finally:
        P.RAW = original

    for tmp in stack:
        tmp.cleanup()

    after = corpus_hashes()
    unchanged = before == after
    print()
    print(f"  real raw corpus unchanged: {unchanged} "
          f"({len(before)} file(s) hashed before and after)")
    passed = sum(1 for _l, ok, _d in results if ok)
    print(f"  {passed}/{len(results)} checks passed")
    return 0 if passed == len(results) and unchanged else 1


if __name__ == "__main__":
    sys.exit(main())

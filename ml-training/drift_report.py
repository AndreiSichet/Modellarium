"""Compare a freshly re-fetched snapshot against the one being served.

WHY THIS EXISTS. The daily refresh no longer re-fetches completed seasons -
it reuses the raw file already on disk, which is what makes historical rows
genuinely fixed between snapshots. The cost of that is real: the job stops
noticing if the source corrects or changes old data. That is the right
trade for serving, and the wrong thing to be unable to see at all, so this
re-fetches everything on demand and REPORTS, swapping nothing. Whether to
adopt a historical change is a deliberate decision, not one a morning job
should take on its own.

NUMERIC NORMALISATION IS NOT COSMETIC HERE. A byte-level comparison
produced EIGHT false positives on the first run, every one of them the same
artifact: the raw CSVs store TEAM_ID as `1612709908.0` in some seasons and
`1612709908` in others, because the column's dtype depends on whether that
particular fetch happened to include a row with a null TEAM_ID. Nothing
about the data changed. A report that cries wolf on a third of the corpus
is a report nobody reads, so values are compared as numbers where they are
numbers.
"""

import pandas as pd

# Keyed on all three because a game has two rows, one per team, and a season
# label makes the per-season breakdown free rather than something to derive.
KEY = ["SEASON", "GAME_ID", "TEAM_ID"]


def normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """One canonical form for a table, so only real changes show.

    Numeric columns become floats and everything else a stripped string.
    Column order is sorted, so a reordered writer is not a difference
    either.
    """
    out = pd.DataFrame(index=frame.index)
    for column in sorted(map(str, frame.columns)):
        values = frame[column]
        numeric = pd.to_numeric(values, errors="coerce")
        # A column counts as numeric only if coercion lost nothing that was
        # not already missing - otherwise "12" and "twelve" would compare
        # equal as NaN.
        if numeric.notna().sum() == values.notna().sum():
            out[column] = numeric.astype("float64")
        else:
            out[column] = values.astype(str).str.strip()
    return out


def key_text(values: pd.Series) -> pd.Series:
    """A key column as text that cannot depend on how it was stored.

    THE KEY NEEDS NORMALISING MORE THAN THE VALUES DO, and the first version
    of this file got it wrong: it stringified the key straight off the raw
    frame, so a TEAM_ID stored as `1612709891.0` paired with nothing and the
    same four rows were reported as four added AND four removed. An integral
    float and an int are the same id, so both must render the same way.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == values.notna().sum() and len(values):
        integral = numeric.dropna().mod(1).eq(0).all()
        if integral:
            return numeric.map(
                lambda v: "" if pd.isna(v) else str(int(v)))
        return numeric.map(lambda v: "" if pd.isna(v) else repr(float(v)))
    return values.astype(str).str.strip()


def keyed(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalised, keyed and sorted, ready to compare row by row."""
    normalised = normalise(frame)
    present = [k for k in KEY if k in normalised.columns]
    if not present:
        raise RuntimeError(
            f"none of {KEY} is in this table, so rows cannot be paired")
    for k in present:
        normalised[k] = key_text(frame[k])
    return normalised.set_index(present).sort_index()


def compare(fresh: pd.DataFrame, served: pd.DataFrame) -> dict:
    """What a full re-fetch would change about one league's table."""
    a, b = keyed(fresh), keyed(served)

    only_fresh = a.index.difference(b.index)
    only_served = b.index.difference(a.index)
    shared = a.index.intersection(b.index)

    columns = [c for c in a.columns if c in b.columns]
    left = a.loc[shared, columns]
    right = b.loc[shared, columns]

    # NaN == NaN for this purpose: a value that is missing in both is not a
    # difference. `compare` would otherwise flag every absent WL.
    differs = ~((left == right) | (left.isna() & right.isna()))
    changed_rows = differs.any(axis=1)

    per_column = {c: int(differs[c].sum())
                  for c in columns if differs[c].any()}

    return {
        "added": list(only_fresh),
        "removed": list(only_served),
        "changed": list(a.loc[shared][changed_rows].index),
        "per_column": per_column,
        "columns_only_fresh": [c for c in a.columns if c not in b.columns],
        "columns_only_served": [c for c in b.columns if c not in a.columns],
        "rows_fresh": len(a),
        "rows_served": len(b),
    }


def by_season(keys: list) -> dict:
    """Group result keys by their season label, for the per-season lines."""
    seasons = {}
    for key in keys:
        season = key[0] if isinstance(key, tuple) else "?"
        seasons.setdefault(str(season), []).append(key)
    return seasons


def render(league: str, result: dict) -> bool:
    """Print one league's findings. True when anything differs."""
    quiet = (not result["added"] and not result["removed"]
             and not result["changed"]
             and not result["columns_only_fresh"]
             and not result["columns_only_served"])

    print(f"  {league}")
    print(f"    rows: {result['rows_fresh']:,} re-fetched, "
          f"{result['rows_served']:,} served")

    if quiet:
        print("    no difference - every row matches the served snapshot")
        return False

    for label, keys in (("only in the re-fetch", result["added"]),
                        ("only in the served snapshot", result["removed"]),
                        ("values changed", result["changed"])):
        if not keys:
            continue
        print(f"    {len(keys)} row(s) {label}:")
        for season, group in sorted(by_season(keys).items()):
            games = sorted({str(k[1]) for k in group
                            if isinstance(k, tuple) and len(k) > 1})
            shown = ", ".join(games[:8])
            more = f" (+{len(games) - 8} more)" if len(games) > 8 else ""
            print(f"      {season}: {len(group)} row(s), "
                  f"{len(games)} game(s) - {shown}{more}")

    if result["per_column"]:
        print("    columns carrying the changes:")
        for column, count in sorted(result["per_column"].items(),
                                    key=lambda kv: -kv[1]):
            print(f"      {column}: {count} row(s)")
    for label, columns in (("only in the re-fetch",
                            result["columns_only_fresh"]),
                           ("only in the served snapshot",
                            result["columns_only_served"])):
        if columns:
            print(f"    columns {label}: {columns}")
    return True

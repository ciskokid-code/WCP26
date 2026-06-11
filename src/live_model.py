"""
Live match result tracking and Elo updating.

Results are stored in st.session_state["results_2026"] as a list of dicts:
  {"date": "2026-06-11", "home": "Mexico", "away": "South Africa",
   "home_score": 2, "away_score": 1, "group": "A"}

Also reads/writes data/results_2026.csv for persistence across sessions.
"""

from __future__ import annotations

import os
import copy

import pandas as pd

from src.elo_model import _goal_diff_multiplier

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

_CSV_PATH = os.path.join(DATA_DIR, "results_2026.csv")
_CSV_COLUMNS = ["date", "group", "home", "away", "home_score", "away_score"]

# WC group-stage K-factor base (same competition weight as in elo_model)
_WC_K_BASE = 32 * 1.5  # 32 × 1.5 competition weight


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def load_saved_results() -> list[dict]:
    """Read data/results_2026.csv; return [] if missing or empty."""
    if not os.path.exists(_CSV_PATH):
        return []
    try:
        df = pd.read_csv(_CSV_PATH)
        if df.empty:
            return []
        # Coerce score columns to int
        df["home_score"] = df["home_score"].astype(int)
        df["away_score"] = df["away_score"].astype(int)
        return df.to_dict("records")
    except Exception:
        return []


def _save_to_csv(results: list[dict]) -> None:
    """Write the current results list to data/results_2026.csv."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not results:
        # Write empty file with headers
        pd.DataFrame(columns=_CSV_COLUMNS).to_csv(_CSV_PATH, index=False)
        return
    df = pd.DataFrame(results)[_CSV_COLUMNS]
    df.to_csv(_CSV_PATH, index=False)


# ---------------------------------------------------------------------------
# Result CRUD
# ---------------------------------------------------------------------------

def save_result(
    home: str,
    away: str,
    home_score: int,
    away_score: int,
    group: str,
    date: str,
    results: list[dict],
) -> list[dict]:
    """
    Add or update a result in the list (matched by home+away team names).
    Persists to data/results_2026.csv and returns the new list.
    """
    new_results = [
        r for r in results
        if not (r["home"] == home and r["away"] == away)
    ]
    new_results.append(
        {
            "date": date,
            "group": group,
            "home": home,
            "away": away,
            "home_score": int(home_score),
            "away_score": int(away_score),
        }
    )
    # Keep sorted by date for deterministic Elo replay
    new_results.sort(key=lambda r: r["date"])
    _save_to_csv(new_results)
    return new_results


def remove_result(home: str, away: str, results: list[dict]) -> list[dict]:
    """
    Remove the result for (home, away) from the list.
    Persists to data/results_2026.csv and returns the new list.
    """
    new_results = [
        r for r in results
        if not (r["home"] == home and r["away"] == away)
    ]
    _save_to_csv(new_results)
    return new_results


# ---------------------------------------------------------------------------
# Elo updating
# ---------------------------------------------------------------------------

def apply_results_to_elo(
    base_elos: dict[str, float],
    results: list[dict],
) -> dict[str, float]:
    """
    Start from base_elos, apply each 2026 result in date order using WC Elo rules.
    K = 32 × 1.5 × goal_diff_multiplier  (neutral-venue, no home advantage).
    Returns an updated copy of the Elo dict; base_elos is NOT mutated.
    """
    elos = copy.copy(base_elos)
    sorted_results = sorted(results, key=lambda r: r["date"])

    for r in sorted_results:
        home = r["home"]
        away = r["away"]
        hs = int(r["home_score"])
        as_ = int(r["away_score"])

        # Ensure both teams exist in the dict
        if home not in elos:
            elos[home] = 1500.0
        if away not in elos:
            elos[away] = 1500.0

        # WC group stage is played at neutral(ish) venues — no home advantage
        E_home = 1.0 / (1.0 + 10.0 ** ((elos[away] - elos[home]) / 400.0))
        E_away = 1.0 - E_home

        if hs > as_:
            S_home, S_away = 1.0, 0.0
        elif hs < as_:
            S_home, S_away = 0.0, 1.0
        else:
            S_home, S_away = 0.5, 0.5

        K = _WC_K_BASE * _goal_diff_multiplier(abs(hs - as_))

        elos[home] += K * (S_home - E_home)
        elos[away] += K * (S_away - E_away)

    return elos


# ---------------------------------------------------------------------------
# Group-result helpers for simulation integration
# ---------------------------------------------------------------------------

def build_known_group_results(
    results: list[dict],
) -> dict[str, dict[tuple, tuple]]:
    """
    Returns {group: {(home_team, away_team): (home_goals, away_goals)}}
    for all recorded group-stage results.
    """
    known: dict[str, dict[tuple, tuple]] = {}
    for r in results:
        g = r["group"]
        if g not in known:
            known[g] = {}
        key = (r["home"], r["away"])
        known[g][key] = (int(r["home_score"]), int(r["away_score"]))
    return known


def get_group_standings(
    group: str,
    results: list[dict],
    teams: list[str],
) -> list[dict]:
    """
    Calculate current standings for a group from known results.
    Returns list of {team, played, won, drawn, lost, gf, ga, gd, pts}
    sorted by pts desc, gd desc, gf desc.
    """
    stats: dict[str, dict] = {
        t: {"played": 0, "won": 0, "drawn": 0, "lost": 0,
            "gf": 0, "ga": 0}
        for t in teams
    }

    group_results = [r for r in results if r["group"] == group]

    for r in group_results:
        home = r["home"]
        away = r["away"]
        hs = int(r["home_score"])
        as_ = int(r["away_score"])

        if home not in stats or away not in stats:
            continue  # skip if team not in this group (shouldn't happen)

        stats[home]["played"] += 1
        stats[away]["played"] += 1
        stats[home]["gf"] += hs
        stats[home]["ga"] += as_
        stats[away]["gf"] += as_
        stats[away]["ga"] += hs

        if hs > as_:
            stats[home]["won"] += 1
            stats[away]["lost"] += 1
        elif hs < as_:
            stats[away]["won"] += 1
            stats[home]["lost"] += 1
        else:
            stats[home]["drawn"] += 1
            stats[away]["drawn"] += 1

    rows = []
    for t in teams:
        s = stats[t]
        pts = s["won"] * 3 + s["drawn"]
        gd = s["gf"] - s["ga"]
        rows.append(
            {
                "team": t,
                "played": s["played"],
                "won": s["won"],
                "drawn": s["drawn"],
                "lost": s["lost"],
                "gf": s["gf"],
                "ga": s["ga"],
                "gd": gd,
                "pts": pts,
            }
        )

    rows.sort(key=lambda r: (r["pts"], r["gd"], r["gf"]), reverse=True)
    return rows


def get_unplayed_fixtures(
    schedule: list[dict],
    results: list[dict],
) -> list[dict]:
    """
    Return fixtures from schedule that do not yet have a recorded result.
    A fixture is considered played if there is a result with the same
    home and away team (order matters — fixtures are always stored canonically).
    """
    played_keys: set[tuple] = {
        (r["home"], r["away"]) for r in results
    }
    return [
        f for f in schedule
        if (f["home"], f["away"]) not in played_keys
    ]

"""
Compute Elo ratings from historical results or load from data/elo.csv.

Priority:
  1. data/elo.csv  (user-supplied, e.g. from eloratings.net)
  2. data/results.csv  (computed via iterative Elo)
  3. Flat 1500 for all teams (fallback, warns loudly)
"""

import os
import pandas as pd
from src.config import ALL_TEAMS, TEAM_ALIASES, ELO_ALIASES

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Elo hyperparameters (calibrated on international football)
K_BASE = 32
INITIAL_ELO = 1500
HOME_ADVANTAGE_ELO = 100  # Elo-point boost for the home team

# Competition importance multipliers
_COMP_WEIGHTS = {
    "fifa world cup": 1.5,
    "copa america": 1.25,
    "uefa euro": 1.25,
    "africa cup": 1.25,
    "afcon": 1.25,
    "nations cup": 1.25,
    "gold cup": 1.10,
    "asian cup": 1.25,
    "confederations cup": 1.10,
}


def _comp_weight(tournament_name: str) -> float:
    t = tournament_name.lower()
    for key, w in _COMP_WEIGHTS.items():
        if key in t:
            return w
    return 1.0


def _goal_diff_multiplier(gd: int) -> float:
    """World Football Elo-style goal-difference K multiplier."""
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return 1.75 + (gd - 3) / 8.0


def compute_elo_from_results(results_path: str | None = None) -> dict[str, float]:
    """Iterate through historical results and return final Elo dict."""
    if results_path is None:
        results_path = os.path.join(DATA_DIR, "results.csv")

    df = pd.read_csv(results_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    elo: dict[str, float] = {}

    for row in df.itertuples(index=False):
        home = row.home_team
        away = row.away_team

        if home not in elo:
            elo[home] = INITIAL_ELO
        if away not in elo:
            elo[away] = INITIAL_ELO

        neutral = getattr(row, "neutral", False)
        home_adv = 0.0 if neutral else float(HOME_ADVANTAGE_ELO)

        E_home = 1.0 / (1.0 + 10.0 ** ((elo[away] - elo[home] - home_adv) / 400.0))
        E_away = 1.0 - E_home

        hs, as_ = int(row.home_score), int(row.away_score)
        if hs > as_:
            S_home, S_away = 1.0, 0.0
        elif hs < as_:
            S_home, S_away = 0.0, 1.0
        else:
            S_home, S_away = 0.5, 0.5

        tournament = getattr(row, "tournament", "friendly")
        comp_w = _comp_weight(str(tournament))
        K = K_BASE * comp_w * _goal_diff_multiplier(abs(hs - as_))

        elo[home] += K * (S_home - E_home)
        elo[away] += K * (S_away - E_away)

    return elo


def load_elo_csv(elo_path: str | None = None) -> dict[str, float]:
    """Load team,elo CSV and return as dict."""
    if elo_path is None:
        elo_path = os.path.join(DATA_DIR, "elo.csv")
    df = pd.read_csv(elo_path)
    return dict(zip(df["team"].str.strip(), df["elo"].astype(float)))


def get_elo_ratings(verbose: bool = True) -> dict[str, float]:
    """
    Return Elo ratings keyed by canonical team name (as in config.GROUPS).
    Warns for any WC team that could not be found.
    """
    elo_path = os.path.join(DATA_DIR, "elo.csv")
    results_path = os.path.join(DATA_DIR, "results.csv")

    if os.path.exists(elo_path):
        if verbose:
            print("Loading Elo from data/elo.csv")
        raw = load_elo_csv(elo_path)
    elif os.path.exists(results_path):
        if verbose:
            print("Computing Elo from data/results.csv …")
        raw = compute_elo_from_results(results_path)
    else:
        if verbose:
            print("WARNING: Neither data/elo.csv nor data/results.csv found.")
            print("         Using flat 1500 for all teams.")
        return {team: INITIAL_ELO for team in ALL_TEAMS}

    # Resolve canonical name → dataset name → elo value
    out: dict[str, float] = {}
    missing: list[str] = []

    for team in ALL_TEAMS:
        # When loading from elo.csv, try the ELO_ALIASES first (handles
        # accent/spelling differences), then canonical, then TEAM_ALIASES.
        elo_name = ELO_ALIASES.get(team, team)
        if elo_name in raw:
            out[team] = raw[elo_name]
        elif team in raw:
            out[team] = raw[team]
        else:
            dataset_name = TEAM_ALIASES.get(team, team)
            if dataset_name in raw:
                out[team] = raw[dataset_name]
            else:
                missing.append(f"  '{team}' (tried '{elo_name}', '{dataset_name}')")
                out[team] = INITIAL_ELO

    if missing and verbose:
        print("\nWARNING: Could not find Elo data for the following teams.")
        print("Add entries to TEAM_ALIASES in src/config.py to fix this.")
        for m in missing:
            print(m)

    # Blend with FIFA ranking points if the file exists
    fifa_path = os.path.join(DATA_DIR, "fifa_rankings.csv")
    if os.path.exists(fifa_path):
        out = _blend_with_fifa(out, fifa_path, elo_weight=0.6, verbose=verbose)

    return out


def _blend_with_fifa(
    elo_ratings: dict[str, float],
    fifa_path: str,
    elo_weight: float = 0.6,
    verbose: bool = True,
) -> dict[str, float]:
    """
    Blend Elo ratings (60%) with FIFA ranking points (40%), both
    normalised to the same scale before mixing.

    FIFA ranking points are scaled linearly so that the weakest
    WC team maps to the minimum Elo and the strongest to the maximum Elo.
    Teams absent from the FIFA file keep their raw Elo.
    """
    df = pd.read_csv(fifa_path)
    df.columns = df.columns.str.strip().str.lower()
    fifa_map: dict[str, float] = dict(zip(df["team"].str.strip(), df["points"].astype(float)))

    # Map canonical names to FIFA file spellings where needed
    FIFA_ALIASES: dict[str, str] = {
        "United States": "United States",
        "South Korea":   "South Korea",
        "Ivory Coast":   "Ivory Coast",
        "DR Congo":      "DR Congo",
        "Bosnia and Herzegovina": "Bosnia and Herzegovina",
        "Curaçao":       "Curaçao",
        "Czechia":       "Czechia",
    }

    # Gather FIFA points for WC teams
    fifa_vals: dict[str, float] = {}
    for team in elo_ratings:
        key = FIFA_ALIASES.get(team, team)
        if key in fifa_map:
            fifa_vals[team] = fifa_map[key]
        elif team in fifa_map:
            fifa_vals[team] = fifa_map[team]

    if not fifa_vals:
        return elo_ratings

    # Normalise FIFA points onto the Elo value range
    elo_min = min(elo_ratings.values())
    elo_max = max(elo_ratings.values())
    f_min   = min(fifa_vals.values())
    f_max   = max(fifa_vals.values())

    def _scale(v: float) -> float:
        if f_max == f_min:
            return (elo_min + elo_max) / 2
        return (v - f_min) / (f_max - f_min) * (elo_max - elo_min) + elo_min

    fifa_weight = 1.0 - elo_weight
    blended: dict[str, float] = {}
    for team, elo_val in elo_ratings.items():
        if team in fifa_vals:
            blended[team] = elo_weight * elo_val + fifa_weight * _scale(fifa_vals[team])
        else:
            blended[team] = elo_val

    if verbose:
        n_blended = sum(1 for t in elo_ratings if t in fifa_vals)
        print(f"FIFA blend applied: {n_blended}/{len(elo_ratings)} teams "
              f"({elo_weight:.0%} Elo + {fifa_weight:.0%} FIFA)")

    return blended


def get_historical_h2h(team_a: str, team_b: str) -> pd.DataFrame:
    """
    Return all historical matches between team_a and team_b from results.csv.
    Returns empty DataFrame if results.csv is unavailable.
    """
    results_path = os.path.join(DATA_DIR, "results.csv")
    if not os.path.exists(results_path):
        return pd.DataFrame()

    df = pd.read_csv(results_path, parse_dates=["date"])

    # Resolve aliases for both teams
    name_a = TEAM_ALIASES.get(team_a, team_a)
    name_b = TEAM_ALIASES.get(team_b, team_b)

    mask = (
        ((df["home_team"] == name_a) & (df["away_team"] == name_b))
        | ((df["home_team"] == name_b) & (df["away_team"] == name_a))
    )
    return df[mask].sort_values("date", ascending=False).reset_index(drop=True)

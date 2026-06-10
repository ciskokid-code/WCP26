"""2026 FIFA World Cup configuration — groups, aliases, and market odds."""

import os
import pandas as pd

# Official 2026 World Cup groups (12 groups × 4 teams)
# Canonical names match data/elo.csv spellings.
GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

ALL_TEAMS: list[str] = [t for teams in GROUPS.values() for t in teams]

# canonical name → name used in data/results.csv (Kaggle dataset)
# Run `python -m src.precompute` to see any remaining mismatches.
TEAM_ALIASES: dict[str, str] = {
    "Czechia": "Czech Republic",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "DR Congo": "DR Congo",
}

# canonical name → name used in data/elo.csv
# Needed when elo.csv spelling differs from canonical.
ELO_ALIASES: dict[str, str] = {
    "Curaçao": "Curacao",   # elo.csv omits the cedilla
}

HOSTS: list[str] = ["United States", "Canada", "Mexico"]

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_market_odds() -> dict[str, float]:
    """
    Load decimal odds from data/market_odds.csv (columns: team, decimal_odds).
    Teams absent from the file are omitted from the returned dict.
    """
    path = os.path.join(_DATA_DIR, "market_odds.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    odds_col = next((c for c in df.columns if "odd" in c), None)
    if odds_col is None:
        return {}
    return dict(zip(df["team"].str.strip(), df[odds_col].astype(float)))

"""
2026 FIFA World Cup — complete 72-match group stage schedule.

Round-robin pattern for teams [t1, t2, t3, t4] (index 0-3 in GROUPS order):
  MD1: t1 vs t2, t3 vs t4
  MD2: t1 vs t3, t2 vs t4
  MD3: t1 vs t4, t2 vs t3
"""

from __future__ import annotations

from src.config import GROUPS

# ---------------------------------------------------------------------------
# Raw fixture data — 72 matches, Jun 11-27 2026
# ---------------------------------------------------------------------------

_FIXTURES: list[dict] = [
    # ── Matchday 1 ──────────────────────────────────────────────────────────
    # Jun 11 — Group A
    {"date": "2026-06-11", "group": "A", "matchday": 1, "home": "Mexico",       "away": "South Africa"},
    {"date": "2026-06-11", "group": "A", "matchday": 1, "home": "South Korea",  "away": "Czechia"},
    # Jun 12 — Group B
    {"date": "2026-06-12", "group": "B", "matchday": 1, "home": "Canada",       "away": "Bosnia and Herzegovina"},
    {"date": "2026-06-12", "group": "B", "matchday": 1, "home": "Qatar",        "away": "Switzerland"},
    # Jun 13 — Groups C & D
    {"date": "2026-06-13", "group": "C", "matchday": 1, "home": "Brazil",       "away": "Morocco"},
    {"date": "2026-06-13", "group": "C", "matchday": 1, "home": "Haiti",        "away": "Scotland"},
    {"date": "2026-06-13", "group": "D", "matchday": 1, "home": "United States","away": "Paraguay"},
    {"date": "2026-06-13", "group": "D", "matchday": 1, "home": "Australia",    "away": "Turkey"},
    # Jun 14 — Groups E, F & H
    {"date": "2026-06-14", "group": "E", "matchday": 1, "home": "Germany",      "away": "Curaçao"},
    {"date": "2026-06-14", "group": "E", "matchday": 1, "home": "Ivory Coast",  "away": "Ecuador"},
    {"date": "2026-06-14", "group": "F", "matchday": 1, "home": "Netherlands",  "away": "Japan"},
    {"date": "2026-06-14", "group": "F", "matchday": 1, "home": "Sweden",       "away": "Tunisia"},
    {"date": "2026-06-14", "group": "H", "matchday": 1, "home": "Spain",        "away": "Cape Verde"},
    {"date": "2026-06-14", "group": "H", "matchday": 1, "home": "Saudi Arabia", "away": "Uruguay"},
    # Jun 15 — Groups G & I
    {"date": "2026-06-15", "group": "G", "matchday": 1, "home": "Belgium",      "away": "Egypt"},
    {"date": "2026-06-15", "group": "G", "matchday": 1, "home": "Iran",         "away": "New Zealand"},
    {"date": "2026-06-15", "group": "I", "matchday": 1, "home": "France",       "away": "Senegal"},
    {"date": "2026-06-15", "group": "I", "matchday": 1, "home": "Iraq",         "away": "Norway"},
    # Jun 16 — Groups J & K
    {"date": "2026-06-16", "group": "J", "matchday": 1, "home": "Argentina",    "away": "Algeria"},
    {"date": "2026-06-16", "group": "J", "matchday": 1, "home": "Austria",      "away": "Jordan"},
    {"date": "2026-06-16", "group": "K", "matchday": 1, "home": "Portugal",     "away": "DR Congo"},
    {"date": "2026-06-16", "group": "K", "matchday": 1, "home": "Uzbekistan",   "away": "Colombia"},
    # Jun 17 — Group L
    {"date": "2026-06-17", "group": "L", "matchday": 1, "home": "England",      "away": "Croatia"},
    {"date": "2026-06-17", "group": "L", "matchday": 1, "home": "Ghana",        "away": "Panama"},

    # ── Matchday 2 ──────────────────────────────────────────────────────────
    # MD2 pattern: t1 vs t3, t2 vs t4
    # Jun 18 — Groups A & B
    {"date": "2026-06-18", "group": "A", "matchday": 2, "home": "Mexico",       "away": "South Korea"},
    {"date": "2026-06-18", "group": "A", "matchday": 2, "home": "South Africa", "away": "Czechia"},
    {"date": "2026-06-18", "group": "B", "matchday": 2, "home": "Canada",       "away": "Qatar"},
    {"date": "2026-06-18", "group": "B", "matchday": 2, "home": "Bosnia and Herzegovina", "away": "Switzerland"},
    # Jun 19 — Groups C & D
    {"date": "2026-06-19", "group": "C", "matchday": 2, "home": "Brazil",       "away": "Haiti"},
    {"date": "2026-06-19", "group": "C", "matchday": 2, "home": "Morocco",      "away": "Scotland"},
    {"date": "2026-06-19", "group": "D", "matchday": 2, "home": "United States","away": "Australia"},
    {"date": "2026-06-19", "group": "D", "matchday": 2, "home": "Paraguay",     "away": "Turkey"},
    # Jun 20 — Groups E & F
    {"date": "2026-06-20", "group": "E", "matchday": 2, "home": "Germany",      "away": "Ivory Coast"},
    {"date": "2026-06-20", "group": "E", "matchday": 2, "home": "Curaçao",      "away": "Ecuador"},
    {"date": "2026-06-20", "group": "F", "matchday": 2, "home": "Netherlands",  "away": "Sweden"},
    {"date": "2026-06-20", "group": "F", "matchday": 2, "home": "Japan",        "away": "Tunisia"},
    # Jun 21 — Groups G & H
    {"date": "2026-06-21", "group": "G", "matchday": 2, "home": "Belgium",      "away": "Iran"},
    {"date": "2026-06-21", "group": "G", "matchday": 2, "home": "Egypt",        "away": "New Zealand"},
    {"date": "2026-06-21", "group": "H", "matchday": 2, "home": "Spain",        "away": "Saudi Arabia"},
    {"date": "2026-06-21", "group": "H", "matchday": 2, "home": "Cape Verde",   "away": "Uruguay"},
    # Jun 22 — Groups I, J, K & L
    {"date": "2026-06-22", "group": "I", "matchday": 2, "home": "France",       "away": "Iraq"},
    {"date": "2026-06-22", "group": "I", "matchday": 2, "home": "Senegal",      "away": "Norway"},
    {"date": "2026-06-22", "group": "J", "matchday": 2, "home": "Argentina",    "away": "Austria"},
    {"date": "2026-06-22", "group": "J", "matchday": 2, "home": "Algeria",      "away": "Jordan"},
    {"date": "2026-06-22", "group": "K", "matchday": 2, "home": "Portugal",     "away": "Uzbekistan"},
    {"date": "2026-06-22", "group": "K", "matchday": 2, "home": "DR Congo",     "away": "Colombia"},
    {"date": "2026-06-22", "group": "L", "matchday": 2, "home": "England",      "away": "Ghana"},
    {"date": "2026-06-22", "group": "L", "matchday": 2, "home": "Croatia",      "away": "Panama"},

    # ── Matchday 3 (same-day concurrent within each group) ──────────────────
    # MD3 pattern: t1 vs t4, t2 vs t3
    # Jun 23 — Groups A & B
    {"date": "2026-06-23", "group": "A", "matchday": 3, "home": "Mexico",       "away": "Czechia"},
    {"date": "2026-06-23", "group": "A", "matchday": 3, "home": "South Africa", "away": "South Korea"},
    {"date": "2026-06-23", "group": "B", "matchday": 3, "home": "Canada",       "away": "Switzerland"},
    {"date": "2026-06-23", "group": "B", "matchday": 3, "home": "Bosnia and Herzegovina", "away": "Qatar"},
    # Jun 24 — Groups C & D
    {"date": "2026-06-24", "group": "C", "matchday": 3, "home": "Brazil",       "away": "Scotland"},
    {"date": "2026-06-24", "group": "C", "matchday": 3, "home": "Morocco",      "away": "Haiti"},
    {"date": "2026-06-24", "group": "D", "matchday": 3, "home": "United States","away": "Turkey"},
    {"date": "2026-06-24", "group": "D", "matchday": 3, "home": "Paraguay",     "away": "Australia"},
    # Jun 25 — Groups E & F
    {"date": "2026-06-25", "group": "E", "matchday": 3, "home": "Germany",      "away": "Ecuador"},
    {"date": "2026-06-25", "group": "E", "matchday": 3, "home": "Curaçao",      "away": "Ivory Coast"},
    {"date": "2026-06-25", "group": "F", "matchday": 3, "home": "Netherlands",  "away": "Tunisia"},
    {"date": "2026-06-25", "group": "F", "matchday": 3, "home": "Japan",        "away": "Sweden"},
    # Jun 25 — Groups G & H (same day, different kickoffs)
    {"date": "2026-06-25", "group": "G", "matchday": 3, "home": "Belgium",      "away": "New Zealand"},
    {"date": "2026-06-25", "group": "G", "matchday": 3, "home": "Egypt",        "away": "Iran"},
    {"date": "2026-06-25", "group": "H", "matchday": 3, "home": "Spain",        "away": "Uruguay"},
    {"date": "2026-06-25", "group": "H", "matchday": 3, "home": "Cape Verde",   "away": "Saudi Arabia"},
    # Jun 26 — Groups I & J
    {"date": "2026-06-26", "group": "I", "matchday": 3, "home": "France",       "away": "Norway"},
    {"date": "2026-06-26", "group": "I", "matchday": 3, "home": "Senegal",      "away": "Iraq"},
    {"date": "2026-06-26", "group": "J", "matchday": 3, "home": "Argentina",    "away": "Jordan"},
    {"date": "2026-06-26", "group": "J", "matchday": 3, "home": "Algeria",      "away": "Austria"},
    # Jun 27 — Groups K & L
    {"date": "2026-06-27", "group": "K", "matchday": 3, "home": "Portugal",     "away": "Colombia"},
    {"date": "2026-06-27", "group": "K", "matchday": 3, "home": "DR Congo",     "away": "Uzbekistan"},
    {"date": "2026-06-27", "group": "L", "matchday": 3, "home": "England",      "away": "Panama"},
    {"date": "2026-06-27", "group": "L", "matchday": 3, "home": "Croatia",      "away": "Ghana"},
]


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------

def get_schedule() -> list[dict]:
    """Return all 72 group-stage fixtures sorted by date."""
    return sorted(_FIXTURES, key=lambda f: (f["date"], f["group"], f["matchday"]))


def get_group_fixtures(group: str) -> list[dict]:
    """Return all fixtures for a single group letter (e.g. 'A')."""
    return [f for f in _FIXTURES if f["group"] == group]


def get_fixtures_by_date(date_str: str) -> list[dict]:
    """Return all fixtures on a given date string (e.g. '2026-06-11')."""
    return [f for f in _FIXTURES if f["date"] == date_str]

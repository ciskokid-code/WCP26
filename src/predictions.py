"""
Match prediction and Monte Carlo tournament simulation.

Match model
-----------
Elo difference → per-team expected-goal rate (μ) via a calibrated logistic
mapping, then goals drawn from independent Poisson(μ) distributions.
Knockout matches add extra-time (reduced μ) and penalties if still tied.

Tournament format (2026)
-----------------------
12 groups × 4 teams → top-2 + 8 best 3rd-placers = 32 qualifiers
→ R32 → R16 → QF → SF → Final
"""

from __future__ import annotations

import numpy as np
from itertools import combinations
from scipy.stats import poisson

from src.config import GROUPS

# ---------------------------------------------------------------------------
# Goal expectation model
# ---------------------------------------------------------------------------

# Calibrated so that μ = 1.15 at Elo parity and shifts ±0.35 at ±400-pt gap
_BASE_GOALS = 1.15
_ELO_SCALE = 400.0
_GOAL_SENSITIVITY = 0.35  # max shift from base at ±400 Elo pts


def elo_to_expected_goals(elo_a: float, elo_b: float) -> tuple[float, float]:
    """Return (mu_a, mu_b) expected goals for a neutral-venue match."""
    dr = elo_a - elo_b
    E_a = 1.0 / (1.0 + 10.0 ** (-dr / _ELO_SCALE))
    mu_a = _BASE_GOALS * (1.0 + _GOAL_SENSITIVITY * (E_a - 0.5) * 2.0)
    mu_b = _BASE_GOALS * (1.0 - _GOAL_SENSITIVITY * (E_a - 0.5) * 2.0)
    return max(0.15, mu_a), max(0.15, mu_b)


# ---------------------------------------------------------------------------
# Analytical match probabilities (no simulation needed)
# ---------------------------------------------------------------------------

def predict_match(elo_a: float, elo_b: float, max_goals: int = 10) -> dict:
    """
    Return analytical W/D/L probabilities and expected score for a single match.
    Useful for the head-to-head page without running a simulation.
    """
    mu_a, mu_b = elo_to_expected_goals(elo_a, elo_b)
    win = draw = loss = 0.0

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = poisson.pmf(i, mu_a) * poisson.pmf(j, mu_b)
            if i > j:
                win += p
            elif i == j:
                draw += p
            else:
                loss += p

    return {
        "win": win,
        "draw": draw,
        "loss": loss,
        "mu_a": mu_a,
        "mu_b": mu_b,
        "most_likely_score": _most_likely_score(mu_a, mu_b, max_goals),
    }


def _most_likely_score(mu_a: float, mu_b: float, max_goals: int = 8) -> tuple[int, int]:
    best_p, best = -1.0, (1, 1)
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = poisson.pmf(i, mu_a) * poisson.pmf(j, mu_b)
            if p > best_p:
                best_p, best = p, (i, j)
    return best


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _sim_match(mu_a: float, mu_b: float, rng: np.random.Generator) -> tuple[int, int]:
    return int(rng.poisson(mu_a)), int(rng.poisson(mu_b))


def _sim_ko_match(elo_a: float, elo_b: float, rng: np.random.Generator) -> str:
    """Simulate one knockout match (ET + pens if tied). Returns 'a' or 'b'."""
    mu_a, mu_b = elo_to_expected_goals(elo_a, elo_b)
    ga, gb = _sim_match(mu_a, mu_b, rng)
    if ga != gb:
        return "a" if ga > gb else "b"

    # Extra time: ~30 min ≈ 0.43 × 90-min rate
    et_a = int(rng.poisson(mu_a * 0.43))
    et_b = int(rng.poisson(mu_b * 0.43))
    if et_a != et_b:
        return "a" if et_a > et_b else "b"

    # Penalties: coin-flip with tiny Elo edge (±5 pp max at 400-pt gap)
    dr = elo_a - elo_b
    p_a_wins_pens = np.clip(0.5 + 0.000125 * dr, 0.35, 0.65)
    return "a" if rng.random() < p_a_wins_pens else "b"


# ---------------------------------------------------------------------------
# Group stage simulation
# ---------------------------------------------------------------------------

def _simulate_group(
    teams: list[str],
    elos: dict[str, float],
    rng: np.random.Generator,
    known_results: dict | None = None,
) -> list[tuple[str, int, int, int]]:
    """
    Simulate one 4-team group (round-robin).
    Returns list of (team, points, gf, ga) sorted by group-stage tiebreakers.

    known_results: {(home, away): (home_goals, away_goals)} for already-played
    matches. If (t1, t2) or (t2, t1) is present, use that score instead of
    simulating.
    """
    known_results = known_results or {}
    stats: dict[str, dict] = {t: {"pts": 0, "gf": 0, "ga": 0} for t in teams}

    for t1, t2 in combinations(teams, 2):
        if (t1, t2) in known_results:
            ga, gb = known_results[(t1, t2)]
        elif (t2, t1) in known_results:
            gb, ga = known_results[(t2, t1)]
        else:
            mu_a, mu_b = elo_to_expected_goals(elos[t1], elos[t2])
            ga, gb = _sim_match(mu_a, mu_b, rng)
        stats[t1]["gf"] += ga
        stats[t1]["ga"] += gb
        stats[t2]["gf"] += gb
        stats[t2]["ga"] += ga
        if ga > gb:
            stats[t1]["pts"] += 3
        elif ga == gb:
            stats[t1]["pts"] += 1
            stats[t2]["pts"] += 1
        else:
            stats[t2]["pts"] += 3

    standings = sorted(
        teams,
        key=lambda t: (
            stats[t]["pts"],
            stats[t]["gf"] - stats[t]["ga"],
            stats[t]["gf"],
        ),
        reverse=True,
    )
    return [(t, stats[t]["pts"], stats[t]["gf"], stats[t]["ga"]) for t in standings]


# ---------------------------------------------------------------------------
# Bracket construction
# ---------------------------------------------------------------------------

def _build_r32_bracket(
    group_results: dict[str, list[tuple]],
) -> tuple[list[tuple[str, str]], list[str]]:
    """
    Given group results, pick 32 qualifiers and return a list of 16 R32 matchups.

    Seeding:
      seeds 1-12  → group winners  (A1, B1, … L1)
      seeds 13-24 → runners-up     (A2, B2, … L2)
      seeds 25-32 → 8 best 3rd-place teams (by pts / GD / GF)

    Bracket: seed 1 vs 32, 2 vs 31, …, 16 vs 17  (standard balanced bracket)
    Returns (matchups, qualifiers_ordered_1_to_32)
    """
    group_order = list(GROUPS.keys())  # A … L

    winners = [group_results[g][0][0] for g in group_order]     # 12 teams
    runners = [group_results[g][1][0] for g in group_order]     # 12 teams
    thirds = [
        {
            "team": group_results[g][2][0],
            "pts": group_results[g][2][1],
            "gf": group_results[g][2][2],
            "ga": group_results[g][2][3],
        }
        for g in group_order
    ]

    thirds_sorted = sorted(
        thirds,
        key=lambda r: (r["pts"], r["gf"] - r["ga"], r["gf"]),
        reverse=True,
    )
    best8_thirds = [r["team"] for r in thirds_sorted[:8]]

    # Seeds 1-32
    seeds = winners + runners + best8_thirds  # length 32
    matchups = [(seeds[i], seeds[31 - i]) for i in range(16)]
    return matchups, seeds


# ---------------------------------------------------------------------------
# Full tournament simulation
# ---------------------------------------------------------------------------

def _run_knockout_round(
    pairs: list[tuple[str, str]],
    elos: dict[str, float],
    rng: np.random.Generator,
) -> list[str]:
    winners = []
    for t_a, t_b in pairs:
        result = _sim_ko_match(elos[t_a], elos[t_b], rng)
        winners.append(t_a if result == "a" else t_b)
    return winners


def run_simulation(
    elos: dict[str, float],
    n_sims: int = 50_000,
    seed: int | None = 42,
    known_group_results: dict | None = None,
) -> dict[str, dict[str, float]]:
    """
    Monte Carlo simulation of the 2026 World Cup.

    Returns a nested dict:
      result["champion"]["Brazil"]   → probability of winning the tournament
      result["qualified"]["France"]  → probability of reaching R32
      etc.

    Keys: champion, finalist, semi, quarter, r16, group_win, qualified

    known_group_results: optional dict from build_known_group_results()
      {group_letter: {(home, away): (home_goals, away_goals)}}
    """
    known_group_results = known_group_results or {}
    rng = np.random.default_rng(seed)
    all_teams = list(elos.keys())

    counters: dict[str, dict[str, int]] = {
        stage: {t: 0 for t in all_teams}
        for stage in ("champion", "finalist", "semi", "quarter", "r16", "group_win", "qualified")
    }

    group_order = list(GROUPS.keys())

    for _ in range(n_sims):
        # ── Group stage ───────────────────────────────────────────────────
        group_results: dict[str, list[tuple]] = {}
        for g, teams in GROUPS.items():
            group_results[g] = _simulate_group(
                teams, elos, rng, known_results=known_group_results.get(g, {})
            )

        for g in group_order:
            counters["group_win"][group_results[g][0][0]] += 1
            for finish in range(2):          # top 2 always advance
                counters["qualified"][group_results[g][finish][0]] += 1

        # ── Build R32 bracket ─────────────────────────────────────────────
        r32_pairs, seeds_32 = _build_r32_bracket(group_results)

        # Seeds 25-32 are the best-8 3rd-place qualifiers
        for t in seeds_32[24:]:
            counters["qualified"][t] += 1

        # ── R32 ───────────────────────────────────────────────────────────
        r16_teams = _run_knockout_round(r32_pairs, elos, rng)
        for t in r16_teams:
            counters["r16"][t] += 1

        # ── R16 (Quarterfinals) ───────────────────────────────────────────
        r16_pairs = [(r16_teams[i], r16_teams[i + 1]) for i in range(0, 16, 2)]
        qf_teams = _run_knockout_round(r16_pairs, elos, rng)
        for t in qf_teams:
            counters["quarter"][t] += 1

        # ── Quarterfinals (Semis) ─────────────────────────────────────────
        qf_pairs = [(qf_teams[i], qf_teams[i + 1]) for i in range(0, 8, 2)]
        sf_teams = _run_knockout_round(qf_pairs, elos, rng)
        for t in sf_teams:
            counters["semi"][t] += 1

        # ── Semifinals (Final) ────────────────────────────────────────────
        sf_pairs = [(sf_teams[i], sf_teams[i + 1]) for i in range(0, 4, 2)]
        finalists = _run_knockout_round(sf_pairs, elos, rng)
        for t in finalists:
            counters["finalist"][t] += 1

        # ── Final ─────────────────────────────────────────────────────────
        champion = _run_knockout_round([tuple(finalists)], elos, rng)[0]  # type: ignore[arg-type]
        counters["champion"][champion] += 1

    # Normalize to probabilities
    return {
        stage: {t: v / n_sims for t, v in stage_counts.items()}
        for stage, stage_counts in counters.items()
    }


def run_group_simulation(
    group_letter: str,
    elos: dict[str, float],
    n_sims: int = 50_000,
    seed: int | None = 42,
    known_results: dict | None = None,
) -> dict[str, dict[str, float]]:
    """
    Simulate a single group n_sims times.
    Returns {team: {1st: prob, 2nd: prob, 3rd: prob, 4th: prob}}.

    known_results: {(home, away): (home_goals, away_goals)} for already-played
    fixtures in this group.
    """
    known_results = known_results or {}
    rng = np.random.default_rng(seed)
    teams = GROUPS[group_letter]
    finish_counts = {t: {1: 0, 2: 0, 3: 0, 4: 0} for t in teams}

    for _ in range(n_sims):
        standings = _simulate_group(teams, elos, rng, known_results=known_results)
        for rank, (team, *_rest) in enumerate(standings, start=1):
            finish_counts[team][rank] += 1

    return {
        team: {rank: count / n_sims for rank, count in counts.items()}
        for team, counts in finish_counts.items()
    }

"""
Precompute and cache all simulation results.

Usage:
    python -m src.precompute          # 50 000 sims (default)
    python -m src.precompute --sims 100000
    python -m src.precompute --sims 10000 --seed 0

Outputs:
    cache/simulation.pkl   – full tournament + per-group probabilities
    cache/elo.pkl          – resolved Elo ratings for all WC teams
"""

from __future__ import annotations

import argparse
import os
import pickle
import time

from src.config import GROUPS
from src.elo_model import get_elo_ratings
from src.predictions import run_simulation, run_group_simulation

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")


def precompute(n_sims: int = 50_000, seed: int = 42) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print("  2026 World Cup Model — Precompute")
    print(f"  Simulations : {n_sims:,}")
    print(f"  Seed        : {seed}")
    print(f"{'='*60}\n")

    # 1. Elo ratings
    elos = get_elo_ratings(verbose=True)
    print(f"\nElo ratings loaded for {len(elos)} teams.")
    top10 = sorted(elos.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop-10 by Elo:")
    for rank, (team, elo) in enumerate(top10, 1):
        print(f"  {rank:2d}. {team:<30} {elo:.0f}")

    elo_path = os.path.join(CACHE_DIR, "elo.pkl")
    with open(elo_path, "wb") as f:
        pickle.dump(elos, f)
    print(f"\nElo saved → {elo_path}")

    # 2. Full tournament simulation
    print(f"\nRunning full tournament simulation ({n_sims:,} iterations)…")
    t0 = time.perf_counter()
    sim_results = run_simulation(elos, n_sims=n_sims, seed=seed)
    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s")

    # 3. Per-group simulations (for detailed group-stage view)
    print("\nRunning per-group simulations…")
    group_probs: dict[str, dict] = {}
    for g in GROUPS:
        group_probs[g] = run_group_simulation(g, elos, n_sims=n_sims, seed=seed)

    # 4. Bundle and save
    payload = {
        "simulation": sim_results,
        "group_probs": group_probs,
        "elos": elos,
        "n_sims": n_sims,
        "seed": seed,
    }
    sim_path = os.path.join(CACHE_DIR, "simulation.pkl")
    with open(sim_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Simulation saved → {sim_path}")

    # 5. Summary printout
    print("\n── Championship probabilities (top 20) ──")
    champ = sorted(
        sim_results["champion"].items(), key=lambda x: x[1], reverse=True
    )
    for team, prob in champ[:20]:
        bar = "█" * int(prob * 200)
        print(f"  {team:<30} {prob*100:5.1f}%  {bar}")

    print("\nPrecompute complete.\n")


def load_cache() -> dict | None:
    """Load cached payload, or return None if cache is missing."""
    sim_path = os.path.join(CACHE_DIR, "simulation.pkl")
    if not os.path.exists(sim_path):
        return None
    with open(sim_path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute WC 2026 simulation cache.")
    parser.add_argument("--sims", type=int, default=50_000, help="Number of Monte Carlo simulations")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    args = parser.parse_args()
    precompute(n_sims=args.sims, seed=args.seed)

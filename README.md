# 2026 World Cup — Exploratory Model Dashboard

Elo-based Poisson simulation of the 2026 FIFA World Cup.
Streamlit dashboard with four views: Title Odds, Group Stage, Head-to-Head, Tournament Path.

---

## Deploy to the web (share with friends)

### Option A — Streamlit Community Cloud (free, easiest)

1. Push this repo to GitHub (public or private).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
3. Click **New app** → pick your repo → set main file to `app.py` → **Deploy**.
4. Share the URL Streamlit gives you.

The simulation cache (`cache/simulation.pkl`) is committed to the repo, so the
app loads instantly with no cold-start computation.

### Option B — Railway (1-click, also free tier)

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template)

1. Push to GitHub.
2. Go to **[railway.app](https://railway.app)** → **New project → Deploy from GitHub repo**.
3. Railway auto-detects the `Dockerfile`.
4. Set a custom domain or use the generated `*.railway.app` URL.

---

## Local setup

```bash
pip install -r requirements.txt

# Put your data files in data/
#   data/results.csv   — Kaggle "International football results from 1872 to present"
#   data/elo.csv       — eloratings.net export (team, elo columns)
#   data/market_odds.csv — decimal odds (team, decimal_odds columns)

# Optional: regenerate the simulation cache
python -m src.precompute --sims 50000

streamlit run app.py
```

---

## Project layout

```
WCP/
├── app.py                  Streamlit dashboard (4 pages)
├── requirements.txt
├── Dockerfile              For Railway / Render / Docker deploy
├── .streamlit/config.toml  Dark theme + server settings
├── data/
│   ├── results.csv         Kaggle historical results
│   ├── elo.csv             Current Elo ratings (team, elo)
│   └── market_odds.csv     Pre-tournament odds (team, decimal_odds)
├── src/
│   ├── config.py           Groups, aliases, market-odds loader
│   ├── elo_model.py        Elo computation and loading
│   ├── predictions.py      Match prediction + Monte Carlo simulation
│   └── precompute.py       CLI to regenerate cache/simulation.pkl
└── cache/
    ├── simulation.pkl      Pre-baked results (committed to repo)
    └── elo.pkl
```

---

## Dashboard pages

| Page | What it shows |
|---|---|
| **Title Odds** | Model vs market championship probability for all 48 teams, edge chart, round-by-round funnel |
| **Group Stage** | Stacked-bar finish probabilities per group; scatter of win vs qualification probability |
| **Head-to-Head** | W/D/L probabilities, score heatmap, historical record from results.csv |
| **Tournament Path** | Funnel chart of a team's tournament probabilities; radar comparison of two teams |

---

## Model

- **Elo**: Loaded from `data/elo.csv` (eloratings.net). Falls back to computing iteratively from `results.csv` if absent.
- **Match model**: Expected goals μ derived from Elo difference (μ ≈ 1.15 at parity). Goals drawn from Poisson(μ).
- **Knockout matches**: Extra time (0.43 × base rate) → penalties if still tied.
- **Tournament format**: Top 2 per group + 8 best 3rd-place = 32 in R32. Standard seeded bracket.

---

## Updating market odds

Edit `data/market_odds.csv` — two columns: `team`, `decimal_odds`.
Teams absent from the file show no market line in the dashboard.

## Fixing team-name mismatches

Run `python -m src.precompute` — it prints a warning for any team it couldn't match.
Add the correction to `TEAM_ALIASES` (results.csv) or `ELO_ALIASES` (elo.csv) in `src/config.py`.

"""
2026 World Cup — Exploratory Model Dashboard
Run: streamlit run app.py
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import ALL_TEAMS, GROUPS, load_market_odds
from src.elo_model import get_elo_ratings, get_historical_h2h
from src.predictions import predict_match, run_group_simulation, run_simulation
from src.precompute import load_cache

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="WC 2026 Model",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

STAGE_LABELS = {
    "champion": "Win Tournament",
    "finalist": "Reach Final",
    "semi": "Reach Semis",
    "quarter": "Reach Quarters",
    "r16": "Reach R16",
    "qualified": "Qualify (Top 2 or Best 3rd)",
    "group_win": "Win Group",
}

FLAG_EMOJI: dict[str, str] = {
    "Mexico": "🇲🇽", "South Africa": "🇿🇦", "South Korea": "🇰🇷", "Czech Republic": "🇨🇿",
    "Canada": "🇨🇦", "Bosnia and Herzegovina": "🇧🇦", "Qatar": "🇶🇦", "Switzerland": "🇨🇭",
    "Brazil": "🇧🇷", "Morocco": "🇲🇦", "Haiti": "🇭🇹", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "United States": "🇺🇸", "Paraguay": "🇵🇾", "Australia": "🇦🇺", "Turkey": "🇹🇷",
    "Germany": "🇩🇪", "Curaçao": "🇨🇼", "Ivory Coast": "🇨🇮", "Ecuador": "🇪🇨",
    "Netherlands": "🇳🇱", "Japan": "🇯🇵", "Sweden": "🇸🇪", "Tunisia": "🇹🇳",
    "Belgium": "🇧🇪", "Egypt": "🇪🇬", "Iran": "🇮🇷", "New Zealand": "🇳🇿",
    "Spain": "🇪🇸", "Cape Verde": "🇨🇻", "Saudi Arabia": "🇸🇦", "Uruguay": "🇺🇾",
    "France": "🇫🇷", "Senegal": "🇸🇳", "Iraq": "🇮🇶", "Norway": "🇳🇴",
    "Argentina": "🇦🇷", "Algeria": "🇩🇿", "Austria": "🇦🇹", "Jordan": "🇯🇴",
    "Portugal": "🇵🇹", "DR Congo": "🇨🇩", "Uzbekistan": "🇺🇿", "Colombia": "🇨🇴",
    "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Croatia": "🇭🇷", "Ghana": "🇬🇭", "Panama": "🇵🇦",
}


def flag(team: str) -> str:
    return FLAG_EMOJI.get(team, "")


def fmt_pct(p: float) -> str:
    return f"{p*100:.1f}%"


# ──────────────────────────────────────────────────────────────────────────────
# Data loading (cached at session level)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Elo ratings…")
def _load_elos() -> dict[str, float]:
    return get_elo_ratings(verbose=False)


@st.cache_resource(show_spinner="Running 50 000-game simulation… (~10 s)")
def _load_simulation(n_sims: int, seed: int) -> dict:
    # Fast path: pre-baked pkl committed to the repo
    cached = load_cache()
    if cached and cached.get("n_sims", 0) >= n_sims:
        return cached

    # Slow path: compute in-process (happens on first deploy if cache/ is absent)
    elos = _load_elos()
    sim = run_simulation(elos, n_sims=n_sims, seed=seed)
    group_probs = {
        g: run_group_simulation(g, elos, n_sims=n_sims, seed=seed)
        for g in GROUPS
    }
    return {"simulation": sim, "group_probs": group_probs, "elos": elos, "n_sims": n_sims}


@st.cache_data(show_spinner=False)
def _load_h2h(team_a: str, team_b: str) -> pd.DataFrame:
    """Cached H2H lookup — results.csv is read once per team pair."""
    return get_historical_h2h(team_a, team_b)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚽ WC 2026 Model")
    st.caption("Elo + Poisson simulation")

    page = st.radio(
        "View",
        ["Title Odds", "Group Stage", "Head-to-Head", "Tournament Path"],
        index=0,
    )

    st.divider()
    n_sims = st.select_slider(
        "Simulations",
        options=[5_000, 10_000, 25_000, 50_000, 100_000],
        value=50_000,
        help="More sims = lower variance, but slower first load.",
    )
    seed = st.number_input("RNG seed", value=42, min_value=0, step=1)

    st.divider()
    st.caption("Data: Kaggle international results + eloratings.net (optional)")
    if st.button("Refresh cache"):
        st.cache_resource.clear()
        st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────────────

elos = _load_elos()
payload = _load_simulation(n_sims, seed)
sim = payload["simulation"]
group_probs_all = payload["group_probs"]
market_odds = load_market_odds()   # dict[team, decimal_odds] from data/market_odds.csv


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 1: Title Odds
# ──────────────────────────────────────────────────────────────────────────────

if page == "Title Odds":
    st.header("Title Odds — Model vs Market")
    st.caption(
        "Model: Elo-based Poisson simulation.  "
        "Market: implied probabilities from pre-tournament decimal odds (Pinnacle/Betfair)."
    )

    # Build comparison dataframe
    champ = sim["champion"]
    rows = []
    for team in ALL_TEAMS:
        model_p = champ.get(team, 0.0)
        mkt_odds = market_odds.get(team)
        mkt_p = (1.0 / mkt_odds) if mkt_odds else None
        rows.append(
            {
                "Team": f"{flag(team)} {team}",
                "Model %": round(model_p * 100, 2),
                "Market %": round(mkt_p * 100, 2) if mkt_p else None,
                "Edge (Model − Market)": (
                    round((model_p - mkt_p) * 100, 2) if mkt_p else None
                ),
                "_model": model_p,
                "_mkt": mkt_p,
            }
        )

    df = (
        pd.DataFrame(rows)
        .sort_values("_model", ascending=False)
        .reset_index(drop=True)
    )
    df.index = range(1, len(df) + 1)

    # Top filter
    col_a, col_b = st.columns([2, 1])
    with col_a:
        show_top_n = st.slider("Show top N teams", 5, 48, 20, key="top_n_title")
    with col_b:
        sort_by = st.selectbox("Sort by", ["Model %", "Market %", "Edge (Model − Market)"])

    df_top = (
        df.sort_values(sort_by, ascending=False, na_position="last")
        .head(show_top_n)
    )

    # Main chart
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Model",
            x=df_top["Team"],
            y=df_top["Model %"],
            marker_color="#1f77b4",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Market implied",
            x=df_top["Team"],
            y=df_top["Market %"],
            marker_color="#ff7f0e",
        )
    )
    fig.update_layout(
        barmode="group",
        xaxis_tickangle=-40,
        yaxis_title="Championship probability (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=480,
        margin=dict(t=20, b=120),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Edge chart (model − market)
    st.subheader("Model edge over market (pp)")
    df_edge = df_top.dropna(subset=["Edge (Model − Market)"]).copy()
    df_edge = df_edge.sort_values("Edge (Model − Market)", ascending=True)
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in df_edge["Edge (Model − Market)"]]
    fig2 = go.Figure(
        go.Bar(
            x=df_edge["Edge (Model − Market)"],
            y=df_edge["Team"],
            orientation="h",
            marker_color=colors,
        )
    )
    fig2.update_layout(
        xaxis_title="Edge (percentage points)",
        height=max(300, 22 * len(df_edge)),
        margin=dict(t=10, b=30),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Sortable table
    st.subheader("Full table")
    display_df = df[["Team", "Model %", "Market %", "Edge (Model − Market)"]].copy()
    display_df = display_df.sort_values(sort_by, ascending=False, na_position="last")
    st.dataframe(
        display_df.style.format(
            {"Model %": "{:.2f}", "Market %": "{:.2f}", "Edge (Model − Market)": "{:+.2f}"},
            na_rep="—",
        ).background_gradient(subset=["Model %"], cmap="Blues"),
        use_container_width=True,
        height=600,
    )

    st.divider()
    # Deep dive: round-by-round probabilities for one team
    st.subheader("Round-by-round probabilities")
    selected_team = st.selectbox(
        "Select team",
        sorted(ALL_TEAMS),
        format_func=lambda t: f"{flag(t)} {t}",
        key="rr_team",
    )
    stages = ["group_win", "qualified", "r16", "quarter", "semi", "finalist", "champion"]
    probs = [sim[s].get(selected_team, 0) for s in stages]
    labels = [STAGE_LABELS.get(s, s) for s in stages]

    fig3 = go.Figure(
        go.Bar(
            x=labels,
            y=[p * 100 for p in probs],
            marker_color="#1f77b4",
            text=[fmt_pct(p) for p in probs],
            textposition="outside",
        )
    )
    fig3.update_layout(
        yaxis=dict(title="Probability (%)", range=[0, 105]),
        height=380,
        margin=dict(t=10, b=60),
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(f"Elo rating: **{elos.get(selected_team, 1500):.0f}**  |  n_sims = {n_sims:,}")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 2: Group Stage
# ──────────────────────────────────────────────────────────────────────────────

elif page == "Group Stage":
    st.header("Group Stage Probabilities")
    st.caption(
        "Probability of each finishing position within the group, "
        "based on Monte Carlo simulation."
    )

    group_sel = st.selectbox(
        "Select group",
        list(GROUPS.keys()),
        format_func=lambda g: f"Group {g}",
    )

    teams = GROUPS[group_sel]
    gp = group_probs_all[group_sel]

    # Build display dataframe
    rows_g = []
    for team in teams:
        rows_g.append(
            {
                "Team": f"{flag(team)} {team}",
                "Elo": f"{elos.get(team, 1500):.0f}",
                "1st": gp[team][1],
                "2nd": gp[team][2],
                "3rd": gp[team][3],
                "4th": gp[team][4],
                "_elo": elos.get(team, 1500),
            }
        )
    df_g = pd.DataFrame(rows_g).sort_values("_elo", ascending=False).reset_index(drop=True)

    # Stacked bar chart
    fig_g = go.Figure()
    colors_pos = ["#2ca02c", "#98df8a", "#ffbb78", "#d62728"]
    labels_pos = ["1st", "2nd", "3rd", "4th"]

    for col, color in zip(labels_pos, colors_pos):
        fig_g.add_trace(
            go.Bar(
                name=col,
                x=df_g["Team"],
                y=df_g[col] * 100,
                marker_color=color,
                text=[fmt_pct(v) for v in df_g[col]],
                textposition="inside",
                insidetextanchor="middle",
            )
        )
    fig_g.update_layout(
        barmode="stack",
        yaxis=dict(title="Probability (%)", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
        margin=dict(t=20, b=60),
    )
    st.plotly_chart(fig_g, use_container_width=True)

    # Numeric table
    display_cols = ["Team", "Elo", "1st", "2nd", "3rd", "4th"]
    fmt = {c: "{:.1%}" for c in labels_pos}
    st.dataframe(
        df_g[display_cols]
        .style.format(fmt)
        .background_gradient(subset=["1st"], cmap="Greens")
        .background_gradient(subset=["4th"], cmap="Reds"),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # All groups overview — qualification probability heatmap
    st.subheader("Qualification probability — all groups")

    qual_rows = []
    for g, g_teams in GROUPS.items():
        for team in g_teams:
            gpp = group_probs_all[g][team]
            # "qualification prob" = 1st + 2nd (guaranteed) + 3rd (best-8 lottery)
            # Approximate 3rd-place qualification as 8/12 × P(3rd)
            qual_p = gpp[1] + gpp[2] + (8 / 12) * gpp[3]
            qual_rows.append(
                {
                    "Group": f"Group {g}",
                    "Team": f"{flag(team)} {team}",
                    "Win Group": gpp[1],
                    "Qualify (est.)": min(qual_p, 1.0),
                    "Elo": elos.get(team, 1500),
                }
            )

    df_qual = pd.DataFrame(qual_rows).sort_values(["Group", "Elo"], ascending=[True, False])

    fig_heat = px.scatter(
        df_qual,
        x="Qualify (est.)",
        y="Win Group",
        color="Group",
        text="Team",
        size="Elo",
        size_max=22,
        labels={"Qualify (est.)": "Est. qualification prob.", "Win Group": "Group-win prob."},
        title="Group win vs qualification probability (bubble size = Elo)",
    )
    fig_heat.update_traces(textposition="top center", textfont_size=9)
    fig_heat.update_layout(height=560, margin=dict(t=50, b=30))
    st.plotly_chart(fig_heat, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 3: Head-to-Head
# ──────────────────────────────────────────────────────────────────────────────

elif page == "Head-to-Head":
    st.header("Head-to-Head Match Prediction")

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox(
            "Team A",
            sorted(ALL_TEAMS),
            index=sorted(ALL_TEAMS).index("Brazil"),
            format_func=lambda t: f"{flag(t)} {t}",
        )
    with col2:
        team_b = st.selectbox(
            "Team B",
            sorted(ALL_TEAMS),
            index=sorted(ALL_TEAMS).index("Argentina"),
            format_func=lambda t: f"{flag(t)} {t}",
        )

    if team_a == team_b:
        st.warning("Select two different teams.")
        st.stop()

    elo_a = elos.get(team_a, 1500)
    elo_b = elos.get(team_b, 1500)
    result = predict_match(elo_a, elo_b)

    st.subheader(f"{flag(team_a)} {team_a}  vs  {flag(team_b)} {team_b}")

    # Probability gauge row
    win_a, draw, win_b = result["win"], result["draw"], result["loss"]
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 2])
    c1.metric(f"{flag(team_a)} Win", fmt_pct(win_a))
    c2.metric("Draw", fmt_pct(draw))
    c3.metric(f"{flag(team_b)} Win", fmt_pct(win_b))
    c4.metric(f"Elo {team_a[:10]}", f"{elo_a:.0f}")
    c5.metric(f"Elo {team_b[:10]}", f"{elo_b:.0f}")

    # Probability bar
    fig_bar = go.Figure(
        go.Bar(
            x=[win_a * 100, draw * 100, win_b * 100],
            y=[f"{flag(team_a)} {team_a}", "Draw", f"{flag(team_b)} {team_b}"],
            orientation="h",
            marker_color=["#1f77b4", "#7f7f7f", "#d62728"],
            text=[fmt_pct(win_a), fmt_pct(draw), fmt_pct(win_b)],
            textposition="auto",
        )
    )
    fig_bar.update_layout(
        xaxis=dict(title="Probability (%)", range=[0, 105]),
        height=200,
        margin=dict(t=10, b=30),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Score probability heatmap
    st.subheader("Score probability heatmap")
    mu_a, mu_b = result["mu_a"], result["mu_b"]
    max_g = 6
    score_matrix = np.zeros((max_g + 1, max_g + 1))
    from scipy.stats import poisson as _poisson

    for i in range(max_g + 1):
        for j in range(max_g + 1):
            score_matrix[i, j] = _poisson.pmf(i, mu_a) * _poisson.pmf(j, mu_b)

    df_heat = pd.DataFrame(
        score_matrix * 100,
        index=[f"{team_a[:3]} {i}" for i in range(max_g + 1)],
        columns=[f"{team_b[:3]} {j}" for j in range(max_g + 1)],
    )

    mls, mlg = result["most_likely_score"]
    fig_heat = px.imshow(
        df_heat,
        color_continuous_scale="Blues",
        labels={"color": "%"},
        title=f"Most likely score: {team_a} {mls} – {mlg} {team_b}",
        aspect="auto",
    )
    fig_heat.update_layout(height=380, margin=dict(t=50, b=30))
    st.plotly_chart(fig_heat, use_container_width=True)

    # Historical head-to-head
    st.subheader("Historical record")
    h2h = _load_h2h(team_a, team_b)

    if h2h.empty:
        st.info("No historical data found (data/results.csv not present or no matches between these teams).")
    else:
        from src.config import TEAM_ALIASES

        name_a = TEAM_ALIASES.get(team_a, team_a)
        name_b = TEAM_ALIASES.get(team_b, team_b)

        wins_a = ((h2h["home_team"] == name_a) & (h2h["home_score"] > h2h["away_score"])).sum() + \
                 ((h2h["away_team"] == name_a) & (h2h["away_score"] > h2h["home_score"])).sum()
        wins_b = ((h2h["home_team"] == name_b) & (h2h["home_score"] > h2h["away_score"])).sum() + \
                 ((h2h["away_team"] == name_b) & (h2h["away_score"] > h2h["home_score"])).sum()
        draws = len(h2h) - wins_a - wins_b

        ca, cb, cc, cd = st.columns(4)
        ca.metric("Total matches", len(h2h))
        cb.metric(f"{flag(team_a)} {team_a[:12]} wins", wins_a)
        cc.metric("Draws", draws)
        cd.metric(f"{flag(team_b)} {team_b[:12]} wins", wins_b)

        st.dataframe(
            h2h[["date", "home_team", "home_score", "away_score", "away_team", "tournament"]]
            .head(25)
            .rename(columns={"home_team": "Home", "away_team": "Away",
                              "home_score": "HG", "away_score": "AG",
                              "tournament": "Competition"}),
            use_container_width=True,
            hide_index=True,
        )
        if len(h2h) > 25:
            st.caption(f"Showing 25 of {len(h2h)} historical matches.")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 4: Tournament Path
# ──────────────────────────────────────────────────────────────────────────────

elif page == "Tournament Path":
    st.header("Tournament Path Probabilities")
    st.caption("Probability of reaching each knockout round, based on simulation.")

    selected = st.selectbox(
        "Select team",
        sorted(ALL_TEAMS),
        format_func=lambda t: f"{flag(t)} {t}",
        key="path_team",
    )

    stages_ordered = [
        "group_win", "qualified", "r16", "quarter", "semi", "finalist", "champion"
    ]
    stage_nice = [
        "Win Group", "Qualify", "Reach R16", "Reach QF", "Reach SF", "Reach Final", "Champion"
    ]

    probs = [sim[s].get(selected, 0) for s in stages_ordered]

    # Funnel chart
    fig_funnel = go.Figure(
        go.Funnel(
            y=stage_nice,
            x=[p * 100 for p in probs],
            textinfo="value+percent initial",
            marker_color=[
                "#d4e6f1", "#85c1e9", "#3498db", "#1a5276",
                "#e74c3c", "#c0392b", "#922b21"
            ],
        )
    )
    fig_funnel.update_layout(
        title=f"{flag(selected)} {selected} — Tournament path",
        height=450,
        margin=dict(t=60, b=20),
    )
    st.plotly_chart(fig_funnel, use_container_width=True)

    # Compared to group rivals
    group_of = next(
        (g for g, teams in GROUPS.items() if selected in teams), None
    )
    if group_of:
        st.subheader(f"Compared to Group {group_of} rivals")
        rivals = GROUPS[group_of]
        stage_cmp = "champion"

        cmp_data = {
            "Team": [f"{flag(t)} {t}" for t in rivals],
            "Elo": [f"{elos.get(t, 1500):.0f}" for t in rivals],
            **{
                STAGE_LABELS[s]: [fmt_pct(sim[s].get(t, 0)) for t in rivals]
                for s in ["group_win", "qualified", "r16", "champion"]
            },
        }
        st.dataframe(pd.DataFrame(cmp_data), use_container_width=True, hide_index=True)

    st.divider()

    # Compare any two teams
    st.subheader("Compare two teams")
    cmp_col1, cmp_col2 = st.columns(2)
    with cmp_col1:
        cmp_a = st.selectbox(
            "Team A",
            sorted(ALL_TEAMS),
            format_func=lambda t: f"{flag(t)} {t}",
            key="cmp_a",
        )
    with cmp_col2:
        cmp_b = st.selectbox(
            "Team B",
            sorted(ALL_TEAMS),
            format_func=lambda t: f"{flag(t)} {t}",
            key="cmp_b",
            index=1,
        )

    fig_cmp = go.Figure()
    for team, color in [(cmp_a, "#1f77b4"), (cmp_b, "#d62728")]:
        fig_cmp.add_trace(
            go.Scatterpolar(
                r=[sim[s].get(team, 0) * 100 for s in stages_ordered],
                theta=stage_nice,
                fill="toself",
                name=f"{flag(team)} {team}",
                line_color=color,
                opacity=0.7,
            )
        )
    fig_cmp.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        height=460,
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Raw numbers for comparison
    cmp_df_rows = []
    for s, label in zip(stages_ordered, stage_nice):
        cmp_df_rows.append(
            {
                "Stage": label,
                f"{flag(cmp_a)} {cmp_a}": fmt_pct(sim[s].get(cmp_a, 0)),
                f"{flag(cmp_b)} {cmp_b}": fmt_pct(sim[s].get(cmp_b, 0)),
            }
        )
    st.dataframe(pd.DataFrame(cmp_df_rows), use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Model: Elo-based Poisson simulation · "
    f"{payload['n_sims']:,} Monte Carlo iterations · "
    "Data: Kaggle international results + eloratings.net"
)

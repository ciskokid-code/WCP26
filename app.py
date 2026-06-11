"""
2026 World Cup Prediction Model
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

from src.config import ALL_TEAMS, GROUPS
from src.elo_model import get_elo_ratings, get_historical_h2h
from src.predictions import predict_match, run_group_simulation, run_simulation
from src.precompute import load_cache
from src.schedule import get_schedule, get_group_fixtures
from src.live_model import (
    load_saved_results,
    save_result,
    remove_result,
    apply_results_to_elo,
    build_known_group_results,
    get_group_standings,
    get_unplayed_fixtures,
)

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
    "Mexico": "🇲🇽", "South Africa": "🇿🇦", "South Korea": "🇰🇷", "Czechia": "🇨🇿",
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
        ["Title Odds", "Group Stage", "Match Predictor", "Head-to-Head", "Tournament Path"],
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

    # ── Match Predictor result entry (only shown on that page) ────────────
    if page == "Match Predictor":
        st.divider()

        # Initialise session state
        if "results_2026" not in st.session_state:
            st.session_state["results_2026"] = load_saved_results()

        _schedule = get_schedule()
        _current_results = st.session_state["results_2026"]
        _unplayed = get_unplayed_fixtures(_schedule, _current_results)

        with st.expander("Enter result", expanded=True):
            if not _unplayed:
                st.info("All fixtures have been recorded.")
            else:
                def _fmt_fixture(f: dict) -> str:
                    return (
                        f"Group {f['group']} MD{f['matchday']} · "
                        f"{f['home']} vs {f['away']}"
                    )

                _sel_fix = st.selectbox(
                    "Match",
                    _unplayed,
                    format_func=_fmt_fixture,
                    key="mp_fixture_sel",
                )

                _col_h, _col_a = st.columns(2)
                with _col_h:
                    _home_score = st.number_input(
                        _sel_fix["home"][:12],
                        min_value=0, max_value=20, value=0, step=1,
                        key="mp_home_score",
                    )
                with _col_a:
                    _away_score = st.number_input(
                        _sel_fix["away"][:12],
                        min_value=0, max_value=20, value=0, step=1,
                        key="mp_away_score",
                    )

                if st.button("Record result", type="primary", use_container_width=True):
                    st.session_state["results_2026"] = save_result(
                        home=_sel_fix["home"],
                        away=_sel_fix["away"],
                        home_score=int(_home_score),
                        away_score=int(_away_score),
                        group=_sel_fix["group"],
                        date=_sel_fix["date"],
                        results=st.session_state["results_2026"],
                    )
                    st.cache_data.clear()
                    st.rerun()

        # List recorded results with remove buttons
        _recorded = st.session_state["results_2026"]
        if _recorded:
            st.markdown(
                '<p style="font-size:12px;color:#888;margin:8px 0 4px;">Recorded results</p>',
                unsafe_allow_html=True,
            )
            for _r in sorted(_recorded, key=lambda x: x["date"]):
                _label = (
                    f"{_r['home']} {_r['home_score']}–{_r['away_score']} {_r['away']}"
                )
                _rcol1, _rcol2 = st.columns([5, 1])
                _rcol1.caption(_label)
                if _rcol2.button("❌", key=f"rm_{_r['home']}_{_r['away']}",
                                 help=f"Remove {_label}"):
                    st.session_state["results_2026"] = remove_result(
                        _r["home"], _r["away"], st.session_state["results_2026"]
                    )
                    st.cache_data.clear()
                    st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────────────

elos = _load_elos()
payload = _load_simulation(n_sims, seed)
sim = payload["simulation"]
group_probs_all = payload["group_probs"]


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 1: Title Odds
# ──────────────────────────────────────────────────────────────────────────────

if page == "Title Odds":

    team_group = {t: g for g, teams in GROUPS.items() for t in teams}
    champ  = sim["champion"]
    ranked = sorted(ALL_TEAMS, key=lambda t: champ.get(t, 0), reverse=True)

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <p style="color:#ff6b35;font-size:12px;font-weight:700;letter-spacing:2px;
                  text-transform:uppercase;margin-bottom:4px;">The Model's View</p>
        <h1 style="font-size:44px;font-weight:800;margin:0 0 8px 0;">
            Who wins the World Cup?
        </h1>
        <p style="color:#888;font-size:15px;margin:0 0 28px 0;">
            Every number is drawn from simulating the entire tournament
            <b>{payload['n_sims']:,}</b> times.
            Sense-check your gut before you lock in picks.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ── Top-3 hero cards ──────────────────────────────────────────────────────
    CARD_LABELS = ["FAVOURITE", "SECOND FAVOURITE", "THIRD FAVOURITE"]
    cols3 = st.columns(3)
    for col, team, label in zip(cols3, ranked[:3], CARD_LABELS):
        p = champ.get(team, 0)
        g = team_group.get(team, "?")
        col.markdown(
            f"""
            <div style="background:#1e1e2e;border-radius:12px;padding:22px 24px;
                        border:1px solid #2a2a3a;height:170px;">
                <div style="font-size:10px;font-weight:700;letter-spacing:2px;
                            color:#888;text-transform:uppercase;margin-bottom:10px;">
                    {label}
                </div>
                <div style="font-size:24px;font-weight:700;margin-bottom:2px;">
                    {flag(team)} {team}
                </div>
                <div style="font-size:46px;font-weight:800;color:#ff6b35;
                            line-height:1.1;margin-bottom:6px;">
                    {p*100:.1f}%
                </div>
                <div style="font-size:12px;color:#555;">
                    to lift the trophy · Group {g}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Circle grid — all 48 teams ────────────────────────────────────────────
    st.markdown(
        '<h3 style="margin-bottom:4px;">All contenders — chance to win</h3>',
        unsafe_allow_html=True,
    )

    items = ""
    for team in ranked:
        p = champ.get(team, 0)
        items += f"""
        <div style="text-align:center;padding:10px 4px;">
            <div style="font-size:24px;margin-bottom:6px;">{flag(team)}</div>
            <div style="width:80px;height:80px;border-radius:50%;
                        border:3px solid #ff6b35;
                        display:flex;align-items:center;justify-content:center;
                        margin:0 auto 8px;font-size:14px;font-weight:800;
                        color:#ff6b35;">
                {p*100:.1f}%
            </div>
            <div style="font-size:11px;font-weight:500;color:#ccc;
                        max-width:96px;margin:0 auto;
                        overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
                {team}
            </div>
        </div>"""

    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(104px,1fr));'
        f'gap:6px;margin-top:8px;">{items}</div>',
        unsafe_allow_html=True,
    )

    st.caption(f"Model: Elo + Poisson · {n_sims:,} simulations")


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
        df_g[display_cols].style.format(fmt),
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
# PAGE 3: Match Predictor
# ──────────────────────────────────────────────────────────────────────────────

elif page == "Match Predictor":

    # ── Session state init ────────────────────────────────────────────────────
    if "results_2026" not in st.session_state:
        st.session_state["results_2026"] = load_saved_results()

    _live_results = st.session_state["results_2026"]

    # ── Live simulation (cached on a hash of the results list) ───────────────
    _results_hash = hash(tuple(
        (r["home"], r["away"], r["home_score"], r["away_score"])
        for r in sorted(_live_results, key=lambda x: (x["date"], x["home"]))
    ))

    @st.cache_data(show_spinner="Running live simulation…", hash_funcs={int: lambda x: x})
    def _live_sim(_results_hash: int, _results: list, _base_elos: dict) -> dict:
        _known_gr = build_known_group_results(_results)
        _updated_elos = apply_results_to_elo(_base_elos, _results)
        return run_simulation(
            _updated_elos,
            n_sims=20_000,
            seed=42,
            known_group_results=_known_gr,
        )

    _live_sim_result = _live_sim(_results_hash, _live_results, elos)
    _known_group_results = build_known_group_results(_live_results)

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(
        '<p style="color:#ff6b35;font-size:12px;font-weight:700;letter-spacing:2px;'
        'text-transform:uppercase;margin-bottom:4px;">Live Updates</p>',
        unsafe_allow_html=True,
    )
    st.title("Match Predictor")
    st.caption("Probabilities update as results come in. Enter scores in the sidebar.")

    # ── Updated championship probability circles (top 10, show delta) ─────────
    _baseline_champ = sim["champion"]
    _live_champ = _live_sim_result["champion"]

    _top10 = sorted(ALL_TEAMS, key=lambda t: _live_champ.get(t, 0), reverse=True)[:10]

    _circle_items = ""
    for _team in _top10:
        _p_live = _live_champ.get(_team, 0)
        _p_base = _baseline_champ.get(_team, 0)
        _delta = (_p_live - _p_base) * 100
        _delta_str = f"△ {_delta:+.1f}pp"
        _delta_color = "#ff6b35" if _delta >= 0 else "#e74c3c"
        _circle_items += f"""
        <div style="text-align:center;padding:10px 4px;">
            <div style="font-size:22px;margin-bottom:4px;">{flag(_team)}</div>
            <div style="width:76px;height:76px;border-radius:50%;
                        border:3px solid #ff6b35;
                        display:flex;align-items:center;justify-content:center;
                        margin:0 auto 6px;font-size:13px;font-weight:800;
                        color:#ff6b35;">
                {_p_live*100:.1f}%
            </div>
            <div style="font-size:10px;font-weight:600;color:{_delta_color};
                        margin-bottom:3px;">{_delta_str}</div>
            <div style="font-size:10px;color:#888;
                        max-width:90px;margin:0 auto;
                        overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
                {_team}
            </div>
        </div>"""

    st.markdown(
        '<h3 style="margin-bottom:4px;">Championship odds — top 10</h3>'
        '<p style="color:#888;font-size:12px;margin-bottom:8px;">'
        'Triangle shows change vs. pre-tournament baseline.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(100px,1fr));'
        f'gap:6px;margin-bottom:24px;">{_circle_items}</div>',
        unsafe_allow_html=True,
    )

    # ── Group tabs ────────────────────────────────────────────────────────────
    _group_letters = list(GROUPS.keys())
    _group_tab_labels = [f"Group {g}" for g in _group_letters]
    _group_tabs = st.tabs(_group_tab_labels)

    for _tab, _grp in zip(_group_tabs, _group_letters):
        with _tab:
            _grp_teams = GROUPS[_grp]
            _grp_fixtures = get_group_fixtures(_grp)
            _grp_standings = get_group_standings(_grp, _live_results, _grp_teams)
            _updated_elos_for_grp = apply_results_to_elo(elos, _live_results)

            # ── Standings table ───────────────────────────────────────────
            if any(r["group"] == _grp for r in _live_results):
                _st_rows = []
                for _pos, _row in enumerate(_grp_standings, start=1):
                    _advance_marker = " ✓" if _pos <= 2 else ""
                    _st_rows.append({
                        "": f"{_pos}{_advance_marker}",
                        "Team": f"{flag(_row['team'])} {_row['team']}",
                        "P": _row["played"],
                        "W": _row["won"],
                        "D": _row["drawn"],
                        "L": _row["lost"],
                        "GF": _row["gf"],
                        "GA": _row["ga"],
                        "GD": _row["gd"],
                        "Pts": _row["pts"],
                    })
                st.dataframe(
                    pd.DataFrame(_st_rows),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("No results recorded yet for this group.")

            st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)

            # ── Match cards ───────────────────────────────────────────────
            _played_keys = {
                (r["home"], r["away"]): r
                for r in _live_results if r["group"] == _grp
            }

            for _fix in _grp_fixtures:
                _home = _fix["home"]
                _away = _fix["away"]
                _md = _fix["matchday"]
                _date = _fix["date"]
                _fix_key = (_home, _away)

                if _fix_key in _played_keys:
                    # Played — show result card
                    _res = _played_keys[_fix_key]
                    _hs = _res["home_score"]
                    _as = _res["away_score"]
                    st.markdown(
                        f"""
                        <div style="background:#1e1e2e;border:1px solid #2a2a3a;
                                    border-radius:10px;padding:14px 18px;margin-bottom:8px;
                                    display:flex;align-items:center;justify-content:space-between;">
                            <div style="font-size:13px;color:#ccc;min-width:120px;">
                                {flag(_home)} {_home}
                            </div>
                            <div style="text-align:center;flex:1;">
                                <div style="font-size:10px;color:#555;margin-bottom:4px;">
                                    MD{_md} · {_date}
                                </div>
                                <div style="font-size:22px;font-weight:800;color:#e8e8e8;">
                                    ✓ {_hs} – {_as}
                                </div>
                            </div>
                            <div style="font-size:13px;color:#ccc;min-width:120px;text-align:right;">
                                {_away} {flag(_away)}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    # Unplayed — show prediction card
                    _elo_h = _updated_elos_for_grp.get(_home, 1500)
                    _elo_a = _updated_elos_for_grp.get(_away, 1500)
                    _pred = predict_match(_elo_h, _elo_a)
                    _w = _pred["win"] * 100
                    _d = _pred["draw"] * 100
                    _l = _pred["loss"] * 100
                    st.markdown(
                        f"""
                        <div style="background:#1e1e2e;border:1px solid #2a2a3a;
                                    border-radius:10px;padding:14px 18px;margin-bottom:8px;
                                    display:flex;align-items:center;justify-content:space-between;">
                            <div style="font-size:13px;color:#ccc;min-width:120px;">
                                {flag(_home)} {_home}
                            </div>
                            <div style="text-align:center;flex:1;">
                                <div style="font-size:10px;color:#555;margin-bottom:4px;">
                                    MD{_md} · {_date}
                                </div>
                                <div style="font-size:13px;font-weight:700;color:#ff6b35;">
                                    {_w:.0f}% / {_d:.0f}% / {_l:.0f}%
                                </div>
                                <div style="font-size:9px;color:#555;margin-top:2px;">
                                    W / D / L
                                </div>
                            </div>
                            <div style="font-size:13px;color:#ccc;min-width:120px;text-align:right;">
                                {_away} {flag(_away)}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ──────────────────────────────────────────────────────────────────────────────
# PAGE 4: Head-to-Head
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
        "group_win", "r16", "quarter", "semi", "finalist", "champion"
    ]
    stage_nice = [
        "Win Group", "Reach R16", "Reach QF", "Reach SF", "Reach Final", "Champion"
    ]

    probs = [sim[s].get(selected, 0) for s in stages_ordered]

    # Funnel chart
    fig_funnel = go.Figure(
        go.Funnel(
            y=stage_nice,
            x=[p * 100 for p in probs],
            textinfo="value+percent initial",
            marker_color=[
                "#85c1e9", "#3498db", "#1a5276",
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
                for s in ["group_win", "r16", "champion"]
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

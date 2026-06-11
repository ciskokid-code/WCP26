"""
2026 World Cup Prediction Model
Run: streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import poisson as _poisson

from src.config import ALL_TEAMS, GROUPS, TEAM_ALIASES
from src.elo_model import get_elo_ratings, get_historical_h2h
from src.live_model import (
    apply_results_to_elo,
    build_known_group_results,
    get_group_standings,
    get_unplayed_fixtures,
    load_saved_results,
    remove_result,
    save_result,
)
from src.predictions import predict_match, run_group_simulation, run_simulation
from src.precompute import load_cache
from src.schedule import get_group_fixtures, get_schedule

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="WC 2026 Model",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Mobile-first global CSS ───────────────────────────────────────────────────

st.markdown("""
<style>
/* hide the hamburger / sidebar toggle on mobile */
[data-testid="collapsedControl"] { display: none; }

/* make the main block fill the viewport nicely */
.block-container { padding: 1rem 1rem 4rem; max-width: 860px; }

/* tab bar: allow horizontal scroll on narrow screens */
[data-testid="stTabs"] > div:first-child {
    overflow-x: auto;
    white-space: nowrap;
    -webkit-overflow-scrolling: touch;
    scrollbar-width: none;
}
[data-testid="stTabs"] > div:first-child::-webkit-scrollbar { display: none; }

/* tighten metric cards on mobile */
[data-testid="metric-container"] { padding: 0.4rem 0.6rem; }

/* make plotly charts not overflow on narrow screens */
.js-plotly-plot { max-width: 100%; }
</style>
""", unsafe_allow_html=True)

# ── Shared constants & helpers ────────────────────────────────────────────────

STAGE_LABELS = {
    "champion": "Win Tournament",
    "finalist": "Reach Final",
    "semi":     "Reach Semis",
    "quarter":  "Reach Quarters",
    "r16":      "Reach R16",
    "group_win":"Win Group",
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
    return f"{p * 100:.1f}%"


def _hbar_list(teams_probs: list[tuple[str, float]], max_p: float, start_rank: int = 1) -> str:
    """Render a mobile-friendly horizontal bar list."""
    html = ""
    for i, (team, p) in enumerate(teams_probs, start_rank):
        bar_w = (p / max_p) * 100 if max_p > 0 else 0
        html += f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;
                    padding:10px 12px;background:#1e1e2e;border-radius:8px;
                    border:1px solid #2a2a3a;">
            <div style="width:20px;text-align:right;color:#555;font-size:12px;
                        flex-shrink:0;">{i}</div>
            <div style="font-size:20px;flex-shrink:0;">{flag(team)}</div>
            <div style="flex:1;font-size:13px;font-weight:500;min-width:0;
                        overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{team}</div>
            <div style="width:90px;background:#2a2a2a;border-radius:4px;
                        height:8px;flex-shrink:0;overflow:hidden;">
                <div style="background:#ff6b35;width:{bar_w:.1f}%;height:8px;
                            border-radius:4px;"></div>
            </div>
            <div style="width:44px;text-align:right;font-size:13px;font-weight:700;
                        color:#ff6b35;flex-shrink:0;">{p * 100:.1f}%</div>
        </div>"""
    return html


# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Elo ratings…")
def _load_elos() -> dict[str, float]:
    return get_elo_ratings(verbose=False)


@st.cache_resource(show_spinner="Running simulation… (~10 s first load)")
def _load_simulation(n_sims: int, seed: int) -> dict:
    cached = load_cache()
    if cached and cached.get("n_sims", 0) >= n_sims:
        return cached
    elos = _load_elos()
    sim = run_simulation(elos, n_sims=n_sims, seed=seed)
    group_probs = {g: run_group_simulation(g, elos, n_sims=n_sims, seed=seed) for g in GROUPS}
    return {"simulation": sim, "group_probs": group_probs, "elos": elos, "n_sims": n_sims}


@st.cache_data(show_spinner=False)
def _load_h2h(team_a: str, team_b: str) -> pd.DataFrame:
    return get_historical_h2h(team_a, team_b)


# ── Settings (collapsed expander) ────────────────────────────────────────────

with st.expander("⚙️ Model settings", expanded=False):
    _c1, _c2, _c3 = st.columns([3, 2, 1])
    with _c1:
        n_sims = st.select_slider(
            "Simulations",
            options=[5_000, 10_000, 25_000, 50_000, 100_000],
            value=50_000,
        )
    with _c2:
        seed = st.number_input("RNG seed", value=42, min_value=0, step=1)
    with _c3:
        st.write("")
        if st.button("↺ Refresh", use_container_width=True):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()

# ── Load baseline data ────────────────────────────────────────────────────────

elos   = _load_elos()
payload = _load_simulation(n_sims, seed)
sim    = payload["simulation"]
group_probs_all = payload["group_probs"]

# Session state for live 2026 results
if "results_2026" not in st.session_state:
    st.session_state["results_2026"] = load_saved_results()

# ── Navigation tabs ───────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Title Odds",
    "⚽ Groups",
    "📊 Match Predictor",
    "⚔️ H2H",
    "🗺️ Path",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Title Odds
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    team_group = {t: g for g, teams in GROUPS.items() for t in teams}
    champ  = sim["champion"]
    ranked = sorted(ALL_TEAMS, key=lambda t: champ.get(t, 0), reverse=True)

    st.markdown(
        f"""
        <p style="color:#ff6b35;font-size:12px;font-weight:700;letter-spacing:2px;
                  text-transform:uppercase;margin-bottom:4px;">The Model's View</p>
        <h1 style="font-size:36px;font-weight:800;margin:0 0 6px 0;">
            Who wins the World Cup?
        </h1>
        <p style="color:#888;font-size:14px;margin:0 0 24px 0;">
            Every number is drawn from simulating the entire tournament
            <b>{payload['n_sims']:,}</b> times.
            Sense-check your gut before you lock in picks.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Hero cards — responsive grid (3 cols on desktop, 1 on mobile)
    hero_html = '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:28px;">'
    for team, label in zip(ranked[:3], ["FAVOURITE", "SECOND FAVOURITE", "THIRD FAVOURITE"]):
        p = champ.get(team, 0)
        g = team_group.get(team, "?")
        hero_html += f"""
        <div style="background:#1e1e2e;border-radius:12px;padding:18px 20px;
                    border:1px solid #2a2a3a;">
            <div style="font-size:10px;font-weight:700;letter-spacing:2px;color:#888;
                        text-transform:uppercase;margin-bottom:8px;">{label}</div>
            <div style="font-size:20px;font-weight:700;margin-bottom:2px;">{flag(team)} {team}</div>
            <div style="font-size:40px;font-weight:800;color:#ff6b35;line-height:1.1;
                        margin-bottom:4px;">{p * 100:.1f}%</div>
            <div style="font-size:11px;color:#555;">to lift the trophy · Group {g}</div>
        </div>"""
    hero_html += "</div>"
    st.markdown(hero_html, unsafe_allow_html=True)

    # Horizontal bar list — all remaining teams
    st.markdown("### All contenders — chance to win")
    max_p = champ.get(ranked[0], 1)
    pairs = [(t, champ.get(t, 0)) for t in ranked]
    st.markdown(_hbar_list(pairs, max_p, start_rank=1), unsafe_allow_html=True)

    st.caption(f"Model: Elo + Poisson · {n_sims:,} simulations")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Group Stage
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Group Stage Probabilities")
    st.caption("Probability of each finishing position within the group.")

    group_sel = st.selectbox("Select group", list(GROUPS.keys()),
                             format_func=lambda g: f"Group {g}")

    teams  = GROUPS[group_sel]
    gp     = group_probs_all[group_sel]
    labels_pos = ["1st", "2nd", "3rd", "4th"]

    rows_g = []
    for team in teams:
        rows_g.append({
            "Team": f"{flag(team)} {team}",
            "Elo":  f"{elos.get(team, 1500):.0f}",
            "1st":  gp[team][1],
            "2nd":  gp[team][2],
            "3rd":  gp[team][3],
            "4th":  gp[team][4],
            "_elo": elos.get(team, 1500),
        })
    df_g = pd.DataFrame(rows_g).sort_values("_elo", ascending=False).reset_index(drop=True)

    fig_g = go.Figure()
    colors_pos = ["#2ca02c", "#98df8a", "#ffbb78", "#d62728"]
    for col, color in zip(labels_pos, colors_pos):
        fig_g.add_trace(go.Bar(
            name=col, x=df_g["Team"], y=df_g[col] * 100,
            marker_color=color,
            text=[fmt_pct(v) for v in df_g[col]],
            textposition="inside", insidetextanchor="middle",
        ))
    fig_g.update_layout(
        barmode="stack",
        yaxis=dict(title="Probability (%)", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=320, margin=dict(t=20, b=60),
    )
    st.plotly_chart(fig_g, use_container_width=True)

    fmt = {c: "{:.1%}" for c in labels_pos}
    st.dataframe(
        df_g[["Team", "Elo", "1st", "2nd", "3rd", "4th"]].style.format(fmt),
        use_container_width=True, hide_index=True,
    )

    st.divider()
    st.subheader("Qualification bubble chart")
    qual_rows = []
    for g, g_teams in GROUPS.items():
        for team in g_teams:
            gpp = group_probs_all[g][team]
            qual_p = gpp[1] + gpp[2] + (8 / 12) * gpp[3]
            qual_rows.append({
                "Group": f"Group {g}", "Team": f"{flag(team)} {team}",
                "Win Group": gpp[1], "Qualify (est.)": min(qual_p, 1.0),
                "Elo": elos.get(team, 1500),
            })
    df_qual = pd.DataFrame(qual_rows).sort_values(["Group", "Elo"], ascending=[True, False])
    fig_heat = px.scatter(
        df_qual, x="Qualify (est.)", y="Win Group",
        color="Group", text="Team", size="Elo", size_max=20,
        labels={"Qualify (est.)": "Est. qualification prob.", "Win Group": "Group-win prob."},
    )
    fig_heat.update_traces(textposition="top center", textfont_size=8)
    fig_heat.update_layout(height=500, margin=dict(t=30, b=20))
    st.plotly_chart(fig_heat, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Match Predictor
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    _live_results = st.session_state["results_2026"]
    _schedule     = get_schedule()
    _unplayed     = get_unplayed_fixtures(_schedule, _live_results)

    st.markdown(
        '<p style="color:#ff6b35;font-size:12px;font-weight:700;letter-spacing:2px;'
        'text-transform:uppercase;margin-bottom:2px;">Live Updates</p>',
        unsafe_allow_html=True,
    )
    st.title("Match Predictor")
    st.caption("Enter scores below — championship odds update automatically.")

    # ── Result entry form (inline, mobile-friendly) ───────────────────────────
    with st.expander("➕ Enter a result", expanded=bool(_unplayed)):
        if not _unplayed:
            st.info("All group-stage fixtures have been recorded.")
        else:
            def _fmt_fix(f: dict) -> str:
                return f"Group {f['group']} MD{f['matchday']} · {f['home']} vs {f['away']}"

            _sel = st.selectbox("Match", _unplayed, format_func=_fmt_fix, key="mp_fix_sel")
            _c_h, _c_a = st.columns(2)
            _home_score = _c_h.number_input(
                f"{flag(_sel['home'])} {_sel['home'][:14]}",
                min_value=0, max_value=20, value=0, step=1, key="mp_hs",
            )
            _away_score = _c_a.number_input(
                f"{flag(_sel['away'])} {_sel['away'][:14]}",
                min_value=0, max_value=20, value=0, step=1, key="mp_as",
            )
            if st.button("Record result ✓", type="primary", use_container_width=True):
                st.session_state["results_2026"] = save_result(
                    home=_sel["home"], away=_sel["away"],
                    home_score=int(_home_score), away_score=int(_away_score),
                    group=_sel["group"], date=_sel["date"],
                    results=st.session_state["results_2026"],
                )
                st.cache_data.clear()
                st.rerun()

    # Recorded results list with ❌ remove
    if _live_results:
        st.markdown(
            '<p style="font-size:12px;color:#888;margin:12px 0 4px;">Recorded results</p>',
            unsafe_allow_html=True,
        )
        for _r in sorted(_live_results, key=lambda x: x["date"]):
            _lbl = f"{flag(_r['home'])} {_r['home']} {_r['home_score']}–{_r['away_score']} {_r['away']} {flag(_r['away'])}"
            _rc1, _rc2 = st.columns([6, 1])
            _rc1.markdown(f'<span style="font-size:13px;">{_lbl}</span>', unsafe_allow_html=True)
            if _rc2.button("❌", key=f"rm_{_r['home']}_{_r['away']}"):
                st.session_state["results_2026"] = remove_result(
                    _r["home"], _r["away"], st.session_state["results_2026"]
                )
                st.cache_data.clear()
                st.rerun()

    st.divider()

    # ── Live simulation ────────────────────────────────────────────────────────
    _results_hash = hash(tuple(
        (r["home"], r["away"], r["home_score"], r["away_score"])
        for r in sorted(_live_results, key=lambda x: (x["date"], x["home"]))
    ))

    @st.cache_data(show_spinner="Updating odds…")
    def _live_sim(_h: int, _res: list, _base_elos: dict) -> dict:
        _known = build_known_group_results(_res)
        _elos  = apply_results_to_elo(_base_elos, _res)
        return run_simulation(_elos, n_sims=20_000, seed=42, known_group_results=_known)

    _live_sim_result = _live_sim(_results_hash, _live_results, elos)
    _updated_elos    = apply_results_to_elo(elos, _live_results)

    # ── Updated championship odds — horizontal bars with delta ────────────────
    _base_champ = sim["champion"]
    _live_champ = _live_sim_result["champion"]
    _ranked_live = sorted(ALL_TEAMS, key=lambda t: _live_champ.get(t, 0), reverse=True)
    _max_live    = _live_champ.get(_ranked_live[0], 1)

    st.markdown("### Championship odds — updated")

    _bars_html = ""
    for i, team in enumerate(_ranked_live[:20], 1):
        p     = _live_champ.get(team, 0)
        p_base = _base_champ.get(team, 0)
        delta  = (p - p_base) * 100
        bar_w  = (p / _max_live) * 100
        d_col  = "#2ca02c" if delta > 0.05 else "#e74c3c" if delta < -0.05 else "#888"
        d_str  = f"{delta:+.1f}pp" if abs(delta) > 0.05 else "—"
        _bars_html += f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;
                    padding:10px 12px;background:#1e1e2e;border-radius:8px;
                    border:1px solid #2a2a3a;">
            <div style="width:20px;text-align:right;color:#555;font-size:12px;
                        flex-shrink:0;">{i}</div>
            <div style="font-size:20px;flex-shrink:0;">{flag(team)}</div>
            <div style="flex:1;font-size:13px;font-weight:500;min-width:0;
                        overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{team}</div>
            <div style="width:70px;background:#2a2a2a;border-radius:4px;
                        height:8px;flex-shrink:0;overflow:hidden;">
                <div style="background:#ff6b35;width:{bar_w:.1f}%;height:8px;
                            border-radius:4px;"></div>
            </div>
            <div style="width:40px;text-align:right;font-size:13px;font-weight:700;
                        color:#ff6b35;flex-shrink:0;">{p * 100:.1f}%</div>
            <div style="width:52px;text-align:right;font-size:11px;font-weight:600;
                        color:{d_col};flex-shrink:0;">{d_str}</div>
        </div>"""
    st.markdown(_bars_html, unsafe_allow_html=True)
    st.caption("pp = percentage points vs pre-tournament baseline")

    st.divider()

    # ── Group tabs ─────────────────────────────────────────────────────────────
    st.markdown("### Group stage matches")
    _grp_tabs = st.tabs([f"Group {g}" for g in GROUPS])

    for _tab, _grp in zip(_grp_tabs, GROUPS):
        with _tab:
            _grp_teams    = GROUPS[_grp]
            _grp_fixtures = get_group_fixtures(_grp)
            _grp_standings = get_group_standings(_grp, _live_results, _grp_teams)
            _played_keys  = {
                (r["home"], r["away"]): r
                for r in _live_results if r["group"] == _grp
            }

            # Standings table
            if any(r["group"] == _grp for r in _live_results):
                _st_rows = []
                for _pos, _row in enumerate(_grp_standings, 1):
                    _st_rows.append({
                        "#": f"{_pos}{'✓' if _pos <= 2 else ''}",
                        "Team": f"{flag(_row['team'])} {_row['team']}",
                        "P": _row["played"], "W": _row["won"],
                        "D": _row["drawn"],  "L": _row["lost"],
                        "GD": _row["gd"],    "Pts": _row["pts"],
                    })
                st.dataframe(pd.DataFrame(_st_rows), use_container_width=True, hide_index=True)
            else:
                st.caption("No results yet.")

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            # Match cards
            for _fix in _grp_fixtures:
                _home, _away = _fix["home"], _fix["away"]
                _md, _date   = _fix["matchday"], _fix["date"]

                if (_home, _away) in _played_keys:
                    _res = _played_keys[(_home, _away)]
                    _hs, _as = _res["home_score"], _res["away_score"]
                    st.markdown(f"""
                    <div style="background:#1a2a1a;border:1px solid #2a3a2a;
                                border-radius:10px;padding:12px 16px;margin-bottom:6px;">
                        <div style="font-size:10px;color:#555;text-align:center;
                                    margin-bottom:4px;">MD{_md} · {_date} ✓</div>
                        <div style="display:flex;align-items:center;justify-content:space-between;">
                            <div style="font-size:13px;color:#ccc;flex:1;">{flag(_home)} {_home}</div>
                            <div style="font-size:22px;font-weight:800;color:#e8e8e8;
                                        padding:0 12px;flex-shrink:0;">{_hs}–{_as}</div>
                            <div style="font-size:13px;color:#ccc;flex:1;text-align:right;">{_away} {flag(_away)}</div>
                        </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    _pred = predict_match(_updated_elos.get(_home, 1500), _updated_elos.get(_away, 1500))
                    _w = _pred["win"] * 100
                    _d = _pred["draw"] * 100
                    _l = _pred["loss"] * 100
                    st.markdown(f"""
                    <div style="background:#1e1e2e;border:1px solid #2a2a3a;
                                border-radius:10px;padding:12px 16px;margin-bottom:6px;">
                        <div style="font-size:10px;color:#555;text-align:center;
                                    margin-bottom:4px;">MD{_md} · {_date}</div>
                        <div style="display:flex;align-items:center;justify-content:space-between;">
                            <div style="font-size:13px;color:#ccc;flex:1;">{flag(_home)} {_home}</div>
                            <div style="text-align:center;flex-shrink:0;padding:0 8px;">
                                <div style="font-size:13px;font-weight:700;color:#ff6b35;">
                                    {_w:.0f}% / {_d:.0f}% / {_l:.0f}%
                                </div>
                                <div style="font-size:9px;color:#555;">Win / Draw / Win</div>
                            </div>
                            <div style="font-size:13px;color:#ccc;flex:1;text-align:right;">{_away} {flag(_away)}</div>
                        </div>
                    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Head-to-Head
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("Head-to-Head")

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox(
            "Team A", sorted(ALL_TEAMS),
            index=sorted(ALL_TEAMS).index("Brazil"),
            format_func=lambda t: f"{flag(t)} {t}",
            key="h2h_a",
        )
    with col2:
        team_b = st.selectbox(
            "Team B", sorted(ALL_TEAMS),
            index=sorted(ALL_TEAMS).index("Argentina"),
            format_func=lambda t: f"{flag(t)} {t}",
            key="h2h_b",
        )

    if team_a == team_b:
        st.warning("Select two different teams.")
    else:
        elo_a = elos.get(team_a, 1500)
        elo_b = elos.get(team_b, 1500)
        result = predict_match(elo_a, elo_b)
        win_a, draw, win_b = result["win"], result["draw"], result["loss"]

        st.subheader(f"{flag(team_a)} {team_a}  vs  {flag(team_b)} {team_b}")

        c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 2])
        c1.metric(f"{flag(team_a)} Win", fmt_pct(win_a))
        c2.metric("Draw", fmt_pct(draw))
        c3.metric(f"{flag(team_b)} Win", fmt_pct(win_b))
        c4.metric(f"Elo", f"{elo_a:.0f}")
        c5.metric(f"Elo", f"{elo_b:.0f}")

        fig_bar = go.Figure(go.Bar(
            x=[win_a * 100, draw * 100, win_b * 100],
            y=[f"{flag(team_a)} {team_a}", "Draw", f"{flag(team_b)} {team_b}"],
            orientation="h",
            marker_color=["#1f77b4", "#7f7f7f", "#d62728"],
            text=[fmt_pct(win_a), fmt_pct(draw), fmt_pct(win_b)],
            textposition="auto",
        ))
        fig_bar.update_layout(
            xaxis=dict(range=[0, 105]), height=200,
            margin=dict(t=10, b=30), showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Score heatmap
        st.subheader("Score probability heatmap")
        mu_a, mu_b = result["mu_a"], result["mu_b"]
        max_g = 6
        score_matrix = np.array([
            [_poisson.pmf(i, mu_a) * _poisson.pmf(j, mu_b)
             for j in range(max_g + 1)]
            for i in range(max_g + 1)
        ]) * 100

        df_heat = pd.DataFrame(
            score_matrix,
            index=[f"{team_a[:3]} {i}" for i in range(max_g + 1)],
            columns=[f"{team_b[:3]} {j}" for j in range(max_g + 1)],
        )
        mls, mlg = result["most_likely_score"]
        fig_heat = px.imshow(
            df_heat, color_continuous_scale="Blues", labels={"color": "%"},
            title=f"Most likely: {team_a} {mls}–{mlg} {team_b}", aspect="auto",
        )
        fig_heat.update_layout(height=340, margin=dict(t=50, b=20))
        st.plotly_chart(fig_heat, use_container_width=True)

        # Historical record
        st.subheader("Historical record")
        h2h = _load_h2h(team_a, team_b)
        if h2h.empty:
            st.info("No historical matches found between these teams.")
        else:
            name_a = TEAM_ALIASES.get(team_a, team_a)
            name_b = TEAM_ALIASES.get(team_b, team_b)
            wins_a = (
                ((h2h["home_team"] == name_a) & (h2h["home_score"] > h2h["away_score"])).sum() +
                ((h2h["away_team"] == name_a) & (h2h["away_score"] > h2h["home_score"])).sum()
            )
            wins_b = (
                ((h2h["home_team"] == name_b) & (h2h["home_score"] > h2h["away_score"])).sum() +
                ((h2h["away_team"] == name_b) & (h2h["away_score"] > h2h["home_score"])).sum()
            )
            draws = len(h2h) - wins_a - wins_b
            ca, cb, cc, cd = st.columns(4)
            ca.metric("Matches", len(h2h))
            cb.metric(f"{flag(team_a)} Wins", wins_a)
            cc.metric("Draws", draws)
            cd.metric(f"{flag(team_b)} Wins", wins_b)
            st.dataframe(
                h2h[["date", "home_team", "home_score", "away_score", "away_team", "tournament"]]
                .head(25)
                .rename(columns={"home_team": "Home", "away_team": "Away",
                                 "home_score": "HG", "away_score": "AG",
                                 "tournament": "Competition"}),
                use_container_width=True, hide_index=True,
            )
            if len(h2h) > 25:
                st.caption(f"Showing 25 of {len(h2h)} matches.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Tournament Path
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.header("Tournament Path")
    st.caption("Probability of reaching each knockout round.")

    selected = st.selectbox(
        "Select team", sorted(ALL_TEAMS),
        format_func=lambda t: f"{flag(t)} {t}", key="path_team",
    )

    stages_ordered = ["group_win", "r16", "quarter", "semi", "finalist", "champion"]
    stage_nice     = ["Win Group", "Reach R16", "Reach QF", "Reach SF", "Reach Final", "Champion"]
    probs          = [sim[s].get(selected, 0) for s in stages_ordered]

    fig_funnel = go.Figure(go.Funnel(
        y=stage_nice, x=[p * 100 for p in probs],
        textinfo="value+percent initial",
        marker_color=["#85c1e9", "#3498db", "#1a5276", "#e74c3c", "#c0392b", "#922b21"],
    ))
    fig_funnel.update_layout(
        title=f"{flag(selected)} {selected}",
        height=400, margin=dict(t=50, b=10),
    )
    st.plotly_chart(fig_funnel, use_container_width=True)

    group_of = next((g for g, teams in GROUPS.items() if selected in teams), None)
    if group_of:
        st.subheader(f"Group {group_of} rivals")
        rivals = GROUPS[group_of]
        cmp_data = {
            "Team": [f"{flag(t)} {t}" for t in rivals],
            "Elo":  [f"{elos.get(t, 1500):.0f}" for t in rivals],
            **{STAGE_LABELS[s]: [fmt_pct(sim[s].get(t, 0)) for t in rivals]
               for s in ["group_win", "r16", "champion"]},
        }
        st.dataframe(pd.DataFrame(cmp_data), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Compare two teams")
    cmp_col1, cmp_col2 = st.columns(2)
    cmp_a = cmp_col1.selectbox("Team A", sorted(ALL_TEAMS),
                                format_func=lambda t: f"{flag(t)} {t}", key="cmp_a")
    cmp_b = cmp_col2.selectbox("Team B", sorted(ALL_TEAMS),
                                format_func=lambda t: f"{flag(t)} {t}", key="cmp_b", index=1)

    fig_cmp = go.Figure()
    for team, color in [(cmp_a, "#ff6b35"), (cmp_b, "#3498db")]:
        fig_cmp.add_trace(go.Scatterpolar(
            r=[sim[s].get(team, 0) * 100 for s in stages_ordered],
            theta=stage_nice, fill="toself",
            name=f"{flag(team)} {team}", line_color=color, opacity=0.7,
        ))
    fig_cmp.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=400, margin=dict(t=30, b=10),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    cmp_rows = [
        {"Stage": lbl,
         f"{flag(cmp_a)} {cmp_a}": fmt_pct(sim[s].get(cmp_a, 0)),
         f"{flag(cmp_b)} {cmp_b}": fmt_pct(sim[s].get(cmp_b, 0))}
        for s, lbl in zip(stages_ordered, stage_nice)
    ]
    st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    f"Model: Elo + Poisson · {payload['n_sims']:,} simulations · "
    "Data: Kaggle results + eloratings.net"
)

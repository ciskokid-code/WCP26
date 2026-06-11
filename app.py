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

# ── Live simulation — reacts to entered results ───────────────────────────────

_live_results_global = st.session_state["results_2026"]
_live_hash = hash(tuple(
    (r["home"], r["away"], r["home_score"], r["away_score"])
    for r in sorted(_live_results_global, key=lambda x: (x["date"], x["home"]))
))


@st.cache_data(show_spinner="Updating simulation…")
def _compute_live_sim(_h: int, _res_tuple: tuple, _base_elos: dict) -> dict:
    """Re-run simulation only when actual results exist; otherwise reuse baseline."""
    if not _res_tuple:
        return payload
    _res   = list(_res_tuple)
    _known = build_known_group_results(_res)
    _elos  = apply_results_to_elo(_base_elos, _res)
    _sim   = run_simulation(_elos, n_sims=20_000, seed=42, known_group_results=_known)
    return {"simulation": _sim, "group_probs": payload["group_probs"],
            "elos": _elos, "n_sims": 20_000}


# Convert results list to a hashable tuple for the cache key
_live_res_tuple = tuple(
    (r["home"], r["away"], r["home_score"], r["away_score"])
    for r in sorted(_live_results_global, key=lambda x: (x["date"], x["home"]))
)
_live_payload = _compute_live_sim(_live_hash, _live_res_tuple, elos)
live_sim  = _live_payload["simulation"]
live_elos = apply_results_to_elo(elos, _live_results_global)


def _expected_group_pos(group: str, pos: int) -> tuple[str, float]:
    """Most likely team for position 1 (winner) or 2 (runner-up) in a group."""
    teams = GROUPS[group]
    standings = get_group_standings(group, _live_results_global, teams)
    fully_played = all(row["played"] == 3 for row in standings)
    if fully_played:
        return standings[pos - 1]["team"], 1.0
    if pos == 1:
        probs = [(t, live_sim["group_win"].get(t, 0)) for t in teams]
    else:
        probs = [(t, max(0.0, live_sim["qualified"].get(t, 0)
                        - live_sim["group_win"].get(t, 0))) for t in teams]
    probs.sort(key=lambda x: x[1], reverse=True)
    return probs[0][0], probs[0][1]


# ── Navigation tabs ───────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Title Odds",
    "⚽ Groups",
    "📊 Match Predictor",
    "⚔️ H2H",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Title Odds
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    team_group = {t: g for g, teams in GROUPS.items() for t in teams}
    champ  = live_sim["champion"]   # updates automatically as results are entered
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
    st.markdown(
        '<h2 style="margin-bottom:4px;">Groups</h2>'
        '<p style="color:#888;font-size:13px;margin-bottom:20px;">'
        'Bar shows chance to win the group. Percentages are 1st / 2nd finish probability.</p>',
        unsafe_allow_html=True,
    )

    for _g, _g_teams in GROUPS.items():
        _gp = group_probs_all[_g]
        # Sort teams by group-win probability descending
        _sorted = sorted(_g_teams, key=lambda t: _gp[t][1], reverse=True)
        _max_p1 = _gp[_sorted[0]][1] or 0.01

        _html = (
            f'<div style="margin-bottom:24px;">'
            f'<div style="font-size:13px;font-weight:700;color:#ff6b35;letter-spacing:1px;'
            f'text-transform:uppercase;margin-bottom:8px;">Group {_g}</div>'
        )
        for _t in _sorted:
            _p1  = _gp[_t][1]
            _p2  = _gp[_t][2]
            _bar = (_p1 / _max_p1) * 100
            _html += f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;
                        padding:9px 12px;background:#1e1e2e;border-radius:8px;
                        border:1px solid #2a2a3a;">
                <div style="font-size:19px;flex-shrink:0;">{flag(_t)}</div>
                <div style="flex:1;font-size:13px;font-weight:500;min-width:0;
                            overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{_t}</div>
                <div style="width:80px;background:#2a2a2a;border-radius:4px;
                            height:7px;flex-shrink:0;overflow:hidden;">
                    <div style="background:#ff6b35;width:{_bar:.1f}%;height:7px;border-radius:4px;"></div>
                </div>
                <div style="width:96px;text-align:right;font-size:12px;color:#aaa;
                            flex-shrink:0;white-space:nowrap;">
                    1st <b style="color:#ff6b35;">{_p1*100:.0f}%</b>
                    &nbsp;·&nbsp;
                    2nd <b style="color:#888;">{_p2*100:.0f}%</b>
                </div>
            </div>"""
        _html += "</div>"
        st.markdown(_html, unsafe_allow_html=True)

    st.caption(f"Model: Elo + Poisson · {n_sims:,} simulations")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Match Predictor
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    _live_results = _live_results_global
    _schedule     = get_schedule()
    _unplayed     = get_unplayed_fixtures(_schedule, _live_results)

    st.markdown(
        '<p style="color:#ff6b35;font-size:12px;font-weight:700;letter-spacing:2px;'
        'text-transform:uppercase;margin-bottom:2px;">Live Updates</p>',
        unsafe_allow_html=True,
    )
    st.title("Match Predictor")
    st.caption("Predictions for every match across all stages of the tournament.")

    # ── Group stage matches ────────────────────────────────────────────────────
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
                    _pred = predict_match(live_elos.get(_home, 1500), live_elos.get(_away, 1500))
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

    # ── Knockout bracket ───────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Knockout bracket")
    st.caption(
        "Each bracket section shows one team's projected path R32 → R16 → QF. "
        "Teams only meet outside their section in the SF or Final."
    )

    _grp_order = list(GROUPS.keys())

    # ── helpers ────────────────────────────────────────────────────────────────

    def _proj_winner(ta: str, tb: str) -> str:
        pred = predict_match(live_elos.get(ta, 1500), live_elos.get(tb, 1500))
        return ta if pred["win"] >= pred["loss"] else tb

    def _advance(pairs):
        return [(_proj_winner(*pairs[i]), _proj_winner(*pairs[i + 1]))
                for i in range(0, len(pairs), 2)]

    def _win_pct(ta: str, tb: str) -> str:
        pred = predict_match(live_elos.get(ta, 1500), live_elos.get(tb, 1500))
        w = pred["win"] * 100
        d = pred["draw"] * 100
        l = pred["loss"] * 100
        return f"{w:.0f}% / {d:.0f}% / {l:.0f}%"

    def _row(ta: str, tb: str, rnd: str, winner: str) -> str:
        """Single match row: Team A vs Team B with round label and W/D/L."""
        pred = predict_match(live_elos.get(ta, 1500), live_elos.get(tb, 1500))
        w, d, l = pred["win"]*100, pred["draw"]*100, pred["loss"]*100
        # Highlight projected winner in orange
        col_a = "#ff6b35" if winner == ta else "#ccc"
        col_b = "#ff6b35" if winner == tb else "#ccc"
        return f"""
        <div style="display:flex;align-items:center;gap:6px;padding:9px 12px;
                    background:#1e1e2e;border-radius:8px;margin-bottom:5px;
                    border:1px solid #2a2a3a;">
            <div style="font-size:9px;color:#555;width:28px;flex-shrink:0;text-align:center;
                        line-height:1.2;">{rnd}</div>
            <div style="flex:1;font-size:13px;font-weight:600;color:{col_a};min-width:0;
                        overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
                {flag(ta)} {ta}
            </div>
            <div style="font-size:11px;color:#ff6b35;font-weight:700;
                        flex-shrink:0;text-align:center;min-width:80px;">
                {w:.0f}% / {d:.0f}% / {l:.0f}%
            </div>
            <div style="flex:1;font-size:13px;font-weight:600;color:{col_b};min-width:0;
                        overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
                        text-align:right;">
                {tb} {flag(tb)}
            </div>
        </div>"""

    # ── Build R32 bracket (16 ordered pairs) ───────────────────────────────────
    # Best-3rd candidates: teams most likely to qualify without winning their group
    _p2_score = {
        t: max(0.0, live_sim["qualified"].get(t, 0) - live_sim["group_win"].get(t, 0))
        for t in ALL_TEAMS
    }
    # Only teams unlikely to win their group qualify as 3rd
    _thirds_sorted = sorted(
        [t for t in ALL_TEAMS if live_sim["group_win"].get(t, 1) < 0.50],
        key=lambda t: _p2_score[t], reverse=True
    )
    # Pick top 8, one per group if possible
    _best_thirds: list[str] = []
    _seen_grp: set[str] = set()
    for _t in _thirds_sorted:
        _tg = next(g for g, ts in GROUPS.items() if _t in ts)
        if _tg not in _seen_grp:
            _seen_grp.add(_tg)
            _best_thirds.append(_t)
        if len(_best_thirds) == 8:
            break
    # Pad with strongest remaining if needed
    for _t in _thirds_sorted:
        if _t not in _best_thirds and len(_best_thirds) < 8:
            _best_thirds.append(_t)

    _r32: list[tuple[str, str]] = []
    for i in range(8):                         # M1-M8: Grp A-H winners vs best-3rds
        _w, _ = _expected_group_pos(_grp_order[i], 1)
        _t3   = _best_thirds[7 - i] if i < len(_best_thirds) else "TBD"
        _r32.append((_w, _t3))
    for i in range(4):                         # M9-M12: Grp I-L winners vs Grp A-D RU
        _w, _ = _expected_group_pos(_grp_order[8 + i], 1)
        _r, _ = _expected_group_pos(_grp_order[i], 2)
        _r32.append((_w, _r))
    for i in range(4):                         # M13-M16: Grp I-L RU vs Grp E-H RU
        _r1, _ = _expected_group_pos(_grp_order[8 + i], 2)
        _r2, _ = _expected_group_pos(_grp_order[4 + i], 2)
        _r32.append((_r1, _r2))

    # Propagate full bracket
    _r16 = _advance(_r32)
    _qf  = _advance(_r16)
    _sf  = _advance(_qf)
    _fin = _advance(_sf)[0]

    # ── 4 bracket sections: each covers 4 R32 → 2 R16 → 1 QF ─────────────────
    # Section indices into _r32 / _r16 / _qf
    # Sec A: R32[0-3] → R16[0-1] → QF[0]
    # Sec B: R32[4-7] → R16[2-3] → QF[1]   (SF-1: QF[0] vs QF[1])
    # Sec C: R32[8-11] → R16[4-5] → QF[2]
    # Sec D: R32[12-15] → R16[6-7] → QF[3]  (SF-2: QF[2] vs QF[3])

    _sections = [
        ("A", _r32[0:4],  _r16[0:2], _qf[0]),
        ("B", _r32[4:8],  _r16[2:4], _qf[1]),
        ("C", _r32[8:12], _r16[4:6], _qf[2]),
        ("D", _r32[12:16],_r16[6:8], _qf[3]),
    ]
    _sf_labels = ["Section A/B winner", "Section C/D winner"]

    for _sec_id, _r32s, _r16s, _qfm in _sections:
        _qf_w = _proj_winner(*_qfm)
        _sec_title = f"Bracket Section {_sec_id}  —  QF favourite: {flag(_qf_w)} {_qf_w}"
        with st.expander(_sec_title, expanded=(_sec_id == "A")):
            _html = ""
            _html += '<div style="font-size:10px;color:#ff6b35;font-weight:700;letter-spacing:1px;margin-bottom:6px;">ROUND OF 32</div>'
            for _m in _r32s:
                _html += _row(_m[0], _m[1], "R32", _proj_winner(*_m))
            _html += '<div style="font-size:10px;color:#ff6b35;font-weight:700;letter-spacing:1px;margin:10px 0 6px;">ROUND OF 16</div>'
            for _m in _r16s:
                _html += _row(_m[0], _m[1], "R16", _proj_winner(*_m))
            _html += '<div style="font-size:10px;color:#ff6b35;font-weight:700;letter-spacing:1px;margin:10px 0 6px;">QUARTER-FINAL</div>'
            _html += _row(_qfm[0], _qfm[1], "QF", _qf_w)
            st.markdown(_html, unsafe_allow_html=True)

    # ── Semi-Finals ────────────────────────────────────────────────────────────
    st.markdown("#### Semi-Finals")
    _sf_html = ""
    for i, _m in enumerate(_sf):
        _sf_html += _row(_m[0], _m[1], "SF", _proj_winner(*_m))
    st.markdown(_sf_html, unsafe_allow_html=True)

    # ── Final ──────────────────────────────────────────────────────────────────
    st.markdown("#### 🏆 Final")
    _champ = _proj_winner(*_fin)
    _final_html = _row(_fin[0], _fin[1], "Final", _champ)
    _final_html += f"""
    <div style="text-align:center;margin-top:12px;padding:14px;background:#1e1e2e;
                border-radius:12px;border:1px solid #ff6b35;">
        <div style="font-size:11px;color:#888;margin-bottom:4px;">Projected Champion</div>
        <div style="font-size:28px;font-weight:800;color:#ff6b35;">
            {flag(_champ)} {_champ}
        </div>
        <div style="font-size:13px;color:#888;margin-top:4px;">
            {live_sim['champion'].get(_champ, 0)*100:.1f}% pre-tournament title probability
        </div>
    </div>"""
    st.markdown(_final_html, unsafe_allow_html=True)

    # ── Team path selector ────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Any team's projected path")
    _path_team = st.selectbox("Select team", sorted(ALL_TEAMS),
                              format_func=lambda t: f"{flag(t)} {t}", key="path_sel")

    # Find this team's R32 match and trace its path
    _path_r32 = next(((a, b) for a, b in _r32 if a == _path_team or b == _path_team), None)
    # Trace this team's projected path round by round
    _stage_names = ["R32", "R16", "QF", "SF", "Final"]
    _all_rounds  = [_r32, _r16, _qf, _sf, [_fin]]
    _cur_team    = _path_team
    _path_html   = ""
    for _sname, _round in zip(_stage_names, _all_rounds):
        _match = next(((a, b) for a, b in _round if a == _cur_team or b == _cur_team), None)
        if _match is None:
            break
        _opp  = _match[1] if _match[0] == _cur_team else _match[0]
        _pred = predict_match(live_elos.get(_cur_team, 1500), live_elos.get(_opp, 1500))
        _w    = _pred["win"] * 100
        _d    = _pred["draw"] * 100
        _l    = _pred["loss"] * 100
        _winner_here = _proj_winner(*_match)
        _col_t = "#ff6b35" if _winner_here == _cur_team else "#ccc"
        _path_html += f"""
        <div style="display:flex;align-items:center;gap:8px;padding:9px 12px;
                    background:#1e1e2e;border-radius:8px;margin-bottom:5px;
                    border:1px solid #2a2a3a;">
            <div style="font-size:9px;color:#555;width:34px;flex-shrink:0;text-align:center;">{_sname}</div>
            <div style="font-size:13px;font-weight:600;color:{_col_t};flex:1;min-width:0;
                        overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
                vs {flag(_opp)} {_opp}
            </div>
            <div style="font-size:11px;color:#ff6b35;font-weight:700;flex-shrink:0;">
                {_w:.0f}% / {_d:.0f}% / {_l:.0f}%
            </div>
        </div>"""
        if _winner_here != _cur_team:
            _path_html += '<div style="font-size:11px;color:#e74c3c;padding:4px 12px;">Eliminated here</div>'
            break
        _cur_team = _winner_here

    st.markdown(_path_html, unsafe_allow_html=True)
    st.caption(f"Championship probability: {live_sim['champion'].get(_path_team, 0)*100:.1f}%")

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


# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    f"Model: Elo + Poisson · {payload['n_sims']:,} simulations · "
    "Data: Kaggle results + eloratings.net"
)

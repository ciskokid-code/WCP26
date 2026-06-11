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
    load_saved_results,
)
from src.predictions import elo_to_expected_goals, predict_match, run_group_simulation, run_simulation
from src.schedule import get_schedule
from src.precompute import load_cache

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


def _top_scores(elo_a: float, elo_b: float, n: int = 5, max_goals: int = 7) -> list[tuple[int, int, float]]:
    """Return the n most probable exact scorelines sorted by probability."""
    mu_a, mu_b = elo_to_expected_goals(elo_a, elo_b)
    scores = [
        (i, j, _poisson.pmf(i, mu_a) * _poisson.pmf(j, mu_b))
        for i in range(max_goals + 1)
        for j in range(max_goals + 1)
    ]
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:n]


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

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Title Odds",
    "⚽ Groups",
    "📊 Match Predictor",
    "⚔️ H2H",
    "🎯 Bet Predictor",
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

    # ── Bracket helpers ────────────────────────────────────────────────────────
    _grp_order = list(GROUPS.keys())

    def _proj_winner(ta: str, tb: str) -> str:
        p = predict_match(live_elos.get(ta, 1500), live_elos.get(tb, 1500))
        return ta if p["win"] >= p["loss"] else tb

    def _advance(pairs):
        return [(_proj_winner(*pairs[i]), _proj_winner(*pairs[i + 1]))
                for i in range(0, len(pairs), 2)]

    # Best-3rd pool
    _p2s = {t: max(0.0, live_sim["qualified"].get(t, 0)
                   - live_sim["group_win"].get(t, 0))
            for t in ALL_TEAMS}
    _t3_pool = sorted([t for t in ALL_TEAMS if live_sim["group_win"].get(t, 1) < 0.50],
                      key=lambda t: _p2s[t], reverse=True)
    _seen_g3: set = set()
    _best3: list = []
    for _t in _t3_pool:
        _tg = next(g for g, ts in GROUPS.items() if _t in ts)
        if _tg not in _seen_g3:
            _seen_g3.add(_tg)
            _best3.append(_t)
        if len(_best3) == 8:
            break
    while len(_best3) < 8:
        _best3.append("TBD")

    _r32: list = []
    for i in range(8):
        _w, _ = _expected_group_pos(_grp_order[i], 1)
        _r32.append((_w, _best3[7 - i]))
    for i in range(4):
        _w, _ = _expected_group_pos(_grp_order[8 + i], 1)
        _r, _ = _expected_group_pos(_grp_order[i], 2)
        _r32.append((_w, _r))
    for i in range(4):
        _r1, _ = _expected_group_pos(_grp_order[8 + i], 2)
        _r2, _ = _expected_group_pos(_grp_order[4 + i], 2)
        _r32.append((_r1, _r2))

    _r16 = _advance(_r32)
    _qf  = _advance(_r16)
    _sf  = _advance(_qf)
    _fin = _advance(_sf)[0]
    _champ = _proj_winner(*_fin)

    # ── HTML generation ────────────────────────────────────────────────────────
    _CW = 124   # card width px
    _CH = 52    # card height px (two team rows)
    _GW = 16    # connector column width px

    def _grp_label(team: str) -> str:
        for g, ts in GROUPS.items():
            if team in ts:
                return f"Grp {g}"
        return ""

    def _mcard(ta: str, tb: str) -> str:
        """Match card: winner on top in orange, loser below in grey."""
        w   = _proj_winner(ta, tb)
        l   = tb if w == ta else ta
        pr  = predict_match(live_elos.get(ta, 1500), live_elos.get(tb, 1500))
        wp  = (pr["win"] if w == ta else pr["loss"]) * 100
        lp  = (pr["loss"] if w == ta else pr["win"]) * 100
        wg  = _grp_label(w)
        lg  = _grp_label(l)
        return (
            f'<div style="background:#161625;border:1px solid #2a2a3a;border-radius:6px;'
            f'width:{_CW}px;overflow:hidden;flex-shrink:0;margin:2px 0;box-sizing:border-box;">'
            # winner row
            f'<div style="display:flex;align-items:center;gap:3px;padding:5px 6px;background:#1c1c30;'
            f'border-bottom:1px solid #1a1a28;">'
            f'<span style="font-size:13px;flex-shrink:0;">{flag(w)}</span>'
            f'<div style="flex:1;overflow:hidden;">'
            f'<div style="font-size:10px;font-weight:700;color:#ff6b35;'
            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{w[:13]}</div>'
            f'<div style="font-size:7px;color:#444;line-height:1;">{wg}</div>'
            f'</div>'
            f'<span style="font-size:9px;font-weight:700;color:#ff6b35;'
            f'flex-shrink:0;margin-left:2px;">{wp:.0f}%</span>'
            f'</div>'
            # loser row
            f'<div style="display:flex;align-items:center;gap:3px;padding:5px 6px;">'
            f'<span style="font-size:13px;flex-shrink:0;">{flag(l)}</span>'
            f'<div style="flex:1;overflow:hidden;">'
            f'<div style="font-size:10px;color:#3a3a55;'
            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{l[:13]}</div>'
            f'<div style="font-size:7px;color:#2a2a3a;line-height:1;">{lg}</div>'
            f'</div>'
            f'<span style="font-size:9px;color:#2d2d45;flex-shrink:0;'
            f'margin-left:2px;">{lp:.0f}%</span>'
            f'</div>'
            f'</div>'
        )

    def _conn(n: int) -> str:
        """n L-shaped connectors pointing RIGHT (for left half of bracket)."""
        pairs = "".join(
            '<div style="flex:1;display:flex;flex-direction:column;">'
            '<div style="flex:1;border-right:2px solid #2d2d4a;border-bottom:2px solid #2d2d4a;border-radius:0 0 3px 0;"></div>'
            '<div style="flex:1;border-right:2px solid #2d2d4a;border-top:2px solid #2d2d4a;border-radius:0 3px 0 0;"></div>'
            '</div>'
            for _ in range(n)
        )
        return (
            f'<div style="width:{_GW}px;display:flex;flex-direction:column;'
            f'align-self:stretch;flex-shrink:0;">{pairs}</div>'
        )

    def _rcol(matches, center=False) -> str:
        jc = "center" if center else "space-around"
        cards = "".join(_mcard(a, b) for a, b in matches)
        return (
            f'<div style="display:flex;flex-direction:column;justify-content:{jc};'
            f'align-self:stretch;flex-shrink:0;width:{_CW}px;">{cards}</div>'
        )

    def _lbl(text: str) -> str:
        return (
            f'<div style="width:{_CW}px;text-align:center;font-size:8px;font-weight:700;'
            f'color:#ff6b35;letter-spacing:1px;flex-shrink:0;">{text}</div>'
        )

    def _lbl_gap() -> str:
        return f'<div style="width:{_GW}px;flex-shrink:0;"></div>'

    # Bracket height: 8 cards per half, each ~56px including gap
    _H = "510px"

    # SF→Final horizontal connectors
    _sf_conn = (
        f'<div style="width:{_GW}px;display:flex;flex-direction:column;'
        f'justify-content:center;align-self:stretch;flex-shrink:0;">'
        f'<div style="height:2px;background:#ff6b35;width:100%;"></div>'
        f'</div>'
    )

    # Champion label inside Final card
    _final_card = _mcard(*_fin)
    _champion_banner = (
        f'<div style="text-align:center;padding:8px 4px;background:#1a1a30;'
        f'border-radius:8px;border:1px solid #ff6b35;flex-shrink:0;">'
        f'<div style="font-size:8px;color:#888;letter-spacing:1px;margin-bottom:4px;">🏆 FINAL</div>'
        f'{_final_card}'
        f'<div style="font-size:10px;font-weight:700;color:#ff6b35;margin-top:6px;">'
        f'Champion: {flag(_champ)} {_champ}</div>'
        f'<div style="font-size:8px;color:#555;">'
        f'{live_sim["champion"].get(_champ,0)*100:.1f}% probability</div>'
        f'</div>'
    )

    # Full bracket HTML
    _bracket = (
        # Label row
        f'<div style="display:flex;align-items:center;padding-bottom:4px;gap:0;">'
        + _lbl("R32") + _lbl_gap() + _lbl("R16") + _lbl_gap()
        + _lbl("QF") + _lbl_gap() + _lbl("SF") + _lbl_gap()
        + _lbl("FINAL")
        + _lbl_gap() + _lbl("SF") + _lbl_gap()
        + _lbl("QF") + _lbl_gap() + _lbl("R16") + _lbl_gap() + _lbl("R32")
        + '</div>'
        # Bracket rows
        + f'<div style="display:flex;align-items:stretch;gap:0;min-height:{_H};">'
        # Left half (R32→SF, left to right)
        + _rcol(_r32[0:8])
        + _conn(4)
        + _rcol(_r16[0:4])
        + _conn(2)
        + _rcol(_qf[0:2])
        + _conn(1)
        + _rcol([_sf[0]], center=True)
        + _sf_conn
        # Final
        + (f'<div style="display:flex;flex-direction:column;justify-content:center;'
           f'flex-shrink:0;width:{_CW + 16}px;padding:0 8px;">{_champion_banner}</div>')
        + _sf_conn
        # Right half (SF→R32, displayed left-to-right with row-reverse)
        + _rcol([_sf[1]], center=True)
        + _conn(1)
        + _rcol(_qf[2:4])
        + _conn(2)
        + _rcol(_r16[4:8])
        + _conn(4)
        + _rcol(_r32[8:16])
        + '</div>'
    )

    # Wrap in scrollable container
    _min_w = (8 * (_CW + _GW) * 2) + _CW + 32
    st.markdown(
        f'<div style="overflow-x:auto;-webkit-overflow-scrolling:touch;'
        f'padding-bottom:8px;">'
        f'<div style="min-width:{_min_w}px;">'
        f'{_bracket}'
        f'</div></div>',
        unsafe_allow_html=True,
    )
    st.caption("← Scroll to see all rounds · Orange team is projected winner of each match")


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
# TAB 5 — Bet Predictor
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    _BP_TODAY = "2026-06-11"

    st.markdown(
        '<h2 style="margin-bottom:2px;">Bet Predictor</h2>'
        '<p style="color:#888;font-size:13px;margin-bottom:16px;">'
        'Model pick for every match. Score calibrated against historical WC averages '
        '(~2.7 goals/game) using Elo-ratio Poisson.</p>',
        unsafe_allow_html=True,
    )

    # ── Filter ─────────────────────────────────────────────────────────────────
    _all_fixtures = get_schedule()
    _bp_choice = st.radio(
        "Round",
        ["Today", "MD1", "MD2", "MD3", "Knockout"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _bp_date_label(d: str) -> str:
        _y, _m, _day = d.split("-")
        _n = int(_day)
        _sfx = "th" if 11 <= _n <= 13 else {1:"st",2:"nd",3:"rd"}.get(_n % 10, "th")
        return f"June {_n}{_sfx}, {_y}"

    def _score_card(home: str, away: str, ea: float, eb: float,
                    label: str = "", is_today: bool = False) -> str:
        """Scoreboard-style match card: teams left/right, big score in centre."""
        pr    = predict_match(ea, eb)
        top   = _top_scores(ea, eb, n=6)
        mls_a, mls_b = pr["most_likely_score"]

        # Alternative scorelines (skip the top pick itself)
        alts  = [(ga, gb, p) for ga, gb, p in top if not (ga == mls_a and gb == mls_b)][:3]
        alt_txt = "  ·  ".join(
            f"{home[:3].upper()} {ga}–{gb} {away[:3].upper()} ({p*100:.0f}%)"
            for ga, gb, p in alts
        )

        border = "#ff6b35" if is_today else "#2a2a3a"
        bg     = "#1b1b2e" if is_today else "#181828"

        today_badge = (
            '<span style="font-size:10px;font-weight:700;color:#ff6b35;'
            'background:#2a1000;padding:2px 8px;border-radius:10px;margin-left:8px;">'
            'TODAY</span>'
            if is_today else ""
        )
        lbl_html = (
            f'<div style="font-size:10px;color:#666;letter-spacing:1px;'
            f'text-transform:uppercase;margin-bottom:8px;">{label}{today_badge}</div>'
            if label else ""
        )

        # Which team is projected winner? colour their name orange
        winner = home if pr["win"] >= pr["loss"] else away
        hcol = "#ff6b35" if winner == home else "#ddd"
        acol = "#ff6b35" if winner == away else "#ddd"

        return (
            f'<div style="background:{bg};border:1px solid {border};'
            f'border-radius:10px;padding:12px 14px;margin-bottom:10px;">'
            f'{lbl_html}'
            # Scoreboard row
            f'<div style="display:flex;align-items:center;gap:8px;">'
            # Home
            f'<div style="flex:1;text-align:right;">'
            f'<div style="font-size:15px;font-weight:700;color:{hcol};">'
            f'{flag(home)} {home}</div>'
            f'<div style="font-size:11px;color:#555;margin-top:2px;">'
            f'{pr["win"]*100:.0f}% win</div>'
            f'</div>'
            # Score box
            f'<div style="background:#0d1117;border:2px solid {"#ff6b35" if is_today else "#333"};'
            f'border-radius:8px;padding:6px 12px;text-align:center;flex-shrink:0;min-width:70px;">'
            f'<div style="font-size:26px;font-weight:900;color:#fff;line-height:1;'
            f'letter-spacing:2px;">{mls_a} – {mls_b}</div>'
            f'<div style="font-size:8px;color:#ff6b35;letter-spacing:1px;margin-top:2px;">'
            f'MODEL PICK</div>'
            f'</div>'
            # Away
            f'<div style="flex:1;text-align:left;">'
            f'<div style="font-size:15px;font-weight:700;color:{acol};">'
            f'{flag(away)} {away}</div>'
            f'<div style="font-size:11px;color:#555;margin-top:2px;">'
            f'{pr["loss"]*100:.0f}% win</div>'
            f'</div>'
            f'</div>'
            # Draw + alternatives row
            f'<div style="margin-top:8px;font-size:11px;color:#555;text-align:center;">'
            f'Draw {pr["draw"]*100:.0f}%'
            f'<span style="margin:0 8px;color:#2a2a3a;">|</span>'
            f'<span style="color:#444;">Also: {alt_txt}</span>'
            f'</div>'
            f'</div>'
        )

    # ── Group stage view ───────────────────────────────────────────────────────
    if _bp_choice != "Knockout":
        if _bp_choice == "Today":
            _shown = [f for f in _all_fixtures if f["date"] == _BP_TODAY]
        elif _bp_choice == "MD1":
            _shown = [f for f in _all_fixtures if f["matchday"] == 1]
        elif _bp_choice == "MD2":
            _shown = [f for f in _all_fixtures if f["matchday"] == 2]
        else:
            _shown = [f for f in _all_fixtures if f["matchday"] == 3]

        _by_date2: dict[str, list] = {}
        for _f in _shown:
            _by_date2.setdefault(_f["date"], []).append(_f)

        if not _shown:
            st.info("No matches for this filter.")
        else:
            for _date, _fixtures in sorted(_by_date2.items()):
                _is_today = (_date == _BP_TODAY)
                _hdr_col  = "#ff6b35" if _is_today else "#777"
                st.markdown(
                    f'<div style="font-size:12px;font-weight:700;color:{_hdr_col};'
                    f'letter-spacing:1.5px;text-transform:uppercase;'
                    f'margin:20px 0 8px;border-bottom:1px solid #222;padding-bottom:4px;">'
                    f'{"🔴 " if _is_today else ""}{_bp_date_label(_date)}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                for _fx in _fixtures:
                    _ea2 = live_elos.get(_fx["home"], 1500)
                    _eb2 = live_elos.get(_fx["away"], 1500)
                    _lbl = f"Group {_fx['group']} · Matchday {_fx['matchday']}"
                    st.markdown(
                        _score_card(_fx["home"], _fx["away"], _ea2, _eb2,
                                    label=_lbl, is_today=_is_today),
                        unsafe_allow_html=True,
                    )

    # ── Knockout view ──────────────────────────────────────────────────────────
    else:
        # Reuse the bracket data already computed for tab3
        _ko_rounds = [
            ("Round of 32", _r32),
            ("Round of 16", _r16),
            ("Quarter-finals", _qf),
            ("Semi-finals", _sf),
            ("Final", [_fin]),
        ]
        for _rnd_name, _pairs in _ko_rounds:
            st.markdown(
                f'<div style="font-size:12px;font-weight:700;color:#ff6b35;'
                f'letter-spacing:1.5px;text-transform:uppercase;'
                f'margin:20px 0 8px;border-bottom:1px solid #222;padding-bottom:4px;">'
                f'{_rnd_name}</div>',
                unsafe_allow_html=True,
            )
            for _ta, _tb in _pairs:
                _ea3 = live_elos.get(_ta, 1500)
                _eb3 = live_elos.get(_tb, 1500)
                st.markdown(
                    _score_card(_ta, _tb, _ea3, _eb3, label=_rnd_name),
                    unsafe_allow_html=True,
                )

    st.caption("Score = single most likely exact result · Elo-ratio Poisson model, calibrated to WC averages")


# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    f"Model: Elo + Poisson · {payload['n_sims']:,} simulations · "
    "Data: Kaggle results + eloratings.net"
)

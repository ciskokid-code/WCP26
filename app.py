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
from src.predictions import predict_match, run_group_simulation, run_simulation
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


@st.cache_data(show_spinner="Updating odds…")
def _compute_live_sim(_h: int, _res: list, _base_elos: dict) -> dict:
    _known = build_known_group_results(_res)
    _elos  = apply_results_to_elo(_base_elos, _res)
    return run_simulation(_elos, n_sims=20_000, seed=42, known_group_results=_known)


live_sim  = _compute_live_sim(_live_hash, _live_results_global, elos)
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

    st.markdown(
        '<p style="color:#ff6b35;font-size:12px;font-weight:700;letter-spacing:2px;'
        'text-transform:uppercase;margin-bottom:2px;">Projected Bracket</p>',
        unsafe_allow_html=True,
    )
    st.title("Match Predictor")
    st.caption("Visual bracket showing projected matchups and chances at every stage.")

    # ── Bracket helpers ────────────────────────────────────────────────────────
    _grp_order = list(GROUPS.keys())

    def _proj_winner(ta: str, tb: str) -> str:
        pred = predict_match(live_elos.get(ta, 1500), live_elos.get(tb, 1500))
        return ta if pred["win"] >= pred["loss"] else tb

    def _advance(pairs):
        return [(_proj_winner(*pairs[i]), _proj_winner(*pairs[i + 1]))
                for i in range(0, len(pairs), 2)]

    _CW = "148px"   # card width
    _GW = "20px"    # connector/gap width

    def _bcard(ta: str, tb: str) -> str:
        """Bracket match card: winner in orange on top, loser below."""
        w = _proj_winner(ta, tb)
        l = tb if w == ta else ta
        pred = predict_match(live_elos.get(ta, 1500), live_elos.get(tb, 1500))
        pct = int(max(pred["win"], pred["loss"]) * 100)
        return (
            f'<div style="background:#161625;border:1px solid #2a2a3a;border-radius:6px;'
            f'width:{_CW};overflow:hidden;box-sizing:border-box;flex-shrink:0;">'
            f'<div style="padding:5px 7px;border-bottom:1px solid #2a2a3a;'
            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
            f'<span style="font-size:12px;">{flag(w)}</span>'
            f'<span style="font-size:11px;font-weight:700;color:#ff6b35;"> {w[:15]}</span>'
            f'</div>'
            f'<div style="padding:5px 7px;border-bottom:1px solid #2a2a3a;'
            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
            f'<span style="font-size:12px;">{flag(l)}</span>'
            f'<span style="font-size:11px;color:#484860;"> {l[:15]}</span>'
            f'</div>'
            f'<div style="padding:2px 7px;font-size:9px;color:#ff6b35;">{pct}% advance</div>'
            f'</div>'
        )

    def _conn(n: int) -> str:
        """n L-shaped connectors stacked — links n pairs of cards to next round."""
        pairs = "".join(
            '<div style="flex:1;display:flex;flex-direction:column;">'
            '<div style="flex:1;border-right:2px solid #2d2d4a;border-bottom:2px solid #2d2d4a;border-radius:0 0 4px 0;"></div>'
            '<div style="flex:1;border-right:2px solid #2d2d4a;border-top:2px solid #2d2d4a;border-radius:0 4px 0 0;"></div>'
            '</div>'
            for _ in range(n)
        )
        return f'<div style="width:{_GW};display:flex;flex-direction:column;align-self:stretch;flex-shrink:0;">{pairs}</div>'

    def _rcol(matches, center=False) -> str:
        jc = "center" if center else "space-around"
        cards = "".join(_bcard(a, b) for a, b in matches)
        return (
            f'<div style="display:flex;flex-direction:column;justify-content:{jc};'
            f'align-self:stretch;gap:4px;flex-shrink:0;width:{_CW};">{cards}</div>'
        )

    def _half(r32m, r16m, qfm, sfm, height="640px") -> str:
        """One bracket half: 8 R32 → 4 R16 → 2 QF → 1 SF."""
        lbl = (
            f'<div style="display:flex;gap:0;padding-bottom:4px;">'
            + "".join(
                f'<div style="width:{_CW};text-align:center;font-size:9px;color:#ff6b35;'
                f'font-weight:700;letter-spacing:1px;flex-shrink:0;">{r}</div>'
                f'<div style="width:{_GW};flex-shrink:0;"></div>'
                for r in ["R32", "R16", "QF", "SF"]
            )
            + "</div>"
        )
        body = (
            f'<div style="display:flex;align-items:stretch;gap:0;min-height:{height};">'
            f'{_rcol(r32m)}{_conn(4)}{_rcol(r16m)}{_conn(2)}{_rcol(qfm)}{_conn(1)}{_rcol([sfm], center=True)}'
            f'</div>'
        )
        return lbl + body

    # ── Build R32 bracket ──────────────────────────────────────────────────────
    # Best 3rd candidates: highest P(qualify) teams that aren't group favourites
    _p2 = {t: max(0.0, live_sim["qualified"].get(t, 0) - live_sim["group_win"].get(t, 0))
           for t in ALL_TEAMS}
    _t3_pool = sorted(
        [t for t in ALL_TEAMS if live_sim["group_win"].get(t, 1) < 0.50],
        key=lambda t: _p2[t], reverse=True,
    )
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

    _r32: list[tuple[str, str]] = []
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

    # ── Render bracket ─────────────────────────────────────────────────────────
    _champ = _proj_winner(*_fin)

    _final_card = (
        f'<div style="text-align:center;padding:16px 20px;background:#1a1a2e;'
        f'border-radius:12px;border:2px solid #ff6b35;margin:16px 0;">'
        f'<div style="font-size:10px;color:#888;letter-spacing:2px;margin-bottom:10px;">🏆 PROJECTED FINAL</div>'
        f'<div style="display:flex;justify-content:center;gap:12px;align-items:center;flex-wrap:wrap;">'
        f'<div style="text-align:center;">'
        f'<div style="font-size:36px;">{flag(_fin[0])}</div>'
        f'<div style="font-size:14px;font-weight:700;color:#ff6b35;">{_fin[0]}</div>'
        f'<div style="font-size:11px;color:#888;">{live_sim["finalist"].get(_fin[0],0)*100:.0f}% to reach final</div>'
        f'</div>'
        f'<div style="font-size:18px;color:#555;font-weight:700;">vs</div>'
        f'<div style="text-align:center;">'
        f'<div style="font-size:36px;">{flag(_fin[1])}</div>'
        f'<div style="font-size:14px;font-weight:700;color:#ccc;">{_fin[1]}</div>'
        f'<div style="font-size:11px;color:#888;">{live_sim["finalist"].get(_fin[1],0)*100:.0f}% to reach final</div>'
        f'</div>'
        f'</div>'
        f'<div style="margin-top:12px;font-size:22px;font-weight:800;color:#ff6b35;">'
        f'Projected champion: {flag(_champ)} {_champ}'
        f'</div>'
        f'<div style="font-size:12px;color:#888;margin-top:2px;">'
        f'{live_sim["champion"].get(_champ,0)*100:.1f}% title probability'
        f'</div>'
        f'</div>'
    )

    _full_bracket = (
        f'<div style="overflow-x:auto;-webkit-overflow-scrolling:touch;">'
        f'<div style="min-width:680px;">'
        f'<div style="font-size:11px;color:#888;font-weight:600;margin-bottom:4px;">TOP HALF</div>'
        f'{_half(_r32[0:8], _r16[0:4], _qf[0:2], _sf[0])}'
        f'{_final_card}'
        f'<div style="font-size:11px;color:#888;font-weight:600;margin:4px 0 4px;">BOTTOM HALF</div>'
        f'{_half(_r32[8:16], _r16[4:8], _qf[2:4], _sf[1])}'
        f'</div>'
        f'</div>'
    )

    st.markdown(_full_bracket, unsafe_allow_html=True)

    st.caption(
        "Bracket built from Elo-based group qualification simulation. "
        "Winner of each match advances. Scroll right on mobile."
    )

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

"""
2026 World Cup Prediction Model
Run: streamlit run app.py
"""

from __future__ import annotations

import datetime
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
from src.precompute import load_cache
from src.schedule import get_schedule

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="WC 2026 Model",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
[data-testid="collapsedControl"] { display: none; }
[data-testid="stSidebar"] { display: none; }
.block-container { padding: 1rem 1rem 4rem; max-width: 860px; }
.js-plotly-plot { max-width: 100%; }
[data-testid="metric-container"] { padding: 0.4rem 0.6rem; }
/* tab bar scrolls horizontally on narrow screens */
[data-testid="stTabs"] > div:first-child {
    overflow-x: auto; white-space: nowrap;
    -webkit-overflow-scrolling: touch; scrollbar-width: none;
}
[data-testid="stTabs"] > div:first-child::-webkit-scrollbar { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Shared helpers ────────────────────────────────────────────────────────────

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
    html = ""
    for i, (team, p) in enumerate(teams_probs, start_rank):
        bar_w = (p / max_p) * 100 if max_p > 0 else 0
        html += (
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;'
            f'padding:10px 12px;background:#1e1e2e;border-radius:8px;border:1px solid #2a2a3a;">'
            f'<div style="width:20px;text-align:right;color:#555;font-size:12px;flex-shrink:0;">{i}</div>'
            f'<div style="font-size:20px;flex-shrink:0;">{flag(team)}</div>'
            f'<div style="flex:1;font-size:13px;font-weight:500;min-width:0;'
            f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{team}</div>'
            f'<div style="width:90px;background:#2a2a2a;border-radius:4px;height:8px;'
            f'flex-shrink:0;overflow:hidden;">'
            f'<div style="background:#ff6b35;width:{bar_w:.1f}%;height:8px;border-radius:4px;"></div></div>'
            f'<div style="width:44px;text-align:right;font-size:13px;font-weight:700;'
            f'color:#ff6b35;flex-shrink:0;">{p * 100:.1f}%</div>'
            f'</div>'
        )
    return html


def _top_scores(elo_a: float, elo_b: float, n: int = 6, max_goals: int = 7) -> list[tuple[int, int, float]]:
    mu_a, mu_b = elo_to_expected_goals(elo_a, elo_b)
    scores = [
        (i, j, _poisson.pmf(i, mu_a) * _poisson.pmf(j, mu_b))
        for i in range(max_goals + 1)
        for j in range(max_goals + 1)
    ]
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:n]


# ── Cached loaders ────────────────────────────────────────────────────────────

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


_ESPN_NAMES: dict[str, str] = {
    "Czech Republic":     "Czechia",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Curacao":            "Curaçao",
    "Congo DR":           "DR Congo",
    "Côte d'Ivoire":      "Ivory Coast",
    "Cape Verde Islands": "Cape Verde",
    "USA":                "United States",
    "Korea Republic":     "South Korea",
}


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_live_scores() -> dict[tuple[str, str], dict]:
    import requests  # noqa: PLC0415
    try:
        resp = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard",
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return {}

    def _canon(name: str) -> str:
        return _ESPN_NAMES.get(name, name)

    out: dict[tuple[str, str], dict] = {}
    for event in data.get("events", []):
        comps       = event.get("competitions", [{}])[0]
        competitors = comps.get("competitors", [])
        if len(competitors) < 2:
            continue
        home_c  = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
        away_c  = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
        h_name  = _canon(home_c.get("team", {}).get("displayName", ""))
        a_name  = _canon(away_c.get("team", {}).get("displayName", ""))
        h_score = home_c.get("score", "")
        a_score = away_c.get("score", "")
        st_obj  = event.get("status", {})
        st_type = st_obj.get("type", {}).get("name", "")
        clock   = st_obj.get("displayClock", "")
        period  = st_obj.get("period", 0)

        _live_types = {"STATUS_IN_PROGRESS", "STATUS_FIRST_HALF", "STATUS_SECOND_HALF",
                       "STATUS_EXTRA_TIME", "STATUS_PENALTY"}
        is_live  = st_type in _live_types
        is_ht    = st_type == "STATUS_HALFTIME"
        is_final = st_type == "STATUS_FINAL"

        if st_type == "STATUS_FIRST_HALF":
            display = f"🔴 LIVE {clock} 1H"
        elif st_type == "STATUS_SECOND_HALF":
            display = f"🔴 LIVE {clock} 2H"
        elif st_type == "STATUS_EXTRA_TIME":
            display = f"🔴 LIVE {clock} ET"
        elif st_type == "STATUS_PENALTY":
            display = "🔴 LIVE PENS"
        elif is_live:
            display = f"🔴 LIVE {clock} {'2H' if period >= 2 else '1H'}"
        elif is_ht:
            display = "⏸ HALF TIME"
        elif is_final:
            display = "✅ FULL TIME"
        else:
            display = st_obj.get("type", {}).get("shortDetail", "")

        out[(h_name, a_name)] = {
            "score_h": h_score, "score_a": a_score,
            "status": display, "is_live": is_live or is_ht, "is_final": is_final,
        }
    return out


# ── Settings ──────────────────────────────────────────────────────────────────

with st.expander("⚙️ Model settings", expanded=False):
    _c1, _c2, _c3 = st.columns([3, 2, 1])
    with _c1:
        n_sims = st.select_slider("Simulations",
                                  options=[5_000, 10_000, 25_000, 50_000, 100_000],
                                  value=50_000)
    with _c2:
        seed = st.number_input("RNG seed", value=42, min_value=0, step=1)
    with _c3:
        st.write("")
        if st.button("↺ Refresh", use_container_width=True):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()

# ── Load data ─────────────────────────────────────────────────────────────────

elos    = _load_elos()
payload = _load_simulation(n_sims, seed)
sim     = payload["simulation"]
group_probs_all = payload["group_probs"]

if "results_2026" not in st.session_state:
    st.session_state["results_2026"] = load_saved_results()

_live_results_global = st.session_state["results_2026"]
_live_hash = hash(tuple(
    (r["home"], r["away"], r["home_score"], r["away_score"])
    for r in sorted(_live_results_global, key=lambda x: (x["date"], x["home"]))
))


@st.cache_data(show_spinner="Updating simulation…")
def _compute_live_sim(_h: int, _res_tuple: tuple, _base_elos: dict) -> dict:
    if not _res_tuple:
        return payload
    _res   = list(_res_tuple)
    _known = build_known_group_results(_res)
    _elos  = apply_results_to_elo(_base_elos, _res)
    _sim   = run_simulation(_elos, n_sims=20_000, seed=42, known_group_results=_known)
    return {"simulation": _sim, "group_probs": payload["group_probs"],
            "elos": _elos, "n_sims": 20_000}


_live_res_tuple = tuple(
    (r["home"], r["away"], r["home_score"], r["away_score"])
    for r in sorted(_live_results_global, key=lambda x: (x["date"], x["home"]))
)
_live_payload = _compute_live_sim(_live_hash, _live_res_tuple, elos)
live_sim  = _live_payload["simulation"]
live_elos = apply_results_to_elo(elos, _live_results_global)


def _expected_group_pos(group: str, pos: int) -> tuple[str, float]:
    teams     = GROUPS[group]
    standings = get_group_standings(group, _live_results_global, teams)
    if all(row["played"] == 3 for row in standings):
        return standings[pos - 1]["team"], 1.0
    if pos == 1:
        probs = [(t, live_sim["group_win"].get(t, 0)) for t in teams]
    else:
        probs = [(t, max(0.0, live_sim["qualified"].get(t, 0)
                        - live_sim["group_win"].get(t, 0))) for t in teams]
    probs.sort(key=lambda x: x[1], reverse=True)
    return probs[0][0], probs[0][1]


# ── Bracket data ──────────────────────────────────────────────────────────────

_grp_order = list(GROUPS.keys())


def _proj_winner(ta: str, tb: str) -> str:
    p = predict_match(live_elos.get(ta, 1500), live_elos.get(tb, 1500))
    return ta if p["win"] >= p["loss"] else tb


def _advance(pairs):
    return [(_proj_winner(*pairs[i]), _proj_winner(*pairs[i + 1]))
            for i in range(0, len(pairs), 2)]


_p2s = {t: max(0.0, live_sim["qualified"].get(t, 0) - live_sim["group_win"].get(t, 0))
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

_r16   = _advance(_r32)
_qf    = _advance(_r16)
_sf    = _advance(_qf)
_fin   = _advance(_sf)[0]
_champ = _proj_winner(*_fin)

# ── Bet Predictor helpers (used inside tab) ───────────────────────────────────

_live_scores = _fetch_live_scores()


def _bp_date_label(d: str) -> str:
    _n   = int(d.split("-")[2])
    _sfx = "th" if 11 <= _n <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(_n % 10, "th")
    return f"June {_n}{_sfx}, 2026"


def _score_card(home: str, away: str, ea: float, eb: float,
                label: str = "", is_today: bool = False) -> str:
    pr       = predict_match(ea, eb)
    top      = _top_scores(ea, eb, n=6)
    mls_a, mls_b = pr["most_likely_score"]
    alts     = [(ga, gb, p) for ga, gb, p in top if not (ga == mls_a and gb == mls_b)][:3]
    alt_txt  = "  ·  ".join(
        f"{home[:3].upper()} {ga}–{gb} {away[:3].upper()} ({p*100:.0f}%)"
        for ga, gb, p in alts
    )

    live = _live_scores.get((home, away)) or _live_scores.get((away, home))
    if live and _live_scores.get((away, home)) and not _live_scores.get((home, away)):
        live = {**live, "score_h": live["score_a"], "score_a": live["score_h"]}

    has_live  = live is not None
    is_active = has_live and live["is_live"]
    is_final  = has_live and live["is_final"]

    border = "#cc2200" if is_active else ("#22aa55" if is_final else ("#ff6b35" if is_today else "#2a2a3a"))

    today_badge = (
        '<span style="font-size:10px;font-weight:700;color:#ff6b35;background:#2a1000;'
        'padding:2px 8px;border-radius:10px;margin-left:6px;">TODAY</span>'
        if is_today else ""
    )
    live_badge = ""
    if is_active:
        live_badge = (
            f'<span style="font-size:10px;font-weight:700;color:#fff;background:#cc2200;'
            f'padding:2px 8px;border-radius:10px;margin-left:6px;">{live["status"]}</span>'
        )
    elif is_final:
        live_badge = (
            '<span style="font-size:10px;font-weight:700;color:#22aa55;background:#0a2a14;'
            'padding:2px 8px;border-radius:10px;margin-left:6px;">✅ FULL TIME</span>'
        )

    lbl_html = (
        f'<div style="font-size:10px;color:#666;letter-spacing:1px;text-transform:uppercase;'
        f'margin-bottom:8px;">{label}{today_badge}{live_badge}</div>'
    ) if (label or today_badge or live_badge) else ""

    if mls_a == mls_b:
        hcol = acol = "#ddd"
    elif mls_a > mls_b:
        hcol, acol = "#ff6b35", "#ddd"
    else:
        hcol, acol = "#ddd", "#ff6b35"

    if has_live and (is_active or is_final):
        sh, sa = live["score_h"], live["score_a"]
        sc = "#cc2200" if is_active else "#22aa55"
        score_box = (
            f'<div style="background:#0d1117;border:2px solid {sc};border-radius:8px;'
            f'padding:5px 12px;text-align:center;flex-shrink:0;min-width:70px;">'
            f'<div style="font-size:26px;font-weight:900;color:#fff;line-height:1;letter-spacing:2px;">{sh} – {sa}</div>'
            f'<div style="font-size:8px;color:{sc};letter-spacing:1px;margin-top:2px;">{"LIVE" if is_active else "FINAL"}</div>'
            f'<div style="border-top:1px solid #222;margin:4px 0;"></div>'
            f'<div style="font-size:26px;font-weight:900;color:#ff6b35;line-height:1;letter-spacing:2px;">{mls_a} – {mls_b}</div>'
            f'<div style="font-size:8px;color:#ff6b35;letter-spacing:1px;margin-top:2px;">MODEL</div>'
            f'</div>'
        )
    else:
        score_box = (
            f'<div style="background:#0d1117;border:2px solid {"#ff6b35" if is_today else "#333"};'
            f'border-radius:8px;padding:6px 12px;text-align:center;flex-shrink:0;min-width:70px;">'
            f'<div style="font-size:26px;font-weight:900;color:#fff;line-height:1;letter-spacing:2px;">{mls_a} – {mls_b}</div>'
            f'<div style="font-size:8px;color:#ff6b35;letter-spacing:1px;margin-top:2px;">MODEL PICK</div>'
            f'</div>'
        )

    return (
        f'<div style="background:#181828;border:1px solid {border};border-radius:10px;'
        f'padding:12px 14px;margin-bottom:10px;">'
        f'{lbl_html}'
        f'<div style="display:flex;align-items:center;gap:8px;">'
        f'<div style="flex:1;text-align:right;">'
        f'<div style="font-size:15px;font-weight:700;color:{hcol};">{flag(home)} {home}</div>'
        f'<div style="font-size:11px;color:#555;margin-top:2px;">{pr["win"]*100:.0f}% win</div>'
        f'</div>'
        f'{score_box}'
        f'<div style="flex:1;text-align:left;">'
        f'<div style="font-size:15px;font-weight:700;color:{acol};">{flag(away)} {away}</div>'
        f'<div style="font-size:11px;color:#555;margin-top:2px;">{pr["loss"]*100:.0f}% win</div>'
        f'</div>'
        f'</div>'
        f'<div style="margin-top:8px;font-size:11px;color:#555;text-align:center;">'
        f'Draw {pr["draw"]*100:.0f}%'
        f'<span style="margin:0 8px;color:#2a2a3a;">|</span>'
        f'<span style="color:#444;">Also: {alt_txt}</span>'
        f'</div>'
        f'</div>'
    )


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Title Odds",
    "🎯 Bet Predictor",
    "⚽ Groups",
    "📊 Match Predictor",
    "⚔️ H2H",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Title Odds
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    _champ_probs = live_sim["champion"]
    _ranked      = sorted(ALL_TEAMS, key=lambda t: _champ_probs.get(t, 0), reverse=True)
    _team_group  = {t: g for g, teams in GROUPS.items() for t in teams}

    st.markdown(
        f'<p style="color:#ff6b35;font-size:12px;font-weight:700;letter-spacing:2px;'
        f'text-transform:uppercase;margin-bottom:4px;">The Model\'s View</p>'
        f'<h1 style="font-size:36px;font-weight:800;margin:0 0 6px 0;">Who wins the World Cup?</h1>'
        f'<p style="color:#888;font-size:14px;margin:0 0 24px 0;">'
        f'Simulated <b>{payload["n_sims"]:,}</b> times end-to-end.</p>',
        unsafe_allow_html=True,
    )

    _hero = '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:28px;">'
    for _team, _lbl in zip(_ranked[:3], ["FAVOURITE", "SECOND FAVOURITE", "THIRD FAVOURITE"]):
        _p = _champ_probs.get(_team, 0)
        _hero += (
            f'<div style="background:#1e1e2e;border-radius:12px;padding:18px 20px;border:1px solid #2a2a3a;">'
            f'<div style="font-size:10px;font-weight:700;letter-spacing:2px;color:#888;'
            f'text-transform:uppercase;margin-bottom:8px;">{_lbl}</div>'
            f'<div style="font-size:20px;font-weight:700;margin-bottom:2px;">{flag(_team)} {_team}</div>'
            f'<div style="font-size:40px;font-weight:800;color:#ff6b35;line-height:1.1;margin-bottom:4px;">'
            f'{_p * 100:.1f}%</div>'
            f'<div style="font-size:11px;color:#555;">to lift the trophy · Group {_team_group.get(_team,"?")}</div>'
            f'</div>'
        )
    _hero += "</div>"
    st.markdown(_hero, unsafe_allow_html=True)

    st.markdown("### All contenders — chance to win")
    _max_cp = _champ_probs.get(_ranked[0], 1)
    st.markdown(_hbar_list([(t, _champ_probs.get(t, 0)) for t in _ranked], _max_cp),
                unsafe_allow_html=True)
    st.caption(f"Model: Elo + Poisson · {n_sims:,} simulations")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Bet Predictor
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown(
        '<h2 style="margin-bottom:2px;">Bet Predictor</h2>'
        '<p style="color:#888;font-size:13px;margin-bottom:16px;">'
        'Model pick for every match with live scores. Refreshes every 60 s.</p>',
        unsafe_allow_html=True,
    )

    _BP_TODAY     = datetime.date.today().strftime("%Y-%m-%d")
    _all_fixtures = get_schedule()
    _bp_choice    = st.radio("Round", ["Today", "MD1", "MD2", "MD3", "Knockout"],
                              horizontal=True, label_visibility="collapsed")

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
                    f'<div style="font-size:12px;font-weight:700;color:{_hdr_col};letter-spacing:1.5px;'
                    f'text-transform:uppercase;margin:20px 0 8px;border-bottom:1px solid #222;padding-bottom:4px;">'
                    f'{"🔴 " if _is_today else ""}{_bp_date_label(_date)}</div>',
                    unsafe_allow_html=True,
                )
                for _fx in _fixtures:
                    st.markdown(
                        _score_card(_fx["home"], _fx["away"],
                                    live_elos.get(_fx["home"], 1500),
                                    live_elos.get(_fx["away"], 1500),
                                    label=f"Group {_fx['group']} · Matchday {_fx['matchday']}",
                                    is_today=_is_today),
                        unsafe_allow_html=True,
                    )
    else:
        for _rnd_name, _pairs in [
            ("Round of 32", _r32), ("Round of 16", _r16),
            ("Quarter-finals", _qf), ("Semi-finals", _sf), ("Final", [_fin]),
        ]:
            st.markdown(
                f'<div style="font-size:12px;font-weight:700;color:#ff6b35;letter-spacing:1.5px;'
                f'text-transform:uppercase;margin:20px 0 8px;border-bottom:1px solid #222;padding-bottom:4px;">'
                f'{_rnd_name}</div>',
                unsafe_allow_html=True,
            )
            for _ta, _tb in _pairs:
                st.markdown(
                    _score_card(_ta, _tb, live_elos.get(_ta, 1500), live_elos.get(_tb, 1500),
                                label=_rnd_name),
                    unsafe_allow_html=True,
                )

    st.caption("Score = most likely exact result · Elo-ratio Poisson · live data from ESPN")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Groups
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown(
        '<h2 style="margin-bottom:4px;">Groups</h2>'
        '<p style="color:#888;font-size:13px;margin-bottom:20px;">'
        'Bar = chance to win the group. Numbers show 1st / 2nd finish probability.</p>',
        unsafe_allow_html=True,
    )

    for _g, _g_teams in GROUPS.items():
        _gp     = group_probs_all[_g]
        _sorted = sorted(_g_teams, key=lambda t: _gp[t][1], reverse=True)
        _max_p1 = _gp[_sorted[0]][1] or 0.01
        _html   = (
            f'<div style="margin-bottom:24px;">'
            f'<div style="font-size:13px;font-weight:700;color:#ff6b35;letter-spacing:1px;'
            f'text-transform:uppercase;margin-bottom:8px;">Group {_g}</div>'
        )
        for _t in _sorted:
            _p1  = _gp[_t][1]
            _p2  = _gp[_t][2]
            _bar = (_p1 / _max_p1) * 100
            _html += (
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;'
                f'padding:9px 12px;background:#1e1e2e;border-radius:8px;border:1px solid #2a2a3a;">'
                f'<div style="font-size:19px;flex-shrink:0;">{flag(_t)}</div>'
                f'<div style="flex:1;font-size:13px;font-weight:500;min-width:0;'
                f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{_t}</div>'
                f'<div style="width:80px;background:#2a2a2a;border-radius:4px;height:7px;flex-shrink:0;overflow:hidden;">'
                f'<div style="background:#ff6b35;width:{_bar:.1f}%;height:7px;border-radius:4px;"></div></div>'
                f'<div style="width:96px;text-align:right;font-size:12px;color:#aaa;flex-shrink:0;white-space:nowrap;">'
                f'1st <b style="color:#ff6b35;">{_p1*100:.0f}%</b>'
                f'&nbsp;·&nbsp;2nd <b style="color:#888;">{_p2*100:.0f}%</b></div>'
                f'</div>'
            )
        _html += "</div>"
        st.markdown(_html, unsafe_allow_html=True)

    st.caption(f"Model: Elo + Poisson · {n_sims:,} simulations")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Match Predictor bracket
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown(
        '<h2 style="margin-bottom:4px;">Match Predictor</h2>'
        '<p style="color:#888;font-size:13px;margin-bottom:12px;">'
        'Projected bracket from R32 to the Final. Scroll right to see all rounds.</p>',
        unsafe_allow_html=True,
    )

    _CW = 124
    _GW = 16

    def _grp_label(team: str) -> str:
        for g, ts in GROUPS.items():
            if team in ts:
                return f"Grp {g}"
        return ""

    def _mcard(ta: str, tb: str) -> str:
        w  = _proj_winner(ta, tb)
        l  = tb if w == ta else ta
        pr = predict_match(live_elos.get(ta, 1500), live_elos.get(tb, 1500))
        wp = (pr["win"] if w == ta else pr["loss"]) * 100
        lp = (pr["loss"] if w == ta else pr["win"]) * 100
        return (
            f'<div style="background:#161625;border:1px solid #2a2a3a;border-radius:6px;'
            f'width:{_CW}px;overflow:hidden;flex-shrink:0;margin:2px 0;box-sizing:border-box;">'
            f'<div style="display:flex;align-items:center;gap:3px;padding:5px 6px;background:#1c1c30;'
            f'border-bottom:1px solid #1a1a28;">'
            f'<span style="font-size:13px;flex-shrink:0;">{flag(w)}</span>'
            f'<div style="flex:1;overflow:hidden;">'
            f'<div style="font-size:10px;font-weight:700;color:#ff6b35;white-space:nowrap;'
            f'overflow:hidden;text-overflow:ellipsis;">{w[:13]}</div>'
            f'<div style="font-size:7px;color:#444;line-height:1;">{_grp_label(w)}</div>'
            f'</div>'
            f'<span style="font-size:9px;font-weight:700;color:#ff6b35;flex-shrink:0;margin-left:2px;">{wp:.0f}%</span>'
            f'</div>'
            f'<div style="display:flex;align-items:center;gap:3px;padding:5px 6px;">'
            f'<span style="font-size:13px;flex-shrink:0;">{flag(l)}</span>'
            f'<div style="flex:1;overflow:hidden;">'
            f'<div style="font-size:10px;color:#3a3a55;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{l[:13]}</div>'
            f'<div style="font-size:7px;color:#2a2a3a;line-height:1;">{_grp_label(l)}</div>'
            f'</div>'
            f'<span style="font-size:9px;color:#2d2d45;flex-shrink:0;margin-left:2px;">{lp:.0f}%</span>'
            f'</div>'
            f'</div>'
        )

    def _conn(n: int) -> str:
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
        return (
            f'<div style="display:flex;flex-direction:column;justify-content:{jc};'
            f'align-self:stretch;flex-shrink:0;width:{_CW}px;">'
            + "".join(_mcard(a, b) for a, b in matches)
            + '</div>'
        )

    def _lbl(text: str) -> str:
        return (
            f'<div style="width:{_CW}px;text-align:center;font-size:8px;font-weight:700;'
            f'color:#ff6b35;letter-spacing:1px;flex-shrink:0;">{text}</div>'
        )

    def _lbl_gap() -> str:
        return f'<div style="width:{_GW}px;flex-shrink:0;"></div>'

    _sf_conn = (
        f'<div style="width:{_GW}px;display:flex;flex-direction:column;justify-content:center;'
        f'align-self:stretch;flex-shrink:0;">'
        f'<div style="height:2px;background:#ff6b35;width:100%;"></div></div>'
    )
    _champion_banner = (
        f'<div style="text-align:center;padding:8px 4px;background:#1a1a30;'
        f'border-radius:8px;border:1px solid #ff6b35;flex-shrink:0;">'
        f'<div style="font-size:8px;color:#888;letter-spacing:1px;margin-bottom:4px;">🏆 FINAL</div>'
        f'{_mcard(*_fin)}'
        f'<div style="font-size:10px;font-weight:700;color:#ff6b35;margin-top:6px;">'
        f'Champion: {flag(_champ)} {_champ}</div>'
        f'<div style="font-size:8px;color:#555;">'
        f'{live_sim["champion"].get(_champ, 0)*100:.1f}% probability</div>'
        f'</div>'
    )

    _bracket = (
        f'<div style="display:flex;align-items:center;padding-bottom:4px;gap:0;">'
        + _lbl("R32") + _lbl_gap() + _lbl("R16") + _lbl_gap()
        + _lbl("QF") + _lbl_gap() + _lbl("SF") + _lbl_gap()
        + _lbl("FINAL") + _lbl_gap()
        + _lbl("SF") + _lbl_gap() + _lbl("QF") + _lbl_gap() + _lbl("R16") + _lbl_gap() + _lbl("R32")
        + '</div>'
        + '<div style="display:flex;align-items:stretch;gap:0;min-height:510px;">'
        + _rcol(_r32[0:8]) + _conn(4) + _rcol(_r16[0:4]) + _conn(2)
        + _rcol(_qf[0:2]) + _conn(1) + _rcol([_sf[0]], center=True)
        + _sf_conn
        + (f'<div style="display:flex;flex-direction:column;justify-content:center;'
           f'flex-shrink:0;width:{_CW + 16}px;padding:0 8px;">{_champion_banner}</div>')
        + _sf_conn
        + _rcol([_sf[1]], center=True) + _conn(1) + _rcol(_qf[2:4]) + _conn(2)
        + _rcol(_r16[4:8]) + _conn(4) + _rcol(_r32[8:16])
        + '</div>'
    )

    _min_w = (8 * (_CW + _GW) * 2) + _CW + 32
    st.markdown(
        f'<div style="overflow-x:auto;-webkit-overflow-scrolling:touch;padding-bottom:8px;">'
        f'<div style="min-width:{_min_w}px;">{_bracket}</div></div>',
        unsafe_allow_html=True,
    )
    st.caption("← Scroll to see all rounds · Orange = projected winner of each match")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Head-to-Head
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.header("Head-to-Head")

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Team A", sorted(ALL_TEAMS),
                              index=sorted(ALL_TEAMS).index("Brazil"),
                              format_func=lambda t: f"{flag(t)} {t}", key="h2h_a")
    with col2:
        team_b = st.selectbox("Team B", sorted(ALL_TEAMS),
                              index=sorted(ALL_TEAMS).index("Argentina"),
                              format_func=lambda t: f"{flag(t)} {t}", key="h2h_b")

    if team_a == team_b:
        st.warning("Select two different teams.")
    else:
        elo_a  = elos.get(team_a, 1500)
        elo_b  = elos.get(team_b, 1500)
        result = predict_match(elo_a, elo_b)
        win_a, draw, win_b = result["win"], result["draw"], result["loss"]

        st.subheader(f"{flag(team_a)} {team_a}  vs  {flag(team_b)} {team_b}")

        c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 2])
        c1.metric(f"{flag(team_a)} Win", fmt_pct(win_a))
        c2.metric("Draw", fmt_pct(draw))
        c3.metric(f"{flag(team_b)} Win", fmt_pct(win_b))
        c4.metric("Elo", f"{elo_a:.0f}")
        c5.metric("Elo", f"{elo_b:.0f}")

        fig_bar = go.Figure(go.Bar(
            x=[win_a * 100, draw * 100, win_b * 100],
            y=[f"{flag(team_a)} {team_a}", "Draw", f"{flag(team_b)} {team_b}"],
            orientation="h",
            marker_color=["#1f77b4", "#7f7f7f", "#d62728"],
            text=[fmt_pct(win_a), fmt_pct(draw), fmt_pct(win_b)],
            textposition="auto",
        ))
        fig_bar.update_layout(xaxis=dict(range=[0, 105]), height=200,
                              margin=dict(t=10, b=30), showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Score probability heatmap")
        mu_a, mu_b = result["mu_a"], result["mu_b"]
        max_g = 6
        score_matrix = np.array([
            [_poisson.pmf(i, mu_a) * _poisson.pmf(j, mu_b) for j in range(max_g + 1)]
            for i in range(max_g + 1)
        ]) * 100
        df_heat = pd.DataFrame(
            score_matrix,
            index=[f"{team_a[:3]} {i}" for i in range(max_g + 1)],
            columns=[f"{team_b[:3]} {j}" for j in range(max_g + 1)],
        )
        mls, mlg = result["most_likely_score"]
        fig_heat = px.imshow(df_heat, color_continuous_scale="Blues", labels={"color": "%"},
                             title=f"Most likely: {team_a} {mls}–{mlg} {team_b}", aspect="auto")
        fig_heat.update_layout(height=340, margin=dict(t=50, b=20))
        st.plotly_chart(fig_heat, use_container_width=True)

        st.subheader("Historical record")
        h2h = _load_h2h(team_a, team_b)
        if h2h.empty:
            st.info("No historical matches found between these teams.")
        else:
            name_a = TEAM_ALIASES.get(team_a, team_a)
            name_b = TEAM_ALIASES.get(team_b, team_b)
            wins_a = (
                ((h2h["home_team"] == name_a) & (h2h["home_score"] > h2h["away_score"])).sum()
                + ((h2h["away_team"] == name_a) & (h2h["away_score"] > h2h["home_score"])).sum()
            )
            wins_b = (
                ((h2h["home_team"] == name_b) & (h2h["home_score"] > h2h["away_score"])).sum()
                + ((h2h["away_team"] == name_b) & (h2h["away_score"] > h2h["home_score"])).sum()
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
    "Data: Kaggle results + eloratings.net + ESPN live"
)

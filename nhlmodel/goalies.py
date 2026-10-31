"""Goalie game logs from NHL box scores, and the goalie saves model.

Saves for a starting goalie are simulated period by period:
  shots against ~ over-dispersed Poisson around the team-shots model's expectation
  goals against ~ binomial(shots, 1 - goalie save %), save % shrunk toward league average
  a goalie who has allowed `pull_ga`+ goals after the 1st or 2nd period is pulled with prob `pull_p`
  overtime adds a few shots in games tied after regulation
"""
import re
import time
import unicodedata

import numpy as np
import pandas as pd

from .config import DATA_DIR, NHL_API
from .nhl_api import _get

GOALIE_CSV = DATA_DIR / "goalie_games.csv"
GCOLS = ["game_id", "date", "season", "team", "opp", "is_home", "player_id", "name", "starter",
         "toi_min", "shots_against", "saves", "goals_against"]

SAVES_DEFAULTS = {
    "sv_prior_shots": 1500,   # shots of league-average save% blended into each goalie's record
    "sv_half_life": 365,      # days; older shots count less
    "shots_disp": 25.0,       # gamma shape for game-to-game shot volume noise (lower = wider)
    "pull_ga": 4,             # goals allowed by end of a period that puts the goalie at risk of a pull
    "pull_p": 0.6,            # chance he is pulled once at that mark
    "ot_shots": 2.0,          # mean shots a goalie faces in OT when the game gets there
    "w_model": 0.35,          # model vs market blend for saves lines
    "goalie_metric": "sv",    # "sv" = shrunk save %, "gsax" = goals allowed vs MoneyPuck expected goals
    "gsax_prior_xg": 40.0,    # expected goals of league-average goaltending blended into each goalie (gsax mode)
    "goalie_b2b_pen": 0.03,   # extra goals-against rate when the goalie also played yesterday
                              # (measured +6% on 101 starts 2024-26; half applied because the sample is small)
}


def _toi(s):
    try:
        m, sec = str(s).split(":"); return int(m) + int(sec) / 60
    except Exception:
        return 0.0


def _num(g, key, alt_key=None, idx=None):
    v = g.get(key)
    if v is None and alt_key and g.get(alt_key):
        parts = re.findall(r"\d+", str(g[alt_key]))
        v = int(parts[idx]) if len(parts) > idx else None
    return v


def boxscore_goalies(game_id):
    js = _get(f"{NHL_API}/gamecenter/{game_id}/boxscore")
    if not js:
        return []
    rows = []
    teams = {"homeTeam": js.get("homeTeam", {}).get("abbrev"), "awayTeam": js.get("awayTeam", {}).get("abbrev")}
    pbg = js.get("playerByGameStats", {})
    for side, opp_side in (("homeTeam", "awayTeam"), ("awayTeam", "homeTeam")):
        gs = pbg.get(side, {}).get("goalies", [])
        tois = [_toi(g.get("toi")) for g in gs]
        for g, toi in zip(gs, tois):
            if toi <= 0:
                continue
            sa = _num(g, "shotsAgainst", "saveShotsAgainst", 1)
            sv = _num(g, "saves", "saveShotsAgainst", 0)
            starter = g.get("starter")
            if starter is None:
                starter = toi == max(tois)
            rows.append({"game_id": game_id, "team": teams[side], "opp": teams[opp_side],
                         "is_home": side == "homeTeam", "player_id": g.get("playerId"),
                         "name": (g.get("name") or {}).get("default"), "starter": bool(starter),
                         "toi_min": round(toi, 2), "shots_against": sa, "saves": sv,
                         "goals_against": g.get("goalsAgainst")})
    return rows


def update_goalies(games: pd.DataFrame, since="2024-09-01", verbose=True) -> pd.DataFrame:
    gl = pd.read_csv(GOALIE_CSV) if GOALIE_CSV.exists() else pd.DataFrame(columns=GCOLS)
    done = set(gl["game_id"].astype(int)) if len(gl) else set()
    todo = games[games["state"].isin(["OFF", "FINAL"]) & (games["date"] >= since) & ~games["game_id"].isin(done)]
    rows = []
    for _, g in todo.iterrows():
        for r in boxscore_goalies(int(g["game_id"])):
            r.update(date=g["date"], season=g["season"]); rows.append(r)
        time.sleep(0.1)
    if verbose:
        print(f"[goalies] fetched {len(todo)} box scores")
    if rows:
        gl = pd.concat([gl, pd.DataFrame(rows)[GCOLS]], ignore_index=True).sort_values(["date", "game_id"])
        gl.to_csv(GOALIE_CSV, index=False)
    return gl


# ---------- names ----------
def _norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", "", s.replace("-", " ")).strip()


def name_key(full_or_abbrev: str) -> str:
    """'Connor Hellebuyck' and 'C. Hellebuyck' -> 'c hellebuyck'."""
    n = _norm(full_or_abbrev).split()
    return f"{n[0][0]} {' '.join(n[1:])}" if len(n) >= 2 else (n[0] if n else "")


def find_goalie(gl: pd.DataFrame, full_name: str, teams: set, as_of: str):
    """Most recent appearance of this goalie before as_of; returns (player_id, team) or (None, None).
    Only accepts him if his latest team is one of the two teams in the game."""
    key = name_key(full_name)
    hist = gl[(gl["date"] < as_of)]
    hist = hist[hist["name"].map(name_key) == key]
    if hist.empty:
        return None, None
    last = hist.iloc[-1]
    if last["team"] not in teams:
        return int(last["player_id"]), None
    return int(last["player_id"]), last["team"]


# ---------- model ----------
def save_pct(gl: pd.DataFrame, player_id: int, as_of: str, p: dict) -> tuple[float, float, int]:
    """Effective save % for a goalie as of a date: (goalie, league, games in history).

    sv mode: shrunk raw save %.
    gsax mode: shrunk goals-allowed / expected-goals ratio from MoneyPuck (shot quality aware),
    turned back into a save % on league-average shot quality."""
    hist = gl[gl["date"] < as_of].dropna(subset=["shots_against", "saves"])
    ts = pd.Timestamp(as_of)
    w_all = 0.5 ** ((ts - pd.to_datetime(hist["date"])).dt.days / p["sv_half_life"])
    league = (hist["saves"] * w_all).sum() / (hist["shots_against"] * w_all).sum()
    mine = hist["player_id"] == player_id
    if p.get("goalie_metric", "sv") == "gsax" and "xga" in hist and hist["xga"].notna().mean() > 0.6:
        hx = hist.dropna(subset=["xga"]); wx = w_all[hx.index]; m = hx["player_id"] == player_id
        ga_l = (hx["goals_against"] * wx).sum(); xg_l = (hx["xga"] * wx).sum()
        r_l = ga_l / xg_l if xg_l else 1.0
        ga = (hx.loc[m, "goals_against"] * wx[m]).sum(); xg = (hx.loc[m, "xga"] * wx[m]).sum()
        k = p.get("gsax_prior_xg", 40.0)
        q = (ga + k * r_l) / (xg + k)                    # goalie's goals per expected goal
        return float(1 - (1 - league) * q / r_l), float(league), int(mine.sum())
    sa = (hist.loc[mine, "shots_against"] * w_all[mine]).sum()
    sv = (hist.loc[mine, "saves"] * w_all[mine]).sum()
    k = p["sv_prior_shots"]
    return float((sv + k * league) / (sa + k)), float(league), int(mine.sum())


def simulate_saves(mu_shots: float, sv: float, p_tie: float, p: dict, n=40000, seed=1) -> np.ndarray:
    """Distribution of saves (array of probabilities indexed by save count)."""
    rng = np.random.default_rng(seed)
    lam = mu_shots * rng.gamma(p["shots_disp"], 1 / p["shots_disp"], n)
    saves = np.zeros(n, int); ga = np.zeros(n, int); active = np.ones(n, bool)
    for per in range(3):
        sh = rng.poisson(lam / 3) * active
        g = rng.binomial(sh, 1 - sv)
        saves += sh - g; ga += g
        if per < 2:
            pulled = (ga >= p["pull_ga"]) & (rng.random(n) < p["pull_p"])
            active &= ~pulled
    ot = (rng.random(n) < p_tie) & active
    sh = rng.poisson(p["ot_shots"], n) * ot
    saves += sh - rng.binomial(sh, 1 - sv)
    return np.bincount(saves, minlength=80)[:80] / n


def over_under(dist: np.ndarray, line: float) -> tuple[float, float, float]:
    k = np.arange(len(dist))
    return float(dist[k > line].sum()), float(dist[k == line].sum()), float(dist[k < line].sum())

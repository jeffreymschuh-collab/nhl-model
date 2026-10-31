"""Skater layer: MoneyPuck game-by-game skater data -> player impact ratings, lineup (missing-player)
adjustments, and a shots-on-goal prop model. Data courtesy of MoneyPuck.com (personal, non-commercial use)."""
import io
import re
import unicodedata
import zipfile

import numpy as np
import pandas as pd
import requests

from .config import DATA_DIR
from .moneypuck import HEADERS, TEAM_FIX, _norm_id

SKATER_ZIP = "https://peter-tanner.com/moneypuck/downloads/seasonPlayersSummary/skaters/{season}.zip"
SK_CSV = DATA_DIR / "mp_skater_games.csv.gz"
COLS = ["playerId", "name", "gameId", "season", "playerTeam", "opposingTeam", "home_or_away", "gameDate",
        "position", "situation", "icetime", "I_F_shotsOnGoal", "I_F_xGoals", "OnIce_F_xGoals", "OnIce_A_xGoals"]

PRIOR_MIN = 400.0      # minutes of league-average play blended into each skater's on-ice rating
HALF_LIFE = 180        # days


def update_skater_games(seasons=(2023, 2024, 2025, 2026), current=2026, verbose=True) -> pd.DataFrame | None:
    have = pd.read_csv(SK_CSV) if SK_CSV.exists() else pd.DataFrame()
    done = set(have["season"].unique()) if len(have) else set()
    frames = [have[have["season"] != current]] if len(have) else []
    for s in seasons:
        if s in done and s != current:
            continue
        try:
            r = requests.get(SKATER_ZIP.format(season=s), timeout=180, headers=HEADERS)
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            d = pd.read_csv(z.open(z.namelist()[0]), usecols=lambda c: c in COLS, low_memory=False)
            d = d[d["situation"] == "all"].drop(columns=["situation"])
            for c in ("playerTeam", "opposingTeam"):
                d[c] = d[c].replace(TEAM_FIX)
            d["game_id"] = [_norm_id(g, y) for g, y in zip(d["gameId"], d["season"])]
            d["date"] = pd.to_datetime(d["gameDate"].astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")
            frames.append(d.drop(columns=["gameId", "gameDate"]))
            if verbose:
                print(f"[skaters] {s}: {len(d)} player-games")
        except Exception as err:
            print(f"[skaters] {s} failed: {err}")
            if s == current and len(have):
                frames.append(have[have["season"] == current])
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["game_id", "playerId"])
    out.to_csv(SK_CSV, index=False, compression="gzip")
    return out


# ------------------------------------------------------------------ player impact
def player_ratings(sk: pd.DataFrame, as_of: str) -> pd.DataFrame:
    """Per player as of a date: time-weighted on-ice xG for/against per 60 relative to league (shrunk),
    average TOI over his last 10 games, shots-on-goal per 60 (shrunk toward his position average)."""
    h = sk[sk["date"] < as_of]
    if h.empty:
        return pd.DataFrame()
    days = (pd.Timestamp(as_of) - pd.to_datetime(h["date"])).dt.days
    w = 0.5 ** (days / HALF_LIFE)
    mins = h["icetime"] / 60.0
    agg = pd.DataFrame({"playerId": h["playerId"], "position": h["position"], "w_min": w * mins,
                        "w_f": w * h["OnIce_F_xGoals"], "w_a": w * h["OnIce_A_xGoals"], "w_sog": w * h["I_F_shotsOnGoal"]})
    g = agg.groupby("playerId").agg(position=("position", "last"), w_min=("w_min", "sum"), w_f=("w_f", "sum"),
                                     w_a=("w_a", "sum"), w_sog=("w_sog", "sum"))
    lg_f = g["w_f"].sum() / g["w_min"].sum() * 60
    lg_a = g["w_a"].sum() / g["w_min"].sum() * 60
    g["xgf60"] = (g["w_f"] * 60 + PRIOR_MIN * lg_f) / (g["w_min"] + PRIOR_MIN)
    g["xga60"] = (g["w_a"] * 60 + PRIOR_MIN * lg_a) / (g["w_min"] + PRIOR_MIN)
    g["off_rel"] = g["xgf60"] - lg_f
    g["def_rel"] = g["xga60"] - lg_a                      # > 0 = worse defensively
    pos_sog = (g.groupby("position")["w_sog"].sum() / g.groupby("position")["w_min"].sum() * 60).to_dict()
    pm = g["position"].map(pos_sog).fillna(np.nanmean(list(pos_sog.values())))
    g["sog60"] = (g["w_sog"] * 60 + 150 * pm) / (g["w_min"] + 150)   # 150 minutes of position-average shooting
    last = h.sort_values("date").groupby("playerId").tail(10)
    g["toi_recent"] = last.groupby("playerId")["icetime"].mean() / 60.0
    g["team"] = h.sort_values("date").groupby("playerId")["playerTeam"].last()
    g["name"] = h.sort_values("date").groupby("playerId")["name"].last()
    return g.reset_index()


def regulars(sk: pd.DataFrame, team: str, before: str, last_n: int = 10, min_games: int = 6) -> pd.DataFrame:
    """Players who dressed in at least `min_games` of the team's last `last_n` games, with their average TOI."""
    since = (pd.Timestamp(before) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")   # no carry-over across the summer
    t = sk[(sk["playerTeam"] == team) & (sk["date"] < before) & (sk["date"] >= since)]
    gids = t.drop_duplicates("game_id").sort_values("date")["game_id"].tail(last_n)
    t = t[t["game_id"].isin(gids)]
    c = t.groupby("playerId").agg(games=("game_id", "nunique"), toi=("icetime", "mean"))
    return c[c["games"] >= min_games]


def missing_impact(ratings: pd.DataFrame, reg: pd.DataFrame, playing: set) -> tuple[float, float]:
    """(offense lost, defense lost) in on-ice xG per game from regulars who are not playing.
    Each missing player's on-ice edge is scaled by his share of a game (TOI/60) and assumed replaced
    by a league-average player. Offense lost > 0 means fewer goals for; defense lost > 0 means more against."""
    if ratings.empty or reg.empty:
        return 0.0, 0.0
    miss = [p for p in reg.index if p not in playing]
    if not miss:
        return 0.0, 0.0
    r = ratings.set_index("playerId").reindex(miss).dropna(subset=["off_rel"])
    share = (reg.loc[r.index, "toi"] / 3600.0)             # fraction of 60 minutes
    off = float((r["off_rel"] * share).sum())
    dfn = float((-r["def_rel"] * share).sum())              # losing a good defender (negative def_rel) hurts
    return off, dfn


def lineup_table(sk: pd.DataFrame, games: pd.DataFrame, min_date: str = "2023-11-01") -> pd.DataFrame:
    """Historical lineup covariates per (game_id, team): off_lost, def_lost, from who actually dressed.
    Ratings are refreshed weekly to keep this fast."""
    g = games[games["date"] >= min_date][["game_id", "date", "home", "away"]].sort_values("date")
    played = sk.groupby(["game_id", "playerTeam"])["playerId"].apply(set).to_dict()
    rows, cache = [], {}
    for _, x in g.iterrows():
        wk = pd.Timestamp(x["date"]).to_period("W").start_time.strftime("%Y-%m-%d")
        if wk not in cache:
            cache = {wk: player_ratings(sk, wk)}
        R = cache[wk]
        for t in (x["home"], x["away"]):
            pl = played.get((x["game_id"], t))
            if pl is None:
                continue
            off, dfn = missing_impact(R, regulars(sk, t, x["date"]), pl)
            rows.append({"game_id": x["game_id"], "team": t, "off_lost": off, "def_lost": dfn})
    return pd.DataFrame(rows).set_index(["game_id", "team"]) if rows else pd.DataFrame()


def add_lineup_covs(f: pd.DataFrame, lt: pd.DataFrame) -> pd.DataFrame:
    """Covariate 'lineup' for each attacking row: att = attacker's offense lost, def = defender's defense lost."""
    f = f.copy()
    get = lambda gid, t, c: (lt.loc[(gid, t), c] if (gid, t) in lt.index else 0.0)
    f["h_lineup_att"] = [get(i, t, "off_lost") for i, t in zip(f["game_id"], f["home"])]
    f["h_lineup_def"] = [get(i, t, "def_lost") for i, t in zip(f["game_id"], f["away"])]
    f["a_lineup_att"] = [get(i, t, "off_lost") for i, t in zip(f["game_id"], f["away"])]
    f["a_lineup_def"] = [get(i, t, "def_lost") for i, t in zip(f["game_id"], f["home"])]
    return f


# ------------------------------------------------------------------ names (Daily Faceoff <-> MoneyPuck)
def _norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", "", s.replace("-", " ")).strip()


def match_players(names: list[str], ratings: pd.DataFrame, team: str, cutoff: float = 0.85) -> dict:
    """{name: playerId} matching Daily Faceoff names to MoneyPuck players on that team.
    Exact normalized match first, then closest spelling (MoneyPuck drops accented letters: 'Slafkovsk')."""
    from difflib import SequenceMatcher
    pool = ratings[ratings["team"] == team]
    cand = [(pid, _norm(n)) for pid, n in zip(pool["playerId"], pool["name"]) if isinstance(n, str)]
    out, used = {}, set()
    for n in names:
        k = _norm(n)
        exact = [pid for pid, c in cand if c == k and pid not in used]
        if exact:
            out[n] = exact[0]; used.add(exact[0]); continue
        best = max(((SequenceMatcher(None, k, c).ratio(), pid) for pid, c in cand if pid not in used), default=(0, None))
        if best[0] >= cutoff:
            out[n] = best[1]; used.add(best[1])
    return out


# ------------------------------------------------------------------ shots-on-goal props
def sog_expectation(r: pd.Series, team_shots_factor: float, toi_min: float | None = None) -> float:
    """Expected shots on goal for one player tonight."""
    toi = toi_min if toi_min else r["toi_recent"]
    return float(r["sog60"] * toi / 60.0 * team_shots_factor)


def sog_dist(mu: float, disp: float = 8.0, maxk: int = 15) -> np.ndarray:
    """Negative binomial with mean mu and shape disp (players' shot counts are a bit over-dispersed)."""
    from scipy.stats import nbinom
    p = disp / (disp + mu)
    k = np.arange(maxk + 1)
    d = nbinom.pmf(k, disp, p)
    return d / d.sum()

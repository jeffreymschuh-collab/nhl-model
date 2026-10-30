"""Pulls game results and today's slate from the official NHL web API and keeps data/games.csv current."""
import time
from datetime import date, datetime, timedelta

import pandas as pd
import requests

from .config import FIRST_SEASON_START, GAMES_CSV, NHL_API

COLS = ["game_id", "season", "game_type", "date", "start_utc", "home", "away", "home_name", "away_name",
        "state", "last_period", "home_goals", "away_goals", "home_sog", "away_sog"]
FINAL_STATES = {"OFF", "FINAL"}

_session = requests.Session()
_session.headers["User-Agent"] = "nhl-model/1.0 (personal research)"


def _get(url, tries=4):
    for i in range(tries):
        try:
            r = _session.get(url, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"NHL API failed: {url}")


def _row(g):
    h, a = g.get("homeTeam", {}), g.get("awayTeam", {})
    pd_ = g.get("periodDescriptor") or {}
    return {
        "game_id": g.get("id"), "season": g.get("season"), "game_type": g.get("gameType"),
        "date": g.get("gameDate"), "start_utc": g.get("startTimeUTC"),
        "home": h.get("abbrev"), "away": a.get("abbrev"),
        "home_name": (h.get("name") or {}).get("default"), "away_name": (a.get("name") or {}).get("default"),
        "state": g.get("gameState"), "last_period": pd_.get("periodType"),
        "home_goals": h.get("score"), "away_goals": a.get("score"),
        "home_sog": h.get("sog"), "away_sog": a.get("sog"),
    }


def games_on(day: str) -> list[dict]:
    js = _get(f"{NHL_API}/score/{day}") or {}
    return [_row(g) for g in js.get("games", []) if g.get("gameType") in (2, 3)]


def load_games() -> pd.DataFrame:
    if GAMES_CSV.exists():
        return pd.read_csv(GAMES_CSV)
    return pd.DataFrame(columns=COLS)


def update_games(through: date | None = None, verbose=True) -> pd.DataFrame:
    """Fetch any missing days (and re-fetch unfinished games) up to `through` (default: yesterday)."""
    through = through or (date.today() - timedelta(days=1))
    df = load_games()
    if len(df):
        unfinished = df[~df["state"].isin(FINAL_STATES)]["date"]
        start = min([df["date"].max()] + list(unfinished))
        start = datetime.strptime(start, "%Y-%m-%d").date()
    else:
        start = datetime.strptime(FIRST_SEASON_START, "%Y-%m-%d").date()

    rows, d, n = [], start, 0
    while d <= through:
        # skip the deep offseason (Jul-Sep) to save calls
        if d.month not in (7, 8, 9):
            rows += games_on(d.isoformat())
            n += 1
            time.sleep(0.15)
        d += timedelta(days=1)
    if verbose:
        print(f"[nhl_api] fetched {n} days, {len(rows)} games from {start} to {through}")
    if rows:
        new = pd.DataFrame(rows, columns=COLS)
        df = pd.concat([df[~df["game_id"].isin(new["game_id"])], new], ignore_index=True)
        df = df.sort_values(["date", "game_id"]).reset_index(drop=True)
        GAMES_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(GAMES_CSV, index=False)
    return df


def finals(df: pd.DataFrame) -> pd.DataFrame:
    """Completed games with regulation-time goals (OT/SO winner's extra goal removed)."""
    f = df[df["state"].isin(FINAL_STATES)].dropna(subset=["home_goals", "away_goals"]).copy()
    f["home_goals"] = f["home_goals"].astype(int)
    f["away_goals"] = f["away_goals"].astype(int)
    extra = f["last_period"].isin(["OT", "SO"])
    home_won = f["home_goals"] > f["away_goals"]
    f["home_reg"] = f["home_goals"] - (extra & home_won).astype(int)
    f["away_reg"] = f["away_goals"] - (extra & ~home_won).astype(int)
    return f

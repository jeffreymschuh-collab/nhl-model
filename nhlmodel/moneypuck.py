"""MoneyPuck team game-by-game expected goals (data courtesy of MoneyPuck.com, personal non-commercial use).

Keeps a small extract in data/mp_team_games.csv: one row per team-game, all situations,
with score-and-venue-adjusted expected goals for and against.
"""
import io
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from .config import DATA_DIR

URL = "https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv"
MP_CSV = DATA_DIR / "mp_team_games.csv"
STAMP = DATA_DIR / "mp_fetched_at.txt"
STATUS = DATA_DIR / "mp_status.txt"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36 (personal research)",
           "Accept": "text/csv,*/*"}
TEAM_FIX = {"T.B": "TBL", "N.J": "NJD", "L.A": "LAK", "S.J": "SJS"}
KEEP = ["gameId", "playerTeam", "opposingTeam", "home_or_away", "gameDate", "season", "situation", "iceTime",
        "xGoalsFor", "xGoalsAgainst", "scoreVenueAdjustedxGoalsFor", "scoreVenueAdjustedxGoalsAgainst",
        "shotsOnGoalFor", "shotsOnGoalAgainst", "goalsFor", "goalsAgainst"]


def _norm_id(gid, season):
    gid = int(gid)
    if gid < 10 ** 9:                      # short ids like 20001 -> 2023020001
        gid = int(season) * 10 ** 6 + gid
    return gid


def update(min_season=2023, max_age_hours=18, verbose=True) -> pd.DataFrame | None:
    """Refresh the extract if older than max_age_hours. Returns the extract, or None if unavailable."""
    fresh = False
    if STAMP.exists() and MP_CSV.exists():
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(STAMP.read_text().strip())).total_seconds() / 3600
        fresh = age < max_age_hours
    if not fresh:
        try:
            r = None
            for i in range(3):
                r = requests.get(URL, timeout=180, headers=HEADERS)
                if r.status_code == 200:
                    break
                time.sleep(5 * (i + 1))
            r.raise_for_status()
            df = parse(r.content, min_season)
            MP_CSV.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(MP_CSV, index=False)
            STAMP.write_text(datetime.now(timezone.utc).isoformat())
            STATUS.write_text(f"ok {datetime.now(timezone.utc).isoformat()} rows={len(df)}\n")
            if verbose:
                print(f"[moneypuck] {len(df)} team-games (all situations) since {min_season}")
        except Exception as err:
            msg = f"failed {datetime.now(timezone.utc).isoformat()} {type(err).__name__}: {err}"
            if r is not None:
                msg += f" | http {r.status_code} bytes {len(r.content)} start {r.content[:120]!r}"
            STATUS.parent.mkdir(parents=True, exist_ok=True)
            STATUS.write_text(msg + "\n")
            print(f"[moneypuck] {msg}; using cached extract if any")
    return pd.read_csv(MP_CSV) if MP_CSV.exists() else None


SITUATIONS = ("all", "5on4", "4on5")


def parse(content: bytes, min_season=2023) -> pd.DataFrame:
    """One row per team-game: all-situations columns as-is, plus power-play (5on4) and penalty-kill (4on5)
    ice time and expected goals, suffixed _pp / _pk."""
    df = pd.read_csv(io.BytesIO(content), usecols=lambda c: c in KEEP, low_memory=False)
    df = df[(pd.to_numeric(df["season"], errors="coerce") >= min_season) & df["situation"].isin(SITUATIONS)].copy()
    for c in ("playerTeam", "opposingTeam"):
        df[c] = df[c].replace(TEAM_FIX)
    df["game_id"] = [_norm_id(g, s) for g, s in zip(df["gameId"], df["season"])]
    df = df.drop(columns=["gameId"]).drop_duplicates(["game_id", "playerTeam", "situation"])
    base = df[df["situation"] == "all"].drop(columns=["situation"])
    for sit, sfx in (("5on4", "_pp"), ("4on5", "_pk")):
        part = df[df["situation"] == sit][["game_id", "playerTeam", "iceTime", "xGoalsFor", "xGoalsAgainst"]]
        part = part.rename(columns={c: c + sfx for c in ("iceTime", "xGoalsFor", "xGoalsAgainst")})
        base = base.merge(part, on=["game_id", "playerTeam"], how="left")
    return base


GOALIE_ZIP = "https://peter-tanner.com/moneypuck/downloads/seasonPlayersSummary/goalies/{season}.zip"
MP_GOALIE_CSV = DATA_DIR / "mp_goalie_games.csv"


def update_goalie_games(seasons=(2023, 2024, 2025, 2026), current=2026, verbose=True) -> pd.DataFrame | None:
    """Per-goalie, per-game expected goals against (all situations) from MoneyPuck's game-by-game goalie files.
    Past seasons are fetched once; the current season refreshes every call."""
    import zipfile
    have = pd.read_csv(MP_GOALIE_CSV) if MP_GOALIE_CSV.exists() else pd.DataFrame()
    done = set(have["season"].unique()) if len(have) else set()
    frames = [have[have["season"] != current]] if len(have) else []
    for s in seasons:
        if s in done and s != current:
            continue
        try:
            r = requests.get(GOALIE_ZIP.format(season=s), timeout=120, headers=HEADERS)
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            g = pd.read_csv(z.open(z.namelist()[0]), usecols=lambda c: c in (
                "playerId", "name", "gameId", "season", "playerTeam", "situation", "icetime", "xGoals", "goals", "ongoal"))
            g = g[g["situation"] == "all"].drop(columns=["situation"])
            g["playerTeam"] = g["playerTeam"].replace(TEAM_FIX)
            g["game_id"] = [_norm_id(x, y) for x, y in zip(g["gameId"], g["season"])]
            frames.append(g.drop(columns=["gameId"]))
            if verbose:
                print(f"[moneypuck] goalie games {s}: {len(g)}")
        except Exception as err:
            print(f"[moneypuck] goalie games {s} failed: {err}")
            if s == current and len(have):
                frames.append(have[have["season"] == current])
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["game_id", "playerId"])
    out.to_csv(MP_GOALIE_CSV, index=False)
    return out


def attach_xg(finals_df: pd.DataFrame, mp: pd.DataFrame | None) -> pd.DataFrame:
    """Adds home_xg / away_xg (score & venue adjusted, all situations) to completed games."""
    f = finals_df.copy()
    if mp is None or mp.empty:
        f["home_xg"] = float("nan"); f["away_xg"] = float("nan")
        return f
    x = mp.set_index(["game_id", "playerTeam"])["scoreVenueAdjustedxGoalsFor"]
    f["home_xg"] = [x.get((g, t)) for g, t in zip(f["game_id"], f["home"])]
    f["away_xg"] = [x.get((g, t)) for g, t in zip(f["game_id"], f["away"])]
    return f


def attach_goalie_xga(gl: pd.DataFrame, mp: pd.DataFrame | None, mpg: pd.DataFrame | None = None) -> pd.DataFrame:
    """Adds xga to goalie games: MoneyPuck's own per-goalie expected goals against when available,
    otherwise the team's expected goals against split by the goalie's time on ice."""
    g = gl.copy()
    g["xga"] = float("nan")
    if mpg is not None and len(mpg):
        x = mpg.set_index(["game_id", "playerId"])["xGoals"]
        g["xga"] = [x.get((gid, int(p))) if pd.notna(p) else None for gid, p in zip(g["game_id"], g["player_id"])]
        g["xga"] = pd.to_numeric(g["xga"], errors="coerce")
    if mp is not None and len(mp):
        x = mp.set_index(["game_id", "playerTeam"])["xGoalsAgainst"]
        team_xga = pd.Series([x.get((gid, t)) for gid, t in zip(g["game_id"], g["team"])], index=g.index, dtype=float)
        tot_toi = g.groupby(["game_id", "team"])["toi_min"].transform("sum").replace(0, float("nan"))
        g["xga"] = g["xga"].fillna(team_xga * g["toi_min"] / tot_toi)
    return g

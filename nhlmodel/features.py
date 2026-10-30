"""Game context the team ratings don't see: rest (back-to-backs) and who is in net."""
import numpy as np
import pandas as pd

from . import goalies


def add_rest(games: pd.DataFrame) -> pd.DataFrame:
    """Adds home_rest/away_rest (days since that team's previous game, capped at 4) and home_b2b/away_b2b."""
    g = games.copy()
    long = pd.concat([
        g[["game_id", "date", "home"]].rename(columns={"home": "team"}),
        g[["game_id", "date", "away"]].rename(columns={"away": "team"}),
    ]).sort_values(["team", "date", "game_id"])
    long["d"] = pd.to_datetime(long["date"])
    long["rest"] = long.groupby("team")["d"].diff().dt.days.clip(upper=4).fillna(4)
    rest = long.set_index(["game_id", "team"])["rest"]
    g["home_rest"] = [rest.get((gid, t), 4) for gid, t in zip(g["game_id"], g["home"])]
    g["away_rest"] = [rest.get((gid, t), 4) for gid, t in zip(g["game_id"], g["away"])]
    g["home_b2b"] = (g["home_rest"] <= 1).astype(int)
    g["away_b2b"] = (g["away_rest"] <= 1).astype(int)
    return g


def rest_today(games: pd.DataFrame, team: str, day: str) -> int:
    """1 if `team` played yesterday (back-to-back today), else 0."""
    played = games[((games["home"] == team) | (games["away"] == team)) & (games["date"] < day)]["date"]
    if played.empty:
        return 0
    return int((pd.Timestamp(day) - pd.Timestamp(played.max())).days <= 1)


class GoalieContext:
    """Starter-vs-team-baseline goaltending ratio, cached by (goalie, team, date).

    ratio = (1 - starter save%) / (1 - team's usual save%), using shrunk save percentages.
    > 1 means a weaker-than-usual goalie (opponent scores more). The model applies ratio ** goalie_k
    to the opponent's expected goals."""

    def __init__(self, gl: pd.DataFrame, sp: dict, window_days: int = 120):
        self.gl = gl.dropna(subset=["shots_against", "saves"]).copy()
        self.gl["d"] = pd.to_datetime(self.gl["date"])
        self.sp, self.window = sp, window_days
        self._sv, self._team = {}, {}

    def sv(self, pid: int, day: str) -> float:
        k = (pid, day)
        if k not in self._sv:
            self._sv[k] = goalies.save_pct(self.gl, pid, day, self.sp)[0]
        return self._sv[k]

    def team_sv(self, team: str, day: str) -> float | None:
        k = (team, day)
        if k not in self._team:
            ts = pd.Timestamp(day)
            h = self.gl[(self.gl["team"] == team) & (self.gl["d"] < ts) & (self.gl["d"] >= ts - pd.Timedelta(days=self.window))]
            if h.empty:
                self._team[k] = None
            else:
                shots = h.groupby("player_id")["shots_against"].sum()
                self._team[k] = float(sum(s * self.sv(int(p), day) for p, s in shots.items()) / shots.sum())
        return self._team[k]

    def ratio(self, pid: int | None, team: str, day: str) -> float:
        if pid is None:
            return 1.0
        base = self.team_sv(team, day)
        if base is None:
            return 1.0
        return float(np.clip((1 - self.sv(int(pid), day)) / (1 - base), 0.7, 1.4))

    def projected_starter(self, team: str, day: str, last_n: int = 10) -> int | None:
        """The goalie with the most starts in the team's last `last_n` games (morning-run fallback)."""
        h = self.gl[(self.gl["team"] == team) & (self.gl["date"] < day) & self.gl["starter"].astype(bool)]
        h = h.sort_values("date").tail(last_n)
        if h.empty:
            return None
        return int(h["player_id"].value_counts().idxmax())

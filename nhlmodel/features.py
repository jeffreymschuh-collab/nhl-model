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


# ---------------------------------------------------------------------------------------------
# Schedule fatigue and special teams (model v3 candidates)

ARENAS = {  # approximate arena coordinates (lat, lon)
    "ANA": (33.808, -117.877), "UTA": (40.768, -111.901), "ARI": (33.532, -112.261), "BOS": (42.366, -71.062),
    "BUF": (42.875, -78.876), "CGY": (51.037, -114.052), "CAR": (35.803, -78.722), "CHI": (41.881, -87.674),
    "COL": (39.749, -105.008), "CBJ": (39.969, -83.006), "DAL": (32.791, -96.810), "DET": (42.341, -83.055),
    "EDM": (53.547, -113.498), "FLA": (26.158, -80.326), "LAK": (34.043, -118.267), "MIN": (44.945, -93.101),
    "MTL": (45.496, -73.569), "NSH": (36.159, -86.779), "NJD": (40.733, -74.171), "NYI": (40.723, -73.591),
    "NYR": (40.751, -73.993), "OTT": (45.297, -75.927), "PHI": (39.901, -75.172), "PIT": (40.440, -79.989),
    "SJS": (37.333, -121.901), "SEA": (47.622, -122.354), "STL": (38.627, -90.203), "TBL": (27.943, -82.452),
    "TOR": (43.644, -79.379), "VAN": (49.278, -123.109), "VGK": (36.103, -115.178), "WSH": (38.898, -77.021),
    "WPG": (49.893, -97.144),
}
COV_NAMES = ["b2b", "3in4", "travel", "st"]


def _km(a, b):
    import math
    if a not in ARENAS or b not in ARENAS:
        return 0.0
    (la1, lo1), (la2, lo2) = ARENAS[a], ARENAS[b]
    p1, p2 = math.radians(la1), math.radians(la2)
    d = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(d))


def team_log(games: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game with venue, sorted by team/date."""
    g = games[["game_id", "date", "home", "away"]]
    long = pd.concat([g.assign(team=g["home"], venue=g["home"]), g.assign(team=g["away"], venue=g["home"])])
    long = long.sort_values(["team", "date", "game_id"]).reset_index(drop=True)
    long["d"] = pd.to_datetime(long["date"])
    return long[["game_id", "date", "d", "team", "venue"]]


def fatigue_table(games: pd.DataFrame) -> pd.DataFrame:
    """Per team-game: 3in4 (third game in four nights) and travel since the previous game (1000 km)."""
    t = team_log(games)
    prev_venue = t.groupby("team")["venue"].shift(1)
    prev2_d = t.groupby("team")["d"].shift(2)
    t["travel"] = [(_km(a, b) / 1000.0) if isinstance(a, str) else 0.0 for a, b in zip(prev_venue, t["venue"])]
    t["3in4"] = ((t["d"] - prev2_d).dt.days <= 3).astype(int)
    return t.set_index(["game_id", "team"])[["travel", "3in4"]]


def special_teams_table(mp: pd.DataFrame | None, window: int = 25) -> pd.DataFrame | None:
    """Per team-game, trailing (previous `window` games, this game excluded) special-teams indices relative to league:
    pp_rate (PP xG per minute), pk_rate (PK xG allowed per minute), pp_time and pk_time (minutes per game)."""
    if mp is None or "iceTime_pp" not in mp:
        return None
    m = mp.sort_values(["playerTeam", "gameDate", "game_id"]).copy()
    for c in ("iceTime_pp", "xGoalsFor_pp", "iceTime_pk", "xGoalsAgainst_pk"):
        m[c] = m[c].fillna(0.0)
    grp = m.groupby("playerTeam")
    roll = lambda c: grp[c].transform(lambda s: s.shift(1).rolling(window, min_periods=5).sum())
    n = grp["game_id"].transform(lambda s: s.shift(1).rolling(window, min_periods=5).count())
    pp_t, pp_x, pk_t, pk_x = roll("iceTime_pp"), roll("xGoalsFor_pp"), roll("iceTime_pk"), roll("xGoalsAgainst_pk")
    lg_pp = m["xGoalsFor_pp"].sum() / m["iceTime_pp"].sum()
    lg_pk = m["xGoalsAgainst_pk"].sum() / m["iceTime_pk"].sum()
    lg_ppt = m["iceTime_pp"].mean(); lg_pkt = m["iceTime_pk"].mean()
    out = pd.DataFrame({
        "game_id": m["game_id"], "team": m["playerTeam"],
        "pp_rate": (pp_x / pp_t) / lg_pp, "pk_rate": (pk_x / pk_t) / lg_pk,
        "pp_time": (pp_t / n) / lg_ppt, "pk_time": (pk_t / n) / lg_pkt,
    }).set_index(["game_id", "team"])
    return out


def st_covs(st_att: dict | None, st_def: dict | None) -> tuple[float, float]:
    """Covariates for one attacking row: (attacker's PP threat, defender's PK leakiness), each scaled by how much
    PP time the matchup should produce (attacker draws, defender takes). 0 when unknown."""
    if not st_att or not st_def or any(pd.isna(v) for v in list(st_att.values()) + list(st_def.values())):
        return 0.0, 0.0
    time_factor = (st_att["pp_time"] + st_def["pk_time"]) / 2
    return float(np.log(st_att["pp_rate"]) * time_factor), float(np.log(st_def["pk_rate"]) * time_factor)


def add_context(games: pd.DataFrame, mp: pd.DataFrame | None = None) -> pd.DataFrame:
    """All per-game covariates for both attacking rows. Columns: h_<cov>_att, h_<cov>_def (home attacking)
    and a_<cov>_att, a_<cov>_def (away attacking), for cov in COV_NAMES."""
    g = add_rest(games)
    fat = fatigue_table(games)
    st = special_teams_table(mp)
    get = lambda tbl, gid, t, c: (tbl.loc[(gid, t), c] if (gid, t) in tbl.index else 0.0)
    for side, me, opp in (("h", "home", "away"), ("a", "away", "home")):
        g[f"{side}_b2b_att"] = g[f"{me}_b2b"]; g[f"{side}_b2b_def"] = g[f"{opp}_b2b"]
        g[f"{side}_3in4_att"] = [get(fat, i, t, "3in4") for i, t in zip(g["game_id"], g[me])]
        g[f"{side}_3in4_def"] = [get(fat, i, t, "3in4") for i, t in zip(g["game_id"], g[opp])]
        g[f"{side}_travel_att"] = [get(fat, i, t, "travel") for i, t in zip(g["game_id"], g[me])]
        g[f"{side}_travel_def"] = [get(fat, i, t, "travel") for i, t in zip(g["game_id"], g[opp])]
        if st is not None:
            vals = []
            for i, a, d in zip(g["game_id"], g[me], g[opp]):
                sa = st.loc[(i, a)].to_dict() if (i, a) in st.index else None
                sd = st.loc[(i, d)].to_dict() if (i, d) in st.index else None
                vals.append(st_covs(sa, sd))
            g[f"{side}_st_att"] = [v[0] for v in vals]; g[f"{side}_st_def"] = [v[1] for v in vals]
        else:
            g[f"{side}_st_att"] = 0.0; g[f"{side}_st_def"] = 0.0
    return g


def context_today(games: pd.DataFrame, mp: pd.DataFrame | None, home: str, away: str, day: str) -> dict:
    """The same covariates for a game being played today (not yet in games.csv)."""
    fake = pd.DataFrame([{"game_id": -1, "date": day, "home": home, "away": away}])
    hist = games[["game_id", "date", "home", "away"]]
    g = add_context(pd.concat([hist, fake], ignore_index=True), None)
    row = g[g["game_id"] == -1].iloc[0].to_dict()
    st = special_teams_table(mp)
    if st is not None:
        last = lambda t: st.xs(t, level="team").iloc[-1].to_dict() if t in st.index.get_level_values("team") else None
        # trailing values as of the team's latest game, which already exclude that game; close enough for today
        for side, me, opp in (("h", home, away), ("a", away, home)):
            row[f"{side}_st_att"], row[f"{side}_st_def"] = st_covs(last(me), last(opp))
    return row

"""Team attack/defense ratings via time-weighted ridge Poisson regression.

Two fits are made — one on regulation goals, one on a shot-based expected-goals proxy
(shots on goal x league shooting %) — and blended in log space. Shots are far less noisy
than goals early in a season; goals capture finishing and goaltending.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor


@dataclass
class Ratings:
    teams: list
    coef_g: np.ndarray
    icpt_g: float
    coef_s: np.ndarray | None
    icpt_s: float | None
    w_shots: float
    sh_pct: float | None = None
    rest: bool = False
    coef_x: np.ndarray | None = None
    icpt_x: float | None = None
    w_xg: float = 0.0

    def expected_shots(self, shooter: str, defender: str, shooter_home: bool, att_b2b: int = 0, def_b2b: int = 0) -> float:
        """Shots on goal `shooter` is expected to take against `defender`."""
        if self.coef_s is None or not self.sh_pct:
            raise ValueError("shots model not fitted")
        return self._lam(self.coef_s, self.icpt_s, shooter, defender, int(shooter_home), att_b2b, def_b2b) / self.sh_pct

    def _lam(self, coef, icpt, att, dfn, home, att_b2b=0, def_b2b=0):
        n = len(self.teams)
        ia, idf = self.teams.index(att), self.teams.index(dfn)
        z = icpt + coef[ia] + coef[n + idf] + coef[2 * n] * home
        if self.rest and len(coef) > 2 * n + 1:
            z += coef[2 * n + 1] * att_b2b + coef[2 * n + 2] * def_b2b
        return float(np.exp(z))

    def b2b_effects(self) -> dict:
        n = len(self.teams)
        if not self.rest or len(self.coef_g) <= 2 * n + 1:
            return {}
        return {"own_goals_when_b2b": round(float(np.exp(self.coef_g[2 * n + 1])), 3),
                "goals_allowed_when_b2b": round(float(np.exp(self.coef_g[2 * n + 2])), 3)}

    def components(self, home: str, away: str, home_b2b: int = 0, away_b2b: int = 0) -> dict:
        """Expected goals from each sub-model: goals ('g'), shots ('s'), MoneyPuck xG ('x'), as (home, away)."""
        for t in (home, away):
            if t not in self.teams:
                raise KeyError(f"no rating for {t}")
        out = {"g": (self._lam(self.coef_g, self.icpt_g, home, away, 1, home_b2b, away_b2b),
                     self._lam(self.coef_g, self.icpt_g, away, home, 0, away_b2b, home_b2b))}
        if self.coef_s is not None:
            out["s"] = (self._lam(self.coef_s, self.icpt_s, home, away, 1, home_b2b, away_b2b),
                        self._lam(self.coef_s, self.icpt_s, away, home, 0, away_b2b, home_b2b))
        if self.coef_x is not None:
            out["x"] = (self._lam(self.coef_x, self.icpt_x, home, away, 1, home_b2b, away_b2b),
                        self._lam(self.coef_x, self.icpt_x, away, home, 0, away_b2b, home_b2b))
        return out

    def expected_goals(self, home: str, away: str, home_b2b: int = 0, away_b2b: int = 0) -> tuple[float, float]:
        c = self.components(home, away, home_b2b, away_b2b)
        ws = self.w_shots if "s" in c else 0.0
        wx = self.w_xg if "x" in c else 0.0
        wg = 1.0 - ws - wx
        res = []
        for i in (0, 1):
            z = wg * np.log(c["g"][i])
            if ws:
                z += ws * np.log(c["s"][i])
            if wx:
                z += wx * np.log(c["x"][i])
            res.append(float(np.exp(z)))
        return res[0], res[1]

    def table(self) -> pd.DataFrame:
        """Per-team expected goals for/against vs a league-average opponent at a neutral site."""
        rows = []
        for t in self.teams:
            gf = np.mean([self.expected_goals(t, o)[0] for o in self.teams if o != t])
            ga = np.mean([self.expected_goals(t, o)[1] for o in self.teams if o != t])
            rows.append({"team": t, "xgf": round(gf, 3), "xga": round(ga, 3), "net": round(gf - ga, 3)})
        return pd.DataFrame(rows).sort_values("net", ascending=False).reset_index(drop=True)


def _design(f: pd.DataFrame, teams: list, rest: bool = False):
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    m = len(f)
    X = np.zeros((2 * m, 2 * n + 1 + (2 if rest else 0)))
    r = np.arange(m)
    h = f["home"].map(idx).to_numpy()
    a = f["away"].map(idx).to_numpy()
    # rows 0..m-1: home attacking; rows m..2m-1: away attacking
    X[r, h] = 1; X[r, n + a] = 1; X[r, 2 * n] = 1
    X[m + r, a] = 1; X[m + r, n + h] = 1
    if rest:
        hb = f["home_b2b"].fillna(0).to_numpy() if "home_b2b" in f else np.zeros(m)
        ab = f["away_b2b"].fillna(0).to_numpy() if "away_b2b" in f else np.zeros(m)
        X[r, 2 * n + 1] = hb; X[r, 2 * n + 2] = ab          # home attacking: own b2b, defender b2b
        X[m + r, 2 * n + 1] = ab; X[m + r, 2 * n + 2] = hb
    return X


def fit(finals_df: pd.DataFrame, as_of: str, params: dict) -> Ratings:
    """Fit on completed games strictly before `as_of` (YYYY-MM-DD)."""
    f = finals_df[finals_df["date"] < as_of].copy()
    if len(f) < 50:
        raise ValueError("not enough games to fit ratings")
    as_of_ts = pd.Timestamp(as_of)
    days = (as_of_ts - pd.to_datetime(f["date"])).dt.days.to_numpy()
    w = 0.5 ** (days / params["half_life_days"])
    cur_season = f["season"].max()
    w = w * np.where(f["season"].to_numpy() == cur_season, 1.0, params["season_carry"])
    keep = w > 1e-4
    f, w = f[keep], w[keep]

    # Teams present in the most recent season only (handles relocations/new teams)
    teams = sorted(set(f.loc[f["season"] == cur_season, "home"]) | set(f.loc[f["season"] == cur_season, "away"]))
    f_mask = f["home"].isin(teams) & f["away"].isin(teams)
    f, w = f[f_mask], w[f_mask]
    rest = bool(params.get("rest", False)) and "home_b2b" in f
    X = _design(f, teams, rest)
    W = np.concatenate([w, w])
    W = W / W.mean()

    y_g = np.concatenate([f["home_reg"].to_numpy(), f["away_reg"].to_numpy()]).astype(float)
    mg = PoissonRegressor(alpha=params["alpha"], max_iter=500).fit(X, y_g, sample_weight=W)

    coef_s = icpt_s = sh_pct = None
    s = f.dropna(subset=["home_sog", "away_sog"])
    if params["w_shots"] > 0 and len(s) > 0.8 * len(f):
        sw = W[: len(f)][f.index.isin(s.index)]
        sh_pct = (f["home_goals"].sum() + f["away_goals"].sum()) / (f["home_sog"].sum() + f["away_sog"].sum())
        y_s = np.concatenate([s["home_sog"].to_numpy(), s["away_sog"].to_numpy()]).astype(float) * sh_pct
        ms = PoissonRegressor(alpha=params["alpha"], max_iter=500).fit(
            _design(s, teams, rest), y_s, sample_weight=np.concatenate([sw, sw]))
        coef_s, icpt_s = ms.coef_, ms.intercept_
    coef_x = icpt_x = None
    w_xg = float(params.get("w_xg", 0.0))
    if w_xg > 0 and "home_xg" in f:
        x = f.dropna(subset=["home_xg", "away_xg"])
        if len(x) > 0.6 * len(f):
            xw = W[: len(f)][f.index.isin(x.index)]
            y_x = np.concatenate([x["home_xg"].to_numpy(), x["away_xg"].to_numpy()]).astype(float)
            mx = PoissonRegressor(alpha=params["alpha"], max_iter=500).fit(
                _design(x, teams, rest), y_x, sample_weight=np.concatenate([xw, xw]))
            coef_x, icpt_x = mx.coef_, mx.intercept_
    return Ratings(teams, mg.coef_, mg.intercept_, coef_s, icpt_s,
                   params["w_shots"] if coef_s is not None else 0.0, sh_pct, rest,
                   coef_x, icpt_x, w_xg if coef_x is not None else 0.0)

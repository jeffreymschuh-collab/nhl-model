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

    def expected_shots(self, shooter: str, defender: str, shooter_home: bool) -> float:
        """Shots on goal `shooter` is expected to take against `defender`."""
        if self.coef_s is None or not self.sh_pct:
            raise ValueError("shots model not fitted")
        return self._lam(self.coef_s, self.icpt_s, shooter, defender, int(shooter_home)) / self.sh_pct

    def _lam(self, coef, icpt, att, dfn, home):
        n = len(self.teams)
        ia, idf = self.teams.index(att), self.teams.index(dfn)
        return float(np.exp(icpt + coef[ia] + coef[n + idf] + coef[2 * n] * home))

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        for t in (home, away):
            if t not in self.teams:
                raise KeyError(f"no rating for {t}")
        lh_g = self._lam(self.coef_g, self.icpt_g, home, away, 1)
        la_g = self._lam(self.coef_g, self.icpt_g, away, home, 0)
        if self.coef_s is None or self.w_shots == 0:
            return lh_g, la_g
        lh_s = self._lam(self.coef_s, self.icpt_s, home, away, 1)
        la_s = self._lam(self.coef_s, self.icpt_s, away, home, 0)
        w = self.w_shots
        return float(np.exp((1 - w) * np.log(lh_g) + w * np.log(lh_s))), \
            float(np.exp((1 - w) * np.log(la_g) + w * np.log(la_s)))

    def table(self) -> pd.DataFrame:
        """Per-team expected goals for/against vs a league-average opponent at a neutral site."""
        rows = []
        for t in self.teams:
            gf = np.mean([self.expected_goals(t, o)[0] for o in self.teams if o != t])
            ga = np.mean([self.expected_goals(t, o)[1] for o in self.teams if o != t])
            rows.append({"team": t, "xgf": round(gf, 3), "xga": round(ga, 3), "net": round(gf - ga, 3)})
        return pd.DataFrame(rows).sort_values("net", ascending=False).reset_index(drop=True)


def _design(f: pd.DataFrame, teams: list):
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    m = len(f)
    X = np.zeros((2 * m, 2 * n + 1))
    r = np.arange(m)
    h = f["home"].map(idx).to_numpy()
    a = f["away"].map(idx).to_numpy()
    # rows 0..m-1: home attacking; rows m..2m-1: away attacking
    X[r, h] = 1; X[r, n + a] = 1; X[r, 2 * n] = 1
    X[m + r, a] = 1; X[m + r, n + h] = 1
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
    X = _design(f, teams)
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
            _design(s, teams), y_s, sample_weight=np.concatenate([sw, sw]))
        coef_s, icpt_s = ms.coef_, ms.intercept_
    return Ratings(teams, mg.coef_, mg.intercept_, coef_s, icpt_s,
                   params["w_shots"] if coef_s is not None else 0.0, sh_pct)

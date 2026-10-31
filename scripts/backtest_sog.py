"""Backtest the player shots-on-goal model on the second half of 2025-26 (no odds needed).
Baseline: each player's own last-10-game average shots on goal (Poisson). Writes results/backtest_sog.json."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nhlmodel import features, moneypuck, nhl_api, ratings, skaters  # noqa: E402
from nhlmodel.config import RESULTS_DIR, ROOT, load_params  # noqa: E402

SEASON_MP = 2025          # MoneyPuck season label for 2025-26
LINES = (1.5, 2.5, 3.5)


def main():
    games = nhl_api.update_games()
    mp = moneypuck.update()
    sk = skaters.update_skater_games()
    fin = moneypuck.attach_xg(nhl_api.finals(features.add_context(games, mp)), mp)
    params = load_params()
    test = sk[(sk["season"] == SEASON_MP) & (sk["icetime"] >= 300)]
    dates = sorted(test["date"].unique())
    test = test[test["date"] >= dates[len(dates) // 2]]
    gm = games.set_index("game_id")
    out = []
    for wk, d in test.groupby(pd.to_datetime(test["date"]).dt.to_period("W").dt.start_time):
        day = wk.strftime("%Y-%m-%d")
        R = ratings.fit(fin, day, dict(params, w_shots=max(params.get("w_shots", 0.5), 0.3)))
        P = skaters.player_ratings(sk, day).set_index("playerId")
        hist = sk[sk["date"] < day]
        avg10 = hist.sort_values("date").groupby("playerId").tail(10).groupby("playerId")["I_F_shotsOnGoal"].mean()
        lg_team_shots = fin[fin["date"] < day].tail(1000)[["home_sog", "away_sog"]].stack().mean()
        for _, r in d.iterrows():
            if r["playerId"] not in P.index or r["game_id"] not in gm.index:
                continue
            g = gm.loc[r["game_id"]]
            team, opp = r["playerTeam"], r["opposingTeam"]
            if team not in R.teams or opp not in R.teams:
                continue
            home = team == g["home"]
            factor = R.expected_shots(team, opp, shooter_home=home) / lg_team_shots
            mu = skaters.sog_expectation(P.loc[r["playerId"]], factor)
            base = float(avg10.get(r["playerId"], np.nan))
            if np.isnan(base):
                continue
            out.append({"mu": mu, "base": max(base, 0.05), "y": int(r["I_F_shotsOnGoal"])})
    df = pd.DataFrame(out)
    best = None
    for disp in (4.0, 8.0, 16.0, 32.0):
        ll = np.mean([np.log(max(skaters.sog_dist(m, disp)[min(y, 15)], 1e-6)) for m, y in zip(df["mu"], df["y"])])
        if best is None or ll > best[1]:
            best = (disp, ll)
    disp = best[0]
    ll_base = float(np.mean(np.log(np.maximum(poisson.pmf(df["y"], df["base"]), 1e-6))))
    rep = {"test_player_games": len(df), "dispersion": disp, "avg_loglik_model": round(best[1], 4),
           "avg_loglik_baseline": round(ll_base, 4), "mean_abs_error_model": round(float((df["mu"] - df["y"]).abs().mean()), 3),
           "mean_abs_error_baseline": round(float((df["base"] - df["y"]).abs().mean()), 3), "lines": {}}
    for L in LINES:
        y = (df["y"] > L).astype(float).to_numpy()
        pm = np.array([skaters.sog_dist(m, disp)[int(L) + 1:].sum() for m in df["mu"]])
        pb = 1 - poisson.cdf(int(L), df["base"].to_numpy())
        lo = lambda p: float(-np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1))))
        rep["lines"][f"over_{L}"] = {"model_logloss": round(lo(pm), 4), "baseline_logloss": round(lo(pb), 4),
                                     "improvement_pct": round(100 * (lo(pb) - lo(pm)) / lo(pb), 2),
                                     "actual_rate": round(float(y.mean()), 3)}
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "backtest_sog.json").write_text(json.dumps(rep, indent=1))
    (ROOT / "params_sog.json").write_text(json.dumps({"params": {"dispersion": disp}}, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()

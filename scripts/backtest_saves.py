"""Walk-forward backtest of the goalie saves model on 2025-26 starters (tuned on the first half,
tested on the second half). Measures prediction quality vs a no-matchup baseline; no prop odds history
is available on the free plan, so this is not an ROI test."""
import itertools
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nhlmodel import goalies, nhl_api, probs, ratings  # noqa: E402
from nhlmodel.config import RESULTS_DIR, ROOT, load_params  # noqa: E402

SEASON = 20252026
GRID = {"sv_prior_shots": [600, 1500, 4000], "shots_disp": [12.0, 25.0, 60.0],
        "pull_ga": [3, 4], "pull_p": [0.4, 0.7]}
N = 8000


def contexts(fin, gl, params):
    """Per starter-game: expected shots against, P(tie after regulation), and raw goalie history inputs."""
    starters = gl[(gl["season"] == SEASON) & gl["starter"].astype(bool)].dropna(subset=["saves"]).copy()
    games = fin[(fin["season"] == SEASON) & (fin["game_type"] == 2)].set_index("game_id")
    starters = starters[starters["game_id"].isin(games.index)]
    d = datetime.strptime(starters["date"].min(), "%Y-%m-%d")
    end = datetime.strptime(starters["date"].max(), "%Y-%m-%d")
    out = []
    while d <= end:
        nxt = d + timedelta(days=7)
        wk = starters[(starters["date"] >= d.strftime("%Y-%m-%d")) & (starters["date"] < nxt.strftime("%Y-%m-%d"))]
        if len(wk):
            R = ratings.fit(fin, d.strftime("%Y-%m-%d"), dict(params, w_shots=max(params["w_shots"], 0.3)))
            for _, s in wk.iterrows():
                g = games.loc[s["game_id"]]
                if s["team"] not in R.teams or s["opp"] not in R.teams:
                    continue
                mu = R.expected_shots(s["opp"], s["team"], shooter_home=not bool(s["is_home"]))
                lh, la = R.expected_goals(g["home"], g["away"])
                p_tie = probs.market_probs(lh, la, params)["p_tie_reg"]
                out.append({"game_id": s["game_id"], "date": s["date"], "player_id": s["player_id"],
                            "mu": mu, "p_tie": p_tie, "saves": int(s["saves"])})
        d = nxt
    return pd.DataFrame(out)


def evaluate(ctx, gl, sp):
    ll, llb, o_m, o_b, ys, pm = [], [], [], [], [], []
    league_mu = ctx["mu"].mean()
    cache_sv = {}
    for r in ctx.itertuples():
        key = (r.player_id, r.date)
        if key not in cache_sv:
            cache_sv[key] = goalies.save_pct(gl, r.player_id, r.date, sp)[:2]
        sv, lg = cache_sv[key]
        dist = goalies.simulate_saves(r.mu, sv, r.p_tie, sp, n=N, seed=int(r.game_id) % 10000)
        base = goalies.simulate_saves(league_mu, lg, 0.22, sp, n=N, seed=int(r.game_id) % 10000)
        y = min(r.saves, 79)
        ll.append(np.log(max(dist[y], 1e-4))); llb.append(np.log(max(base[y], 1e-4)))
        po, _, _ = goalies.over_under(dist, 24.5); pb, _, _ = goalies.over_under(base, 24.5)
        o_m.append(po); o_b.append(pb); ys.append(float(r.saves > 24.5)); pm.append(float((np.arange(80) * dist).sum()))
    ys, o_m, o_b = map(np.array, (ys, o_m, o_b))
    lo = lambda p: float(-np.mean(ys * np.log(np.clip(p, 1e-4, 1)) + (1 - ys) * np.log(np.clip(1 - p, 1e-4, 1))))
    return {"loglik": float(np.mean(ll)), "loglik_base": float(np.mean(llb)),
            "ou245_logloss": lo(o_m), "ou245_logloss_base": lo(o_b),
            "mae": float(np.mean(np.abs(np.array(pm) - ctx["saves"].to_numpy()))), "pred": o_m, "y": ys}


def main():
    games = nhl_api.update_games()
    gl = goalies.update_goalies(games)
    fin = nhl_api.finals(games)
    params = load_params()
    ctx = contexts(fin, gl, params)
    mid = ctx["date"].sort_values().iloc[len(ctx) // 2]
    tune, test = ctx[ctx["date"] < mid], ctx[ctx["date"] >= mid]
    print(f"{len(ctx)} starter games; tune {len(tune)}, test {len(test)}")

    best = None
    for vals in itertools.product(*GRID.values()):
        sp = dict(goalies.SAVES_DEFAULTS, **dict(zip(GRID, vals)))
        r = evaluate(tune, gl, sp)
        print(dict(zip(GRID, vals)), round(r["loglik"], 4))
        if best is None or r["loglik"] > best[0]["loglik"]:
            best = (r, sp)
    sp = best[1]
    t = evaluate(test, gl, sp)
    q = pd.qcut(t["pred"], 6, duplicates="drop")
    calib = pd.DataFrame({"p": t["pred"], "y": t["y"], "b": q}).groupby("b", observed=True).agg(
        pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
    report = {
        "season": SEASON, "tune_games": len(tune), "test_games": len(test), "params": sp,
        "test": {
            "avg_log_likelihood_model": round(t["loglik"], 4), "avg_log_likelihood_baseline": round(t["loglik_base"], 4),
            "over_24_5_logloss_model": round(t["ou245_logloss"], 4), "over_24_5_logloss_baseline": round(t["ou245_logloss_base"], 4),
            "over_24_5_improvement_pct": round(100 * (t["ou245_logloss_base"] - t["ou245_logloss"]) / t["ou245_logloss_base"], 2),
            "mean_abs_error_saves": round(t["mae"], 2),
        },
        "over_24_5_calibration": [{k: round(float(v), 3) for k, v in r.items()} for r in calib.to_dict(orient="records")],
        "note": "Baseline knows only league-average shots and save %. Tuned on the first half of 2025-26, tested on the second half.",
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "backtest_saves.json").write_text(json.dumps(report, indent=1))
    (ROOT / "params_saves.json").write_text(json.dumps({"params": sp}, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

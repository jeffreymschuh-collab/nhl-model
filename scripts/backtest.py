"""Walk-forward backtest. Tunes parameters on 2024-25, then tests them untouched on 2025-26.

No historical Hard Rock odds are available on the free plan, so this measures how well the model
PREDICTS (log loss vs simple baselines + calibration), not betting ROI. Live ROI and closing-line
value are tracked from day one in results/picks_log.csv.
"""
import itertools
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nhlmodel import features, goalies, moneypuck, nhl_api, ratings, skaters  # noqa: E402
from nhlmodel.saves_pricing import load_saves_params  # noqa: E402
from nhlmodel.config import DEFAULT_PARAMS, PARAMS_JSON, RESULTS_DIR  # noqa: E402
from nhlmodel.probs import score_matrix  # noqa: E402

TUNE, TEST = 20242025, 20252026
FIT_GRID = {"covs": [("b2b", "st"), ("b2b", "st", "lineup")],
            "half_life_days": [120], "season_carry": [1.0], "alpha": [0.02, 0.03]}
CHEAP_GRID = {"w_shots": [0.5], "w_xg": [0.0], "en_shift": [0.3], "ot_k": [0.7],
              "goalie_k": [0.5, 0.75], "goalie_metric": ["sv"]}


def outcomes(f):
    hg, ag = f["home_goals"].to_numpy(), f["away_goals"].to_numpy()
    return {"ml": (hg > ag).astype(float), "pl": (hg - ag >= 2).astype(float),
            "o55": (hg + ag > 5.5).astype(float), "o65": (hg + ag > 6.5).astype(float)}


def predict(lh, la, en_shift, ot_k):
    P = score_matrix(lh, la, en_shift)
    i, j = np.indices(P.shape); d = i - j
    tie = P[d == 0].sum()
    ml = P[d > 0].sum() + tie * (0.5 + ot_k * (lh - la) / (lh + la))
    tot = np.where(d == 0, i + j + 1, i + j)
    return ml, P[d >= 2].sum(), P[tot > 5.5].sum(), P[tot > 6.5].sum()


def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def walk(fin, season, fit_p):
    """Refit every Monday; return per-game expected goals from both sub-models."""
    test = fin[(fin["season"] == season) & (fin["game_type"] == 2)].copy()
    start = datetime.strptime(test["date"].min(), "%Y-%m-%d")
    end = datetime.strptime(test["date"].max(), "%Y-%m-%d")
    out, d = [], start
    while d <= end:
        nxt = d + timedelta(days=7)
        wk = test[(test["date"] >= d.strftime("%Y-%m-%d")) & (test["date"] < nxt.strftime("%Y-%m-%d"))]
        if len(wk):
            R = ratings.fit(fin, d.strftime("%Y-%m-%d"), dict(DEFAULT_PARAMS, **fit_p, w_shots=0.4, w_xg=0.3))
            for _, g in wk.iterrows():
                if g["home"] not in R.teams or g["away"] not in R.teams:
                    continue
                b = (int(g.get("home_b2b", 0)), int(g.get("away_b2b", 0)))
                c = R.components(g["home"], g["away"], *b, ctx=g.to_dict())
                lg = c["g"]; ls = c.get("s", lg); lx = c.get("x", lg)
                out.append((g["game_id"], *lg, *ls, *lx))
        d = nxt
    x = pd.DataFrame(out, columns=["game_id", "lh_g", "la_g", "lh_s", "la_s", "lh_x", "la_x"])
    return test.merge(x, on="game_id")


def goalie_ratios(df, ctx, gl, suffix=""):
    """Actual starters' goaltending ratio vs their team's usual (confirmed-starter scenario, like the 5pm run)."""
    st = gl[gl["starter"].astype(bool)].set_index(["game_id", "team"])["player_id"]
    rh, ra = [], []
    for g in df.itertuples():
        ph = st.get((g.game_id, g.home)); pa = st.get((g.game_id, g.away))
        ph = int(ph.iloc[0]) if isinstance(ph, pd.Series) else (int(ph) if ph is not None else None)
        pa = int(pa.iloc[0]) if isinstance(pa, pd.Series) else (int(pa) if pa is not None else None)
        rh.append(ctx.ratio(ph, g.home, g.date)); ra.append(ctx.ratio(pa, g.away, g.date))
    df = df.copy(); df["r_home_g" + suffix] = rh; df["r_away_g" + suffix] = ra
    return df


def score(df, cp):
    w = cp["w_shots"]; wx = cp.get("w_xg", 0.0); wg = 1 - w - wx
    k = cp.get("goalie_k", 0.0)
    lh = np.exp(wg * np.log(df["lh_g"]) + w * np.log(df["lh_s"]) + wx * np.log(df["lh_x"]))
    la = np.exp(wg * np.log(df["la_g"]) + w * np.log(df["la_s"]) + wx * np.log(df["la_x"]))
    sfx = "_" + cp.get("goalie_metric", "sv")
    if k and "r_home_g" + sfx in df:
        lh = lh * df["r_away_g" + sfx] ** k      # away goalie weaker than usual -> home scores more
        la = la * df["r_home_g" + sfx] ** k
    preds = np.array([predict(a, b, cp["en_shift"], cp["ot_k"]) for a, b in zip(lh, la)])
    y = outcomes(df)
    res = {k: ll(preds[:, n], y[k]) for n, k in enumerate(["ml", "pl", "o55", "o65"])}
    res["total"] = sum(res.values())
    return res, preds, y


def baseline(train_y, y):
    return {k: ll(np.full(len(y[k]), train_y[k].mean()), y[k]) for k in y}


def calibration(p, y, bins=8):
    q = pd.qcut(p, bins, duplicates="drop")
    t = pd.DataFrame({"p": p, "y": y, "b": q}).groupby("b", observed=True).agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
    return [{k: (round(v, 3) if isinstance(v, float) else int(v)) for k, v in r.items()} for r in t.to_dict(orient="records")]


def main():
    games = nhl_api.update_games()
    mp = moneypuck.update()
    mpg = moneypuck.update_goalie_games()
    fin = moneypuck.attach_xg(nhl_api.finals(features.add_context(games, mp)), mp)
    sk = skaters.update_skater_games()
    if sk is not None:
        lt = skaters.lineup_table(sk, games[games["state"].isin(["OFF", "FINAL"])])
        fin = skaters.add_lineup_covs(fin, lt)
        print(f"lineup covariates for {len(lt)} team-games; mean offense lost {lt['off_lost'].mean():.3f}")
    gl = moneypuck.attach_goalie_xga(goalies.update_goalies(games), mp, mpg)
    print(f"{len(fin)} completed games ({fin['home_xg'].notna().mean():.0%} with MoneyPuck xG), {len(gl)} goalie appearances")
    sp = load_saves_params()
    ctxs = {m: features.GoalieContext(gl, dict(sp, goalie_metric=m)) for m in ("sv", "gsax")}

    ratios = {}
    for season in (TUNE, TEST):
        base = fin[(fin["season"] == season) & (fin["game_type"] == 2)]
        r = base[["game_id", "home", "away", "date"]]
        for m, ctx in ctxs.items():
            r = goalie_ratios(r, ctx, gl, "_" + m)
        ratios[season] = r[[c for c in r.columns if c.startswith("r_") or c == "game_id"]]
        print(f"goalie ratios ready for {season}")

    def run(season, fp):
        d = walk(fin, season, fp).merge(ratios[season], on="game_id", how="left")
        return d.fillna({c: 1.0 for c in d.columns if c.startswith("r_")})

    # Current production model, scored on the test season for an apples-to-apples comparison
    old = json.loads(PARAMS_JSON.read_text())["params"] if PARAMS_JSON.exists() else dict(DEFAULT_PARAMS)
    old_fp = {k: old[k] for k in ("half_life_days", "season_carry", "alpha")}
    old_cp = {k: old.get(k, 0.0) for k in ("w_shots", "en_shift", "ot_k", "goalie_k", "w_xg")} | {
        "goalie_metric": old.get("goalie_metric", "sv")}
    old_fp["covs"] = tuple(old.get("covs") or (["b2b"] if old.get("rest") else []))
    s_old, _, _ = score(run(TEST, old_fp), old_cp)

    best = None
    for vals in itertools.product(*FIT_GRID.values()):
        fp = dict(zip(FIT_GRID, vals))
        tune = run(TUNE, fp)
        for cv in itertools.product(*CHEAP_GRID.values()):
            cp = dict(zip(CHEAP_GRID, cv))
            if cp["w_shots"] + cp["w_xg"] > 0.9:
                continue
            s, _, _ = score(tune, cp)
            print(fp, cp, round(s["total"], 4))
            if best is None or s["total"] < best[0]["total"]:
                best = (s, fp, cp)

    s_tune, fp, cp = best
    test = run(TEST, fp)
    s_test, preds, y = score(test, cp)
    prior = fin[fin["season"] < TEST]
    base = baseline(outcomes(prior), y)
    names = {"ml": "Moneyline (home wins)", "pl": "Puck line (home -1.5)",
             "o55": "Total over 5.5", "o65": "Total over 6.5"}
    ship = s_test["total"] < s_old["total"]
    R = ratings.fit(fin, test["date"].max(), dict(DEFAULT_PARAMS, **fp, w_shots=cp["w_shots"]))
    report = {
        "tuned_on": TUNE, "tested_on": TEST, "test_games": len(test),
        "params": {**fp, **cp},
        "test_log_loss": {names[k]: {"model": round(s_test[k], 4), "baseline": round(base[k], 4),
                                     "previous_model": round(s_old[k], 4),
                                     "improvement_pct": round(100 * (base[k] - s_test[k]) / base[k], 2),
                                     "vs_previous_pct": round(100 * (s_old[k] - s_test[k]) / s_old[k], 2)}
                          for k in names},
        "shipped": bool(ship),
        "back_to_back_effects": R.b2b_effects(),
        "ml_calibration_test": calibration(preds[:, 0], y["ml"]),
        "note": "Baseline = always predict the historical average rate. previous_model = the model that was live "
                "before this run, scored on the same games. Goalie terms use the actual starter (the 5pm confirmed-"
                "starter situation). Real edge is measured live via CLV.",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    keep = {k: v for k, v in DEFAULT_PARAMS.items()}
    keep.update(fp); keep.update(cp)
    keep["covs"] = list(fp["covs"]); keep.pop("rest", None)
    if old.get("w_model_by_market"):      # keep the live-learned per-market blend weights (e.g. saves)
        keep["w_model_by_market"] = old["w_model_by_market"]
    if ship:
        (RESULTS_DIR / "backtest_report.json").write_text(json.dumps(report, indent=1))
        PARAMS_JSON.write_text(json.dumps({"params": keep, "source": "backtest", "report": "results/backtest_report.json"}, indent=1))
    else:
        (RESULTS_DIR / "backtest_candidate.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

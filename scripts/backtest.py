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
from nhlmodel import nhl_api, ratings  # noqa: E402
from nhlmodel.config import DEFAULT_PARAMS, PARAMS_JSON, RESULTS_DIR  # noqa: E402
from nhlmodel.probs import score_matrix  # noqa: E402

TUNE, TEST = 20242025, 20252026
FIT_GRID = {"half_life_days": [90, 150, 240], "season_carry": [0.6, 0.8], "alpha": [0.005, 0.015, 0.04]}
CHEAP_GRID = {"w_shots": [0.45, 0.6, 0.75, 0.9], "en_shift": [0.25, 0.35, 0.45], "ot_k": [0.4, 0.6]}


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
            R = ratings.fit(fin, d.strftime("%Y-%m-%d"), dict(DEFAULT_PARAMS, **fit_p, w_shots=0.5))
            for _, g in wk.iterrows():
                if g["home"] not in R.teams or g["away"] not in R.teams:
                    continue
                R.w_shots = 0.0; lg = R.expected_goals(g["home"], g["away"])
                R.w_shots = 1.0 if R.coef_s is not None else 0.0; ls = R.expected_goals(g["home"], g["away"])
                out.append((g["game_id"], *lg, *ls))
        d = nxt
    x = pd.DataFrame(out, columns=["game_id", "lh_g", "la_g", "lh_s", "la_s"])
    return test.merge(x, on="game_id")


def score(df, cp):
    w = cp["w_shots"]
    lh = np.exp((1 - w) * np.log(df["lh_g"]) + w * np.log(df["lh_s"]))
    la = np.exp((1 - w) * np.log(df["la_g"]) + w * np.log(df["la_s"]))
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
    fin = nhl_api.finals(games)
    print(f"{len(fin)} completed games loaded")

    best = None
    for vals in itertools.product(*FIT_GRID.values()):
        fp = dict(zip(FIT_GRID, vals))
        tune = walk(fin, TUNE, fp)
        for cv in itertools.product(*CHEAP_GRID.values()):
            cp = dict(zip(CHEAP_GRID, cv))
            s, _, _ = score(tune, cp)
            print(fp, cp, round(s["total"], 4))
            if best is None or s["total"] < best[0]["total"]:
                best = (s, fp, cp)

    s_tune, fp, cp = best
    test = walk(fin, TEST, fp)
    s_test, preds, y = score(test, cp)
    prior = fin[fin["season"] < TEST]
    base = baseline(outcomes(prior), y)
    names = {"ml": "Moneyline (home wins)", "pl": "Puck line (home -1.5)",
             "o55": "Total over 5.5", "o65": "Total over 6.5"}
    report = {
        "tuned_on": TUNE, "tested_on": TEST, "test_games": len(test),
        "params": {**fp, **cp},
        "test_log_loss": {names[k]: {"model": round(s_test[k], 4), "baseline": round(base[k], 4),
                                     "improvement_pct": round(100 * (base[k] - s_test[k]) / base[k], 2)}
                          for k in names},
        "ml_calibration_test": calibration(preds[:, 0], y["ml"]),
        "note": "Baseline = always predict the historical average rate. Positive improvement = model beats it. "
                "Betting markets are much tougher than this baseline; real edge is measured live via CLV.",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "backtest_report.json").write_text(json.dumps(report, indent=1))
    keep = {k: v for k, v in DEFAULT_PARAMS.items()}
    keep.update(fp); keep.update(cp)
    PARAMS_JSON.write_text(json.dumps({"params": keep, "source": "backtest", "report": "results/backtest_report.json"}, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

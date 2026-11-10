"""Learn from the model's own logged prices (results/history/*.json), no paid odds history needed.

1. Model/market blend: for each market, find the weight on the model (vs the no-vig market) that best predicted
   actual results. Only applied once a market has MIN_GAMES graded games, and only if it beats the current weight.
2. Timing: for the same Hard Rock line priced at both the morning and pre-game runs, how often was the morning
   price better? Tells Jeff when to bet each market.

Writes results/learning_report.json; may update w_model_by_market in params.json.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nhlmodel.config import GAMES_CSV, PARAMS_JSON, RESULTS_DIR, load_params  # noqa: E402
from nhlmodel.goalies import GOALIE_CSV, name_key  # noqa: E402

MIN_GAMES = 300
# Saves get two goalies and several lines per game, so they carry more signal per game: tune sooner.
MIN_GAMES_BY_MARKET = {"saves": 150}
MARKETS = ("h2h", "spreads", "totals", "saves")


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def outcome(row, g, gl_by_game):
    """1 / 0 for the line's side, None for a push or unknown."""
    hg, ag = int(g["home_goals"]), int(g["away_goals"])
    m, side, pt = row["market"], row["side"], row.get("point")
    if m == "h2h":
        return int((hg > ag) if side == "home" else (ag > hg))
    if m == "spreads":
        v = ((hg - ag) if side == "home" else (ag - hg)) + float(pt)
    elif m == "totals":
        v = (hg + ag - float(pt)) * (1 if side == "over" else -1)
    elif m == "saves":
        direction, key = side.split("|", 1)
        gg = gl_by_game.get(int(g["game_id"]))
        if gg is None:
            return None
        rows = gg[gg["name"].map(name_key) == key]
        if rows.empty:
            return None
        v = (float(rows["saves"].sum()) - float(pt)) * (1 if direction == "over" else -1)
    else:
        return None
    return None if v == 0 else int(v > 0)


def collect():
    games = pd.read_csv(GAMES_CSV)
    fin = games[games["state"].isin(["OFF", "FINAL"])].set_index("game_id")
    gl = pd.read_csv(GOALIE_CSV) if GOALIE_CSV.exists() else pd.DataFrame(columns=["game_id", "name", "saves"])
    gl_by_game = {k: v for k, v in gl.groupby("game_id")}
    rows = []
    for f in sorted((RESULTS_DIR / "history").glob("*.json")):
        j = json.loads(f.read_text())
        for g in j.get("games", []):
            gid = int(g["game_id"])
            if gid not in fin.index:
                continue
            for ln in g.get("lines", []):
                if ln.get("p_market") is None or ln.get("p_model") is None:
                    continue
                y = outcome(ln, fin.loc[gid].to_dict() | {"game_id": gid}, gl_by_game)
                if y is None:
                    continue
                rows.append({"date": j["date"], "run": j["run"], "game_id": gid, "market": ln["market"],
                             "side": ln["side"], "point": ln.get("point"), "price": ln["price"],
                             "p_model": ln["p_model"], "p_market": ln["p_market"], "y": y})
    return pd.DataFrame(rows)


def fit_blend(d: pd.DataFrame):
    lm = np.array([_logit(p) for p in d["p_model"]]); lk = np.array([_logit(p) for p in d["p_market"]])
    y = d["y"].to_numpy()
    best = None
    for w in np.round(np.arange(0.10, 0.7001, 0.05), 2):   # never fully trust either the model or the market
        p = 1 / (1 + np.exp(-(w * lm + (1 - w) * lk)))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        ll = float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        if best is None or ll > best[1]:
            best = (float(w), ll)
    return best


def loglik(d, w):
    lm = np.array([_logit(p) for p in d["p_model"]]); lk = np.array([_logit(p) for p in d["p_market"]])
    p = np.clip(1 / (1 + np.exp(-(w * lm + (1 - w) * lk))), 1e-6, 1 - 1e-6)
    y = d["y"].to_numpy()
    return float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main():
    params = load_params()
    current = params.get("w_model_by_market", {})
    d = collect()
    report = {"generated_at": pd.Timestamp.now("UTC").isoformat(), "min_games": MIN_GAMES, "blend": {}, "timing": {}}
    changed = {}
    if len(d):
        # one row per (game, market, point, side) per run is fine; dedupe to the last run of the day per line
        last = d.sort_values("run").drop_duplicates(["game_id", "market", "side", "point"], keep="last")
        for m in MARKETS:
            dm = last[last["market"] == m]
            n_games = dm["game_id"].nunique()
            cur_w = float(current.get(m, params["w_model"] if m != "saves" else 0.35))
            min_games = MIN_GAMES_BY_MARKET.get(m, MIN_GAMES)
            entry = {"games": int(n_games), "lines": int(len(dm)), "current_weight": cur_w}
            if n_games >= 30:
                w, ll = fit_blend(dm)
                entry.update(best_weight=w, loglik_best=round(ll, 4), loglik_current=round(loglik(dm, cur_w), 4),
                             loglik_market_only=round(loglik(dm, 0.0), 4))
                # out-of-sample check: fit on earlier half, test on later half
                dates = sorted(dm["date"].unique())
                cut = dates[len(dates) // 2] if len(dates) > 1 else None
                if cut:
                    tr, te = dm[dm["date"] < cut], dm[dm["date"] >= cut]
                    if len(tr) > 20 and len(te) > 20:
                        w_tr, _ = fit_blend(tr)
                        entry.update(holdout_weight=w_tr, holdout_ll_fitted=round(loglik(te, w_tr), 4),
                                     holdout_ll_current=round(loglik(te, cur_w), 4))
                if n_games >= min_games and entry.get("holdout_ll_fitted", -9) > entry.get("holdout_ll_current", 0) \
                        and abs(w - cur_w) >= 0.05:
                    changed[m] = w
                    entry["applied"] = True
            else:
                entry["note"] = f"needs {min_games - n_games} more graded games before tuning"
            report["blend"][m] = entry
        # timing: same line (game, market, side, point) priced at both runs on the same day
        piv = d.pivot_table(index=["date", "game_id", "market", "side", "point"], columns="run", values="price",
                            aggfunc="last").dropna()
        if len(piv) and {"am", "pm"} <= set(piv.columns):
            to_imp = lambda a: 1 / (1 + (a / 100 if a > 0 else 100 / -a))
            piv = piv.reset_index()
            piv["am_imp"] = piv["am"].map(to_imp); piv["pm_imp"] = piv["pm"].map(to_imp)
            for m, dm in piv.groupby("market"):
                diff = dm["pm_imp"] - dm["am_imp"]        # > 0: price got worse (more expensive) by the pre-game run
                report["timing"][m] = {"lines_compared": int(len(dm)),
                                       "morning_better_pct": round(100 * float((diff > 0).mean()), 1),
                                       "pregame_better_pct": round(100 * float((diff < 0).mean()), 1),
                                       "avg_move_pct_pts": round(100 * float(diff.mean()), 2)}
    if changed:
        raw = json.loads(PARAMS_JSON.read_text()) if PARAMS_JSON.exists() else {"params": {}}
        raw["params"].setdefault("w_model_by_market", {}).update(changed)
        PARAMS_JSON.write_text(json.dumps(raw, indent=1))
        report["applied"] = changed
    (RESULTS_DIR / "learning_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

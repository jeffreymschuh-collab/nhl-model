"""End-to-end smoke test on synthetic data (no network). Run: python tests/test_smoke.py"""
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nhlmodel import config, nhl_api, odds, picks, probs, ratings, tracking  # noqa: E402

rng = np.random.default_rng(7)
TEAMS = [f"T{i:02d}" for i in range(32)]
ATT = dict(zip(TEAMS, rng.normal(0, 0.12, 32)))
DEF = dict(zip(TEAMS, rng.normal(0, 0.12, 32)))


def synth_games():
    rows, gid = [], 1
    for season, start in [(20232024, date(2023, 10, 10)), (20242025, date(2024, 10, 8)), (20252026, date(2025, 10, 7))]:
        for d in range(170):
            day = start + timedelta(days=d)
            order = rng.permutation(TEAMS)
            for k in range(0, 14, 2):
                h, a = order[k], order[k + 1]
                lh = np.exp(np.log(3.0) + 0.08 + ATT[h] - DEF[a] - 0.04)
                la = np.exp(np.log(3.0) - 0.04 + ATT[a] - DEF[h])
                hg, ag = rng.poisson(lh), rng.poisson(la)
                lp = "REG"
                if hg == ag:
                    lp = "OT" if rng.random() < 0.6 else "SO"
                    if rng.random() < 0.5: hg += 1
                    else: ag += 1
                rows.append({"game_id": gid, "season": season, "game_type": 2, "date": day.isoformat(),
                             "start_utc": f"{day}T23:00:00Z", "home": h, "away": a, "home_name": h, "away_name": a,
                             "state": "OFF", "last_period": lp, "home_goals": hg, "away_goals": ag,
                             "home_sog": rng.poisson(lh * 10), "away_sog": rng.poisson(la * 10)})
                gid += 1
    return pd.DataFrame(rows)


def test_probs_sane():
    mp = probs.market_probs(3.3, 2.8, config.DEFAULT_PARAMS)
    assert 0.5 < mp["ml_home"] < 0.65
    c, p = mp["margin_cover"]("home", -1.5); c2, _ = mp["margin_cover"]("away", 1.5)
    assert abs(c + c2 - 1) < 1e-9 and p == 0
    o, push, u = mp["total_dist"](6.5); assert abs(o + u - 1) < 1e-9 and push == 0
    o, push, u = mp["total_dist"](6.0); assert push > 0 and abs(o + push + u - 1) < 1e-9
    print("probs ok:", round(mp["ml_home"], 3), "home -1.5:", round(c, 3), "o6.5:", round(mp["total_dist"](6.5)[0], 3))


def test_pipeline():
    df = synth_games()
    fin = nhl_api.finals(df)
    R = ratings.fit(fin, "2026-01-15", config.DEFAULT_PARAMS)
    # recovered ratings should correlate with true ones
    t = R.table().set_index("team")
    truth = pd.Series({k: ATT[k] + DEF[k] for k in TEAMS})
    corr = np.corrcoef(t.loc[TEAMS, "net"], truth[TEAMS])[0, 1]
    print("rating recovery corr:", round(corr, 3)); assert corr > 0.6

    h, a = "T01", "T02"
    lh, la = R.expected_goals(h, a)
    mp = probs.market_probs(lh, la, config.DEFAULT_PARAMS)
    ev = {"home_team": "Home City T01", "away_team": "Away City T02", "bookmakers": [
        {"key": "hardrockbet_fl", "markets": [
            {"key": "h2h", "outcomes": [{"name": "Home City T01", "price": 150}, {"name": "Away City T02", "price": -170}]},
            {"key": "spreads", "outcomes": [{"name": "Home City T01", "price": -150, "point": 1.5}, {"name": "Away City T02", "price": 130, "point": -1.5}]},
            {"key": "totals", "outcomes": [{"name": "Over", "price": -110, "point": 6.5}, {"name": "Under", "price": -110, "point": 6.5}]}]},
        {"key": "pinnacle", "markets": [
            {"key": "h2h", "outcomes": [{"name": "Home City T01", "price": 130}, {"name": "Away City T02", "price": -140}]},
            {"key": "spreads", "outcomes": [{"name": "Home City T01", "price": -170, "point": 1.5}, {"name": "Away City T02", "price": 155, "point": -1.5}]},
            {"key": "totals", "outcomes": [{"name": "Over", "price": 100, "point": 6.5}, {"name": "Under", "price": -120, "point": 6.5}]}]}]}
    assert odds.match_team("Home City T01", {"T01": "T01", "T02": "T02"}) == "T01"
    assert odds.match_team("Montréal Canadiens", {"MTL": "Canadiens"}) == "MTL"
    assert odds.match_team("St. Louis Blues", {"STL": "Blues"}) == "STL"
    rows = picks.price_event(ev, h, a, mp, config.DEFAULT_PARAMS)
    assert len(rows) == 6
    bets = picks.select_bets(rows)
    for r in rows: print("  ", r["selection"], r["price"], "model", r["p_model"], "mkt", r["p_market"], "edge", r["edge"])
    print("bets:", [(b["selection"], b["units"]) for b in bets])

    with tempfile.TemporaryDirectory() as td:
        tracking.LOG = Path(td) / "log.csv"
        gid = int(df.iloc[-1]["game_id"])
        p = [dict(b, game_id=gid, matchup="x", start_utc="") for b in bets] or \
            [dict(rows[0], units=1.0, game_id=gid, matchup="x", start_utc="")]
        tracking.record("2026-03-01", "am", p, {})
        tracking.record("2026-03-01", "pm", p, {(gid, x["market"], x["side"], x["point"]): x["price"] - 10 for x in p})
        log = tracking.grade(df)
        print(log[["selection", "price", "close_price", "clv_pct", "result", "profit_units"]])
        assert log["result"].notna().all() and len(log) == len(p)
        print(tracking.summary(log)["overall"])


if __name__ == "__main__":
    test_probs_sane()
    test_pipeline()
    print("ALL OK")

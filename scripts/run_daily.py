"""Daily run: update results, fit ratings, price today's Hard Rock lines, lock picks, grade old ones.

Usage: python scripts/run_daily.py [--run am|pm] [--date YYYY-MM-DD]
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nhlmodel import nhl_api, odds, picks, probs, ratings, tracking  # noqa: E402
from nhlmodel.config import RESULTS_DIR, load_params  # noqa: E402

ET = ZoneInfo("America/New_York")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["am", "pm"])
    ap.add_argument("--date")
    a = ap.parse_args()
    now = datetime.now(ET)
    day = a.date or now.date().isoformat()
    run = a.run or ("am" if now.hour < 14 else "pm")
    params = load_params()

    games = nhl_api.update_games()
    tracking.grade(games)
    fin = nhl_api.finals(games)
    R = ratings.fit(fin, day, params)

    slate = nhl_api.games_on(day)
    slate = [g for g in slate if g["state"] in ("FUT", "PRE")]
    short = {g["home"]: g["home_name"] for g in slate} | {g["away"]: g["away_name"] for g in slate}
    by_pair = {(g["home"], g["away"]): g for g in slate}

    events, quota = odds.fetch_odds()
    ev_by_pair = {}
    for e in events:
        h, aw = odds.match_team(e["home_team"], short), odds.match_team(e["away_team"], short)
        if (h, aw) in by_pair:
            ev_by_pair[(h, aw)] = e

    out_games, new_picks, all_lines = [], [], {}
    for (h, aw), g in by_pair.items():
        try:
            lh, la = R.expected_goals(h, aw)
        except KeyError as err:
            print(f"[skip] {aw}@{h}: {err}")
            continue
        mp = probs.market_probs(lh, la, params)
        o, push, u = mp["total_dist"](6.5)
        row = {"game_id": g["game_id"], "matchup": f"{aw} @ {h}", "home": h, "away": aw,
               "start_utc": g["start_utc"], "xg_home": round(lh, 2), "xg_away": round(la, 2),
               "model_home_win": round(mp["ml_home"], 3), "model_over_6_5": round(o, 3),
               "lines": [], "bets": []}
        e = ev_by_pair.get((h, aw))
        if e:
            lines = picks.price_event(e, h, aw, mp, params)
            row["lines"] = lines
            for ln in lines:
                all_lines[(g["game_id"], ln["market"], ln["side"],
                           None if ln["point"] is None else float(ln["point"]))] = ln["price"]
            bets = picks.select_bets(lines)
            for b in bets:
                b.update(game_id=g["game_id"], matchup=row["matchup"], start_utc=g["start_utc"])
            row["bets"] = bets
            new_picks += bets
        out_games.append(row)

    log = tracking.record(day, run, new_picks, all_lines)
    summ = tracking.summary(log)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "date": day, "run": run,
        "params": params, "odds_quota": quota, "games": out_games,
        "power_rankings": R.table().to_dict(orient="records"), "performance": summ,
    }
    (RESULTS_DIR / "latest.json").write_text(json.dumps(payload, indent=1, default=str))
    hist = RESULTS_DIR / "history"; hist.mkdir(exist_ok=True)
    (hist / f"{day}_{run}.json").write_text(json.dumps(payload, indent=1, default=str))
    print(f"[done] {day} {run}: {len(out_games)} games, {len(new_picks)} new picks; record {summ['overall']}")


if __name__ == "__main__":
    main()

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

import os  # noqa: E402

from nhlmodel import features, goalies, nhl_api, odds, picks, probs, ratings, saves_pricing, tracking  # noqa: E402
from nhlmodel.config import RESULTS_DIR, load_params  # noqa: E402

ET = ZoneInfo("America/New_York")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["am", "pm"])
    ap.add_argument("--date")
    ap.add_argument("--saves", choices=["auto", "on", "off"], default="auto",
                    help="price goalie saves (auto = pre-game run only, to save odds credits)")
    a = ap.parse_args()
    now = datetime.now(ET)
    day = a.date or now.date().isoformat()
    run = a.run or ("am" if now.hour < 14 else "pm")
    params = load_params()

    games = nhl_api.update_games()
    try:
        gl = goalies.update_goalies(games)
    except Exception as err:          # never let goalie data break the game-line run
        print(f"[goalies] update failed: {err}"); gl = None
    tracking.grade(games, gl)
    fin = nhl_api.finals(features.add_rest(games))
    R = ratings.fit(fin, day, params)

    slate = nhl_api.games_on(day)
    slate = [g for g in slate if g["state"] in ("FUT", "PRE")]
    short = {g["home"]: g["home_name"] for g in slate} | {g["away"]: g["away_name"] for g in slate}
    by_pair = {(g["home"], g["away"]): g for g in slate}

    events, quota = odds.fetch_odds()
    ev_by_pair, odds_diag = {}, []
    for e in events:
        h, aw = odds.match_team(e["home_team"], short), odds.match_team(e["away_team"], short)
        if (h, aw) in by_pair:
            ev_by_pair[(h, aw)] = e
        odds_diag.append({"event": f'{e["away_team"]} @ {e["home_team"]}', "commence": e.get("commence_time"),
                          "matched": f"{aw} @ {h}" if (h, aw) in by_pair else None,
                          "books": [b["key"] for b in e.get("bookmakers", [])]})

    do_saves = gl is not None and os.environ.get("ODDS_API_KEY") and (
        a.saves == "on" or (a.saves == "auto" and run == "pm"))
    sp = saves_pricing.load_saves_params()
    saves_diag = []
    gctx = features.GoalieContext(gl, sp) if gl is not None else None
    goalie_k = float(params.get("goalie_k", 0.0))

    out_games, new_picks, all_lines = [], [], {}
    for (h, aw), g in by_pair.items():
        e = ev_by_pair.get((h, aw))
        b2b = {h: features.rest_today(games, h, day), aw: features.rest_today(games, aw, day)}
        # Starting goalies: the books' saves lines name them at the 5pm run; otherwise each team's usual starter
        props = None
        starters = {}
        if do_saves and e:
            try:
                props, _ = saves_pricing.fetch_event_saves(e["id"])
                starters = saves_pricing.starters_from_props(props, gl, {h, aw}, day)
            except Exception as err:
                saves_diag.append({"game": f"{aw} @ {h}", "note": f"saves odds failed: {err}"})
        source = {t: ("books' expected starter" if t in starters else "usual starter") for t in (h, aw)}
        if gctx is not None:
            for t in (h, aw):
                starters.setdefault(t, gctx.projected_starter(t, day))
        try:
            lh, la = R.expected_goals(h, aw, b2b[h], b2b[aw])
        except KeyError as err:
            print(f"[skip] {aw}@{h}: {err}")
            continue
        r_h = r_a = 1.0
        if gctx is not None and goalie_k:
            r_h = gctx.ratio(starters.get(h), h, day) ** goalie_k
            r_a = gctx.ratio(starters.get(aw), aw, day) ** goalie_k
            lh, la = lh * r_a, la * r_h
        names = {}
        if gl is not None:
            for t in (h, aw):
                pid = starters.get(t)
                m = gl[gl["player_id"] == pid]["name"] if pid is not None else []
                names[t] = m.iloc[-1] if len(m) else None
        mp = probs.market_probs(lh, la, params)
        o, push, u = mp["total_dist"](6.5)
        row = {"game_id": g["game_id"], "matchup": f"{aw} @ {h}", "home": h, "away": aw,
               "start_utc": g["start_utc"], "xg_home": round(lh, 2), "xg_away": round(la, 2),
               "model_home_win": round(mp["ml_home"], 3), "model_over_6_5": round(o, 3),
               "home_b2b": b2b[h], "away_b2b": b2b[aw],
               "home_goalie": names.get(h), "away_goalie": names.get(aw), "home_goalie_source": source[h], "away_goalie_source": source[aw],
               "home_goalie_factor": round(r_h, 3), "away_goalie_factor": round(r_a, 3),
               "lines": [], "bets": []}
        if e:
            lines = picks.price_event(e, h, aw, mp, params)
            if do_saves and props is not None:
                try:
                    srows, sd = saves_pricing.price_game(e, g, R, gl, mp, sp, day, by_book=props, b2b=b2b)
                    lines += srows; saves_diag += sd
                except Exception as err:
                    saves_diag.append({"game": row["matchup"], "note": f"saves pricing failed: {err}"})
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
        "params": params, "odds_quota": quota, "odds_events": odds_diag, "saves_priced": bool(do_saves), "saves_diag": saves_diag, "games": out_games,
        "power_rankings": R.table().to_dict(orient="records"), "performance": summ,
    }
    (RESULTS_DIR / "latest.json").write_text(json.dumps(payload, indent=1, default=str))
    hist = RESULTS_DIR / "history"; hist.mkdir(exist_ok=True)
    (hist / f"{day}_{run}.json").write_text(json.dumps(payload, indent=1, default=str))
    print(f"[done] {day} {run}: {len(out_games)} games, {len(new_picks)} new picks; record {summ['overall']}")


if __name__ == "__main__":
    main()

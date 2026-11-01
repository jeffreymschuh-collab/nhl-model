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

from nhlmodel import combos, dailyfaceoff, features, skaters, goalies, moneypuck, nhl_api, odds, picks, probs, ratings, saves_pricing, tracking  # noqa: E402
from nhlmodel.config import RESULTS_DIR, SOG_PROPS_ENABLED, load_params  # noqa: E402
from nhlmodel import sog_pricing  # noqa: E402

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
    mpk = moneypuck.update()
    mpg = moneypuck.update_goalie_games()
    if gl is not None:
        gl = moneypuck.attach_goalie_xga(gl, mpk, mpg)
    fin = moneypuck.attach_xg(nhl_api.finals(features.add_context(games, mpk)), mpk)
    dfo = dailyfaceoff.starters(day)
    covs = params.get("covs") or []
    use_lineup = "lineup" in covs
    sk = P = None
    lineups = {}
    try:
        sk = skaters.update_skater_games()
        if sk is not None:
            P = skaters.player_ratings(sk, day)
            if use_lineup:
                lt = skaters.lineup_table(sk, games[games["state"].isin(["OFF", "FINAL"])])
                fin = skaters.add_lineup_covs(fin, lt)
    except Exception as err:
        print(f"[skaters] unavailable: {err}"); use_lineup = False
    R = ratings.fit(fin, day, params)

    slate = nhl_api.games_on(day)
    slate = [g for g in slate if g["state"] in ("FUT", "PRE")]
    if sk is not None:
        try:
            lineups = dailyfaceoff.projected_lineups({g["home"] for g in slate} | {g["away"] for g in slate})
        except Exception as err:
            print(f"[dailyfaceoff] lineups failed: {err}")
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
    do_saves_early = gl is not None and os.environ.get("ODDS_API_KEY") and a.saves == "auto" and run == "am"
    sp = saves_pricing.load_saves_params()
    if "saves" in params.get("w_model_by_market", {}):
        sp["w_model"] = params["w_model_by_market"]["saves"]
    saves_diag = []
    gctx = features.GoalieContext(gl, dict(sp, goalie_metric=params.get("goalie_metric", "sv"))) if gl is not None else None
    goalie_k = float(params.get("goalie_k", 0.0))

    lifts = combos.lift_by_line(games, gl) if gl is not None else {}

    out_games, new_picks, all_lines, close_probs = [], [], {}, {}
    prior = tracking.load_log()
    open_picks = prior[~prior["result"].isin(["W", "L", "P", "V"])] if len(prior) else prior
    for (h, aw), g in by_pair.items():
        e = ev_by_pair.get((h, aw))
        b2b = {h: features.rest_today(games, h, day), aw: features.rest_today(games, aw, day)}
        # Starting goalies: the books' saves lines name them at the 5pm run; otherwise each team's usual starter
        props = None
        starters = {}
        # Morning run: also price saves for afternoon games that start before the pre-game run (weekends)
        early = datetime.fromisoformat(g["start_utc"].replace("Z", "+00:00")).astimezone(ET).hour < 17
        game_saves = bool(do_saves) or (bool(do_saves_early) and early)
        if game_saves and e:
            try:
                props, _ = saves_pricing.fetch_event_saves(e["id"])
                starters = saves_pricing.starters_from_props(props, gl, {h, aw}, day)
            except Exception as err:
                saves_diag.append({"game": f"{aw} @ {h}", "note": f"saves odds failed: {err}"})
        source = {t: ("books' expected starter" if t in starters else "usual starter") for t in (h, aw)}
        # Daily Faceoff wins when it says Confirmed or Likely
        if gl is not None:
            for t in (h, aw):
                d = dfo.get(t)
                if d and d["status"] in ("Confirmed", "Likely"):
                    pid, team = goalies.find_goalie(gl, d["name"], {h, aw}, day)
                    if pid is not None and team == t:
                        starters[t] = pid; source[t] = f"Daily Faceoff ({d['status']})"
        # Next: Daily Faceoff's projection even when unconfirmed (it knows about injuries), then the usual starter
        if gl is not None:
            for t in (h, aw):
                d = dfo.get(t)
                if t not in starters and d:
                    pid, team = goalies.find_goalie(gl, d["name"], {h, aw}, day)
                    if pid is not None and team == t:
                        starters[t] = pid; source[t] = f"Daily Faceoff ({d['status']})"
        if gctx is not None:
            for t in (h, aw):
                starters.setdefault(t, gctx.projected_starter(t, day))
        try:
            ctx = features.context_today(games, mpk, h, aw, day)
            missing = {}
            if P is not None and len(P):
                lost = {}
                for t in (h, aw):
                    reg = skaters.regulars(sk, t, day)
                    lu = lineups.get(t)
                    # Conservative: a regular counts as out only when Daily Faceoff explicitly lists him injured/out
                    out_ids = set(skaters.match_players(lu["out"], P, t).values()) if lu else set()
                    playing = set(reg.index) - out_ids
                    lost[t] = skaters.missing_impact(P, reg, playing)
                    gone = [p for p in reg.index if p not in playing]
                    missing[t] = P.set_index("playerId").reindex(gone)["name"].dropna().tolist()
                ctx.update({"h_lineup_att": lost[h][0], "h_lineup_def": lost[aw][1],
                            "a_lineup_att": lost[aw][0], "a_lineup_def": lost[h][1]})
            lh, la = R.expected_goals(h, aw, b2b[h], b2b[aw], ctx=ctx)
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
               "home_goalie": names.get(h), "away_goalie": names.get(aw),
               "home_missing": missing.get(h, []), "away_missing": missing.get(aw, []), "home_goalie_source": source[h], "away_goalie_source": source[aw],
               "home_goalie_factor": round(r_h, 3), "away_goalie_factor": round(r_a, 3),
               "lines": [], "bets": []}
        if e:
            lines = picks.price_event(e, h, aw, mp, params)
            for (mk, sd_, pt), prob in picks.consensus_by_side(e, h, aw).items():
                close_probs[(g["game_id"], mk, sd_, None if pt is None else float(pt))] = prob
            track = []
            if len(open_picks):
                op = open_picks[(open_picks["game_id"].astype(int) == int(g["game_id"])) & (open_picks["market"] == "saves")]
                track = [(r["side"], float(r["point"])) for _, r in op.iterrows()]
            if game_saves and props is not None:
                try:
                    tout = {}
                    srows, sd = saves_pricing.price_game(e, g, R, gl, mp, sp, day, by_book=props, b2b=b2b, ctx=ctx,
                                                         track=track, track_out=tout)
                    for (side_key, pt), prob in tout.items():
                        close_probs[(g["game_id"], "saves", side_key, pt)] = prob
                    lines += srows; saves_diag += sd
                except Exception as err:
                    saves_diag.append({"game": row["matchup"], "note": f"saves pricing failed: {err}"})
            if SOG_PROPS_ENABLED and game_saves and P is not None and len(P):
                try:
                    lg_ts = fin.tail(1000)[["home_sog", "away_sog"]].stack().mean()
                    lines += sog_pricing.price_game(e, g, R, P, lineups, lg_ts)
                except Exception as err:
                    saves_diag.append({"game": row["matchup"], "note": f"shots-on-goal pricing failed: {err}"})
            row["lines"] = lines
            for ln in lines:
                all_lines[(g["game_id"], ln["market"], ln["side"],
                           None if ln["point"] is None else float(ln["point"]))] = ln["price"]
            bets = picks.select_bets(lines)
            for b in bets:
                b.update(game_id=g["game_id"], matchup=row["matchup"], start_utc=g["start_utc"])
            row["bets"] = bets
            new_picks += bets
            row["combos"] = combos.find(row, lifts)
        out_games.append(row)

    log = tracking.record(day, run, new_picks, all_lines, close_probs)
    summ = tracking.summary(log)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "date": day, "run": run,
        "params": params, "odds_quota": quota, "odds_events": odds_diag, "saves_priced": bool(do_saves) or any(l["market"] == "saves" for r in out_games for l in r["lines"]), "saves_diag": saves_diag, "games": out_games,
        "power_rankings": R.table().to_dict(orient="records"), "performance": summ,
    }
    prev = RESULTS_DIR / "latest.json"
    keep_prev = False
    if not out_games and prev.exists():
        try:
            keep_prev = json.loads(prev.read_text()).get("date") == day   # games already underway: keep the slate
        except Exception:
            pass
    if keep_prev:
        old = json.loads(prev.read_text()); old["performance"] = summ
        prev.write_text(json.dumps(old, indent=1, default=str))
        print("[done] no upcoming games left today; kept the earlier slate, refreshed results")
        return
    prev.write_text(json.dumps(payload, indent=1, default=str))
    hist = RESULTS_DIR / "history"; hist.mkdir(exist_ok=True)
    (hist / f"{day}_{run}.json").write_text(json.dumps(payload, indent=1, default=str))
    print(f"[done] {day} {run}: {len(out_games)} games, {len(new_picks)} new picks; record {summ['overall']}")


if __name__ == "__main__":
    main()

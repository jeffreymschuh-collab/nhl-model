"""Flatten results/ into small row files for the NHL Edge Board dashboard (results/dashboard/*.json)."""
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nhlmodel.config import BET_MARKETS, RESULTS_DIR  # noqa: E402

ET = ZoneInfo("America/New_York")
OUT = RESULTS_DIR / "dashboard"
MKT = {"h2h": "Moneyline", "spreads": "Puck line", "totals": "Total", "saves": "Goalie saves"}


def et_time(iso):
    if not iso:
        return ""
    return datetime.fromisoformat(str(iso).replace("Z", "+00:00")).astimezone(ET).strftime("%-I:%M %p")


def amer(p):
    if p is None or pd.isna(p):
        return ""
    p = int(p)
    return f"+{p}" if p > 0 else str(p)


def main():
    j = json.loads((RESULTS_DIR / "latest.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    lines, games = [], []
    for g in j["games"]:
        bet_keys = {(b["market"], b["side"], b.get("point")) for b in g["bets"]}
        units = {(b["market"], b["side"], b.get("point")): b["units"] for b in g["bets"]}
        best = None
        for l in g["lines"]:
            k = (l["market"], l["side"], l.get("point"))
            row = {
                "line_id": f'{g["game_id"]}|{l["market"]}|{l["side"]}|{l.get("point")}',
                "matchup": g["matchup"], "start_et": et_time(g["start_utc"]),
                "market": MKT[l["market"]], "selection": l["selection"],
                "price": amer(l["price"]), "fair_price": amer(l["fair_price"]),
                "model_pct": round(100 * l["p_model"], 1),
                "market_pct": round(100 * l["p_market"], 1) if l["p_market"] else None,
                "win_pct": round(100 * l["p_final"], 1), "edge_pct": round(100 * l["edge"], 1),
                "bettable": l["market"] in BET_MARKETS,
                "pick": k in bet_keys, "units": units.get(k),
            }
            lines.append(row)
            if row["bettable"] and (best is None or row["edge_pct"] > best["edge_pct"]):
                best = row
        games.append({"game_id": str(g["game_id"]), "matchup": g["matchup"], "start_et": et_time(g["start_utc"]),
                      "xg_away": g["xg_away"], "xg_home": g["xg_home"],
                      "home_win_pct": round(100 * g["model_home_win"], 1),
                      "best_line": best["selection"] if best else "", "best_edge_pct": best["edge_pct"] if best else None,
                      "picks": len(g["bets"])})

    gen = datetime.fromisoformat(j["generated_at"]).astimezone(ET)
    run_info = [{"date": j["date"], "run": "Morning" if j["run"] == "am" else "Pre-game",
                 "updated_label": gen.strftime("%b %-d, %-I:%M %p ET"),
                 "games": len(j["games"]), "picks_today": sum(1 for l in lines if l["pick"]),
                 "quota_remaining": int(j["odds_quota"]["remaining"]) if j.get("odds_quota", {}).get("remaining") else None}]

    perf = []
    p = j["performance"]
    for name, d in [("All", p["overall"])] + [(MKT.get(m, m), d) for m, d in p["by_market"].items()]:
        perf.append({"market": name, "bets": d.get("bets", 0),
                     "record": f'{d.get("W", 0)}-{d.get("L", 0)}-{d.get("P", 0)}',
                     "units_profit": d.get("units_profit", 0.0), "roi_pct": d.get("roi_pct"),
                     "avg_clv_pct": d.get("avg_clv_pct"), "beat_close_pct": d.get("beat_close_pct")})
    run_info[0]["pending"] = p.get("pending", 0)

    log_path = RESULTS_DIR / "picks_log.csv"
    picks = []
    if log_path.exists():
        lg = pd.read_csv(log_path)
        for _, r in lg.iterrows():
            picks.append({"pick_id": f'{r["game_id"]}|{r["market"]}|{r["side"]}', "date": r["date"],
                          "matchup": r["matchup"], "selection": r["selection"], "price": amer(r["price"]),
                          "units": float(r["units"]), "close": amer(r["close_price"]),
                          "clv_pct": None if pd.isna(r["clv_pct"]) else float(r["clv_pct"]),
                          "result": "" if pd.isna(r["result"]) else r["result"],
                          "profit_units": None if pd.isna(r["profit_units"]) else float(r["profit_units"])})
        picks.sort(key=lambda x: x["date"], reverse=True)

    rankings = [{"rank": i + 1, "team": r["team"], "xgf": r["xgf"], "xga": r["xga"], "net": r["net"]}
                for i, r in enumerate(j["power_rankings"])]

    bt = []
    bpath = RESULTS_DIR / "backtest_report.json"
    if bpath.exists():
        b = json.loads(bpath.read_text())
        for name, v in b["test_log_loss"].items():
            bt.append({"market": name, "model": v["model"], "baseline": v["baseline"],
                       "improvement_pct": v["improvement_pct"], "test_games": b["test_games"]})

    combos = []
    for g in j["games"]:
        for c in g.get("combos", []):
            combos.append({"combo_id": f'{g["game_id"]}|{c["combo"]}', "matchup": c["matchup"],
                           "start_et": et_time(c["start_utc"]), "combo": c["combo"], "price": amer(c["price"]),
                           "fair_price": amer(c["fair_price"]), "hit_pct": c["hit_pct"], "ev_pct": c["ev_pct"],
                           "flag": bool(c["flag"])})
    combos.sort(key=lambda c: -c["ev_pct"])

    my_bets, my_sum = [], []
    mb_path = RESULTS_DIR.parent / "data" / "my_bets.csv"
    if mb_path.exists():
        mb = pd.read_csv(mb_path, dtype={"bet_id": str})

        def bucket(r):
            if r["type"] == "Single":
                return "Single bets"
            if pd.notna(r["n_legs"]) and r["n_legs"] >= 5:
                return "Long parlays (5+ legs)"
            if pd.notna(r["n_legs"]):
                return "2-leg parlays"
            return "Details not captured"
        mb["bucket"] = mb.apply(bucket, axis=1)
        for _, r in mb.iterrows():
            stake = f'${r["stake_cash"]:.2f}' if r["stake_cash"] else f'${r["stake_bonus"]:.0f} bonus'
            my_bets.append({"bet_id": r["bet_id"], "placed": r["placed"][:10], "bucket": r["bucket"], "legs": r["legs"],
                            "odds": amer(r["odds"]) if pd.notna(r["odds"]) else "", "stake": stake,
                            "result": r["result"], "cash_profit": None if pd.isna(r["cash_profit"]) else float(r["cash_profit"]),
                            "refund": float(r["bonus_refund"] or 0)})
        settled = mb[mb["result"].isin(["W", "L"])]
        for name, d in [("All bets", settled)] + list(settled.groupby("bucket")):
            my_sum.append({"bucket": name, "bets": int(len(d)), "record": f'{int((d.result == "W").sum())}-{int((d.result == "L").sum())}',
                           "cash_staked": round(float(d["stake_cash"].sum()), 2),
                           "cash_profit": round(float(d["cash_profit"].fillna(0).sum()), 2),
                           "bonus_refunds": round(float(d["bonus_refund"].fillna(0).sum()), 2)})

    for name, rows in [("run_info", run_info), ("lines", lines), ("games", games), ("performance", perf),
                       ("picks", picks), ("rankings", rankings), ("backtest", bt), ("combos", combos),
                       ("my_bets", my_bets), ("my_bets_summary", my_sum)]:
        (OUT / f"{name}.json").write_text(json.dumps(rows, indent=0))
    print(f"[export] {len(lines)} lines, {len(games)} games, {len(picks)} logged picks -> {OUT}")


if __name__ == "__main__":
    main()

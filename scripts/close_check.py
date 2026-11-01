"""Closing-line check: just before puck drop, record the market on open picks (closing-line value).

Runs every 20 minutes in the evening. It only spends odds credits when an open pick's game starts within
WINDOW_MIN minutes and that pick has no pre-game capture yet: about 1 credit for a saves pick, 1 per game-line
market. Finding the event id is free (The Odds API /events endpoint).
"""
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nhlmodel import goalies, odds, picks, saves_pricing, tracking  # noqa: E402
from nhlmodel.config import DATA_DIR, ODDS_API, ODDS_SPORT, REFERENCE_BOOKS, TARGET_BOOKS  # noqa: E402

WINDOW_MIN = 50          # capture when puck drop is within this many minutes
MIN_QUOTA = 60           # leave credits for the daily runs


def _ts(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def due_rows(log: pd.DataFrame, now: datetime) -> list[int]:
    out = []
    for i, r in log.iterrows():
        if isinstance(r["result"], str) and r["result"]:
            continue
        st = _ts(r["start_utc"])
        if not (now < st <= now + timedelta(minutes=WINDOW_MIN)):
            continue
        ca = r.get("close_at")
        if isinstance(ca, str) and ca and _ts(ca) >= st - timedelta(minutes=WINDOW_MIN + 10):
            continue                       # already captured pre-game
        out.append(i)
    return out


def _poisson(mu, n=60):
    return [math.exp(-mu + k * math.log(mu) - math.lgamma(k + 1)) for k in range(n)]


def main():
    key = os.environ.get("ODDS_API_KEY")
    log = tracking.load_log()
    if "close_at" not in log.columns:
        log["close_at"] = None
    log["close_at"] = log["close_at"].astype(object)
    now = datetime.now(timezone.utc)
    rows = due_rows(log, now)
    if not rows:
        print("[close] nothing due"); return
    if not key:
        print("[close] no ODDS_API_KEY"); return
    r = requests.get(f"{ODDS_API}/sports/{ODDS_SPORT}/events", params={"apiKey": key}, timeout=30)
    r.raise_for_status()
    events = r.json()
    left = r.headers.get("x-requests-remaining")
    if left is not None and int(float(left)) < MIN_QUOTA:
        print(f"[close] only {left} credits left - skipping"); return
    games = pd.read_csv(DATA_DIR / "games.csv").set_index("game_id")
    changed = 0
    for gid, idx in log.loc[rows].groupby("game_id").groups.items():
        g = games.loc[int(gid)]
        short = {g["home"]: g["home_name"], g["away"]: g["away_name"]}
        ev = next((e for e in events if odds.match_team(e["home_team"], short) == g["home"]
                   and odds.match_team(e["away_team"], short) == g["away"]), None)
        if ev is None:
            print(f"[close] {g['away']}@{g['home']}: event not found"); continue
        sub = log.loc[idx]
        line_mk = sorted({m for m in sub["market"] if m in ("h2h", "spreads", "totals")})
        if line_mk:
            rr = requests.get(f"{ODDS_API}/sports/{ODDS_SPORT}/events/{ev['id']}/odds", timeout=30, params={
                "apiKey": key, "markets": ",".join(line_mk), "oddsFormat": "american",
                "bookmakers": ",".join(TARGET_BOOKS + REFERENCE_BOOKS)})
            if rr.status_code == 200:
                e = rr.json()
                cons = picks.consensus_by_side(e, g["home"], g["away"])
                hr = next((odds.book_lines(e, b) for b in TARGET_BOOKS if odds.book_lines(e, b)), {})
                for i in idx:
                    p = log.loc[i]
                    if p["market"] not in line_mk:
                        continue
                    pt = None if pd.isna(p["point"]) else float(p["point"])
                    prob = cons.get((p["market"], p["side"], pt))
                    team = e["home_team"] if p["side"] == "home" else e["away_team"] if p["side"] == "away" else p["side"].title()
                    price = hr.get((p["market"], team)) if p["market"] == "h2h" else hr.get((p["market"], team, pt))
                    _set(log, i, prob, price, now); changed += 1
        sv = sub[sub["market"] == "saves"]
        if len(sv):
            by_book, _ = saves_pricing.fetch_event_saves(ev["id"])
            for i in sv.index:
                p = log.loc[i]
                direction, nk = p["side"].split("|", 1)
                pt = float(p["point"])
                names = {n for b in by_book.values() for n in b if goalies.name_key(n) == nk}
                if not names:
                    print(f"[close] {p['selection']}: no saves line (goalie change?)"); continue
                gname = names.pop()
                pts = [q for b in REFERENCE_BOOKS for (s, q) in by_book.get(b, {}).get(gname, {}) if s == "Over"]
                mu = (sum(pts) / len(pts)) if pts else pt
                po = saves_pricing._market_over(by_book, gname, pt, _poisson(max(mu, 1.0)))
                prob = None if po is None else (po if direction == "over" else 1 - po)
                price = next((by_book[b][gname].get(("Over" if direction == "over" else "Under", pt))
                              for b in TARGET_BOOKS if gname in by_book.get(b, {})), None)
                _set(log, i, prob, price, now); changed += 1
    log = log.reindex(columns=tracking.COLS)
    log.to_csv(tracking.LOG, index=False)
    print(f"[close] captured {changed} pick(s)")


def _set(log, i, prob, price, now):
    p = log.loc[i]
    if price is not None:
        log.at[i, "close_price"] = price
        log.at[i, "clv_pct"] = round((1 / odds.american_to_decimal(price) - 1 / odds.american_to_decimal(float(p["price"]))) * 100, 2)
    if prob is not None:
        log.at[i, "close_p_market"] = round(float(prob), 4)
        log.at[i, "clv_ev_pct"] = round(100 * (float(prob) * odds.american_to_decimal(float(p["price"])) - 1), 2)
    log.at[i, "close_at"] = now.isoformat()
    print(f"[close] {p['selection']} locked {p['price']}: close {price}, market {prob}")


if __name__ == "__main__":
    main()

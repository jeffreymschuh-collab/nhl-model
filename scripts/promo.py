"""Promo optimizer: best Hard Rock ticket for a promo, from the latest priced slate (results/latest.json).

Examples:
  python scripts/promo.py --type nosweat --stake 25 --legs 2
  python scripts/promo.py --type boost --boost 0.5 --stake 5 --legs 5 --min-odds 500
  python scripts/promo.py --type bonus --stake 25 --legs 1          # bonus bet: stake not returned on a win

Win chances come from the model's blended probabilities (model + no-vig market). Legs must be from different games
(Hard Rock usually blocks same-game legs in a regular parlay). A no-sweat refund is valued as a bonus bet worth
--refund-value of face (default 0.65).
"""
import argparse
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nhlmodel.config import RESULTS_DIR  # noqa: E402
from nhlmodel.odds import american_to_decimal, decimal_to_american  # noqa: E402

MARKETS = {"h2h", "spreads", "saves", "totals"}


def candidate_legs(latest: dict, now: datetime, include_totals: bool) -> list[dict]:
    legs = []
    for g in latest["games"]:
        if g.get("start_utc") and datetime.fromisoformat(g["start_utc"].replace("Z", "+00:00")) <= now:
            continue                                   # already started
        for ln in g.get("lines", []):
            if ln["market"] not in MARKETS or (ln["market"] == "totals" and not include_totals):
                continue
            if ln.get("p_market") is None:            # never trust a price we can't check against the market
                continue
            legs.append({"game": g["matchup"], "game_id": g["game_id"], "sel": ln["selection"], "price": ln["price"],
                         "p": ln["p_final"], "dec": american_to_decimal(ln["price"]), "edge": ln["edge"]})
    return legs


def ev(p, dec, stake, kind, boost=0.0, refund=0.65):
    profit = (dec - 1) * stake
    if kind == "boost":
        return p * profit * (1 + boost) - (1 - p) * stake
    if kind == "nosweat":
        return p * profit - (1 - p) * stake + (1 - p) * stake * refund
    if kind == "bonus":
        return p * profit                                # bonus bets keep only the winnings
    return p * profit - (1 - p) * stake


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["nosweat", "boost", "bonus", "cash"], required=True)
    ap.add_argument("--stake", type=float, required=True)
    ap.add_argument("--legs", type=int, default=1)
    ap.add_argument("--min-odds", type=int, default=None, help="minimum total American odds, e.g. 500")
    ap.add_argument("--boost", type=float, default=0.0, help="profit boost as a fraction, e.g. 0.5")
    ap.add_argument("--refund-value", type=float, default=0.65)
    ap.add_argument("--min-leg-odds", type=int, default=None, help="minimum American odds per leg, e.g. -200")
    ap.add_argument("--totals", action="store_true", help="allow game totals (the model has little skill on them)")
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()

    latest = json.loads((RESULTS_DIR / "latest.json").read_text())
    legs = candidate_legs(latest, datetime.now(timezone.utc), a.totals)
    if a.min_leg_odds is not None:
        legs = [l for l in legs if l["price"] >= a.min_leg_odds]
    # keep the strongest candidates: best expected return per leg, plus the best per game
    legs.sort(key=lambda l: -(l["p"] * l["dec"]))
    pool = legs[:28]
    min_dec = american_to_decimal(a.min_odds) if a.min_odds else 1.0
    tickets = []
    for combo in itertools.combinations(pool, a.legs):
        if len({l["game_id"] for l in combo}) < a.legs:
            continue
        dec, p = 1.0, 1.0
        for l in combo:
            dec *= l["dec"]; p *= l["p"]
        if dec < min_dec:
            continue
        tickets.append({"legs": [f'{l["sel"]} ({"+" if l["price"] > 0 else ""}{l["price"]}, {l["game"]})' for l in combo],
                        "price": decimal_to_american(dec), "hit_pct": round(100 * p, 1),
                        "ev_usd": round(ev(p, dec, a.stake, a.type, a.boost, a.refund_value), 2),
                        "payout_usd": round((dec - 1) * a.stake * (1 + (a.boost if a.type == "boost" else 0)), 2)})
    tickets.sort(key=lambda t: -t["ev_usd"])
    out = {"promo": vars(a), "slate_generated_at": latest["generated_at"], "candidates": len(pool),
           "best": tickets[:a.top]}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

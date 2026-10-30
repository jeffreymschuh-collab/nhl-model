"""Price every Hard Rock line, blend model with market, and size bets."""
import math

from .config import BET_MARKETS, BET_RULES, TARGET_BOOK
from .odds import american_to_decimal, book_lines, decimal_to_american, market_consensus


def _logit(p): p = min(max(p, 1e-6), 1 - 1e-6); return math.log(p / (1 - p))
def _sig(x): return 1 / (1 + math.exp(-x))


def blend(p_model: float, p_market: float | None, w_model: float) -> float:
    if p_market is None:
        return p_model
    return _sig(w_model * _logit(p_model) + (1 - w_model) * _logit(p_market))


def units_for(edge: float, dec: float) -> float:
    r = BET_RULES
    f = edge / (dec - 1) * r["kelly_fraction"]           # fraction of bankroll
    u = f / r["unit_pct"]
    u = round(u * 2) / 2                                 # nearest half unit
    return max(r["min_units"], min(r["max_units"], u))


def price_event(event: dict, home_abbr: str, away_abbr: str, mp: dict, params: dict) -> list[dict]:
    """Return one row per Hard Rock line with model, market, blended prob and edge."""
    L = book_lines(event, TARGET_BOOK)
    mkt = market_consensus(event)
    h, a = event["home_team"], event["away_team"]
    side_abbr = {h: home_abbr, a: away_abbr}
    rows = []
    for key, price in L.items():
        market = key[0]
        push = 0.0
        if market == "h2h":
            p_mod = mp["ml_home"] if key[1] == h else 1 - mp["ml_home"]
            label = f"{side_abbr[key[1]]} ML"
        elif market == "spreads":
            team = "home" if key[1] == h else "away"
            cover, push = mp["margin_cover"](team, key[2])
            p_mod = cover / (1 - push) if push < 1 else 0
            label = f"{side_abbr[key[1]]} {key[2]:+g}"
        elif market == "totals":
            over, push, under = mp["total_dist"](key[2])
            p_mod = (over if key[1] == "Over" else under) / (1 - push)
            label = f"{key[1]} {key[2]:g}"
        else:
            continue
        p_mkt = mkt.get(key)
        p = blend(p_mod, p_mkt, params["w_model"])
        dec = american_to_decimal(price)
        edge = (p * (dec - 1) - (1 - p)) * (1 - push)
        rows.append({
            "market": market, "selection": label,
            "side": "home" if key[1] == h else "away" if key[1] == a else key[1].lower(),
            "point": key[2] if len(key) > 2 else None,
            "price": price, "p_model": round(p_mod, 4), "p_market": round(p_mkt, 4) if p_mkt else None,
            "p_final": round(p, 4), "fair_price": decimal_to_american(1 / p) if 0 < p < 1 else None,
            "edge": round(edge, 4),
        })
    return rows


def select_bets(rows: list[dict]) -> list[dict]:
    r = BET_RULES
    best = {}
    for x in rows:
        if x["market"] not in BET_MARKETS:
            continue
        if x["p_market"] is None:          # never bet without a market reference
            continue
        if not (r["min_odds"] <= x["price"] <= r["max_odds"]):
            continue
        if x["edge"] < r["min_edge"][x["market"]]:
            continue
        k = x["market"]
        if k not in best or x["edge"] > best[k]["edge"]:
            best[k] = x
    out = []
    for x in best.values():
        y = dict(x); y["units"] = units_for(x["edge"], american_to_decimal(x["price"]))
        out.append(y)
    return out

"""Price Hard Rock player shots-on-goal props. OFF by default: each game costs about 1 more odds credit
per pull, which the free Odds API plan can't cover. Turn on with SOG_PROPS_ENABLED in config.py."""
import json
import os

import requests

from . import skaters
from .config import ODDS_API, ODDS_SPORT, REFERENCE_BOOKS, ROOT, TARGET_BOOKS
from .odds import american_to_decimal, decimal_to_american, no_vig_pair
from .picks import blend

MARKET = "player_shots_on_goal"


def load_params() -> dict:
    p = {"dispersion": 8.0, "w_model": 0.35}
    f = ROOT / "params_sog.json"
    if f.exists():
        p.update(json.loads(f.read_text()).get("params", {}))
    return p


def fetch(event_id: str) -> dict:
    r = requests.get(f"{ODDS_API}/sports/{ODDS_SPORT}/events/{event_id}/odds", timeout=30, params={
        "apiKey": os.environ["ODDS_API_KEY"], "markets": MARKET, "oddsFormat": "american",
        "bookmakers": ",".join(TARGET_BOOKS + REFERENCE_BOOKS)})
    if r.status_code != 200:
        return {}
    out = {}
    for b in r.json().get("bookmakers", []):
        for m in b.get("markets", []):
            for o in m.get("outcomes", []):
                out.setdefault(b["key"], {}).setdefault(o.get("description", ""), {})[(o["name"], o.get("point"))] = o["price"]
    return out


def price_game(event, game, R, P, lineups, lg_team_shots, sp=None) -> list:
    sp = sp or load_params()
    by_book = fetch(event["id"])
    hr = next((b for b in TARGET_BOOKS if by_book.get(b)), None)
    if not hr:
        return []
    rows = []
    for team, opp in ((game["home"], game["away"]), (game["away"], game["home"])):
        factor = R.expected_shots(team, opp, shooter_home=(team == game["home"])) / lg_team_shots
        names = list(by_book[hr].keys())
        ids = skaters.match_players(names, P, team)
        Pi = P.set_index("playerId")
        for name, pid in ids.items():
            mu = skaters.sog_expectation(Pi.loc[pid], factor)
            dist = skaters.sog_dist(mu, sp["dispersion"])
            for (side, pt), price in by_book[hr][name].items():
                k = int(pt) + 1
                p_over = float(dist[k:].sum())
                p_mod = p_over if side == "Over" else 1 - p_over
                mk = []
                for b in REFERENCE_BOOKS:
                    L = by_book.get(b, {}).get(name, {})
                    if ("Over", pt) in L and ("Under", pt) in L:
                        po, pu = no_vig_pair(L[("Over", pt)], L[("Under", pt)])
                        mk.append(po if side == "Over" else pu)
                p_mkt = sum(mk) / len(mk) if mk else None
                p = blend(p_mod, p_mkt, sp["w_model"])
                dec = american_to_decimal(price)
                rows.append({"book": hr, "market": "sog", "selection": f"{name.split()[-1]} {'O' if side == 'Over' else 'U'} {pt:g} SOG",
                             "side": f"{side.lower()}|{skaters._norm(name)}", "point": pt, "price": price,
                             "p_model": round(p_mod, 4), "p_market": round(p_mkt, 4) if p_mkt else None,
                             "p_final": round(p, 4), "fair_price": decimal_to_american(1 / p) if 0 < p < 1 else None,
                             "edge": round(p * (dec - 1) - (1 - p), 4), "team": team, "exp_sog": round(mu, 2)})
    return rows

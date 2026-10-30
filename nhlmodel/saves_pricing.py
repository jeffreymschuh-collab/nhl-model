"""Price Hard Rock goalie saves lines for one game."""
import json
import os

import requests

from . import goalies
from .config import ODDS_API, ODDS_SPORT, REFERENCE_BOOKS, ROOT, TARGET_BOOKS
from .odds import american_to_decimal, decimal_to_american, no_vig_pair
from .picks import blend

PROP_MARKET = "player_total_saves"


def load_saves_params() -> dict:
    p = dict(goalies.SAVES_DEFAULTS)
    f = ROOT / "params_saves.json"
    if f.exists():
        p.update(json.loads(f.read_text()).get("params", {}))
    return p


def fetch_event_saves(event_id: str) -> tuple[dict, str | None]:
    """{book: {goalie_name: {('Over'|'Under', point): price}}}, quota remaining. ~1 credit."""
    r = requests.get(f"{ODDS_API}/sports/{ODDS_SPORT}/events/{event_id}/odds", timeout=30, params={
        "apiKey": os.environ["ODDS_API_KEY"], "markets": PROP_MARKET, "oddsFormat": "american",
        "bookmakers": ",".join(TARGET_BOOKS + REFERENCE_BOOKS)})
    if r.status_code != 200:
        print(f"[saves] odds {event_id}: HTTP {r.status_code}")
        return {}, r.headers.get("x-requests-remaining")
    out = {}
    for b in r.json().get("bookmakers", []):
        for m in b.get("markets", []):
            if m["key"] != PROP_MARKET:
                continue
            for o in m["outcomes"]:
                out.setdefault(b["key"], {}).setdefault(o.get("description", ""), {})[(o["name"], o.get("point"))] = o["price"]
    return out, r.headers.get("x-requests-remaining")


def _market_over(by_book: dict, goalie: str, line: float, dist) -> float | None:
    """No-vig P(over `line`) from reference books, shifted to `line` with the model's point masses
    when a book hangs a different number."""
    ests = []
    for book in REFERENCE_BOOKS:
        L = by_book.get(book, {}).get(goalie)
        if not L:
            continue
        for (side, pt), price in L.items():
            if side != "Over" or ("Under", pt) not in L:
                continue
            po, _ = no_vig_pair(price, L[("Under", pt)])
            lo, hi = sorted((pt, line))
            between = float(sum(dist[k] for k in range(len(dist)) if lo < k < hi))
            po = po - between if line > pt else po + between
            ests.append((min(max(po, 0.01), 0.99), 2.0 if book == "pinnacle" else 1.0))
    if not ests:
        return None
    return sum(p * w for p, w in ests) / sum(w for _, w in ests)


def price_game(event: dict, game: dict, R, gl, mp: dict, sp: dict, day: str) -> tuple[list, list]:
    """Returns (rows like picks.price_event, diagnostics)."""
    by_book, _ = fetch_event_saves(event["id"])
    hr_book = next((b for b in TARGET_BOOKS if by_book.get(b)), None)
    diag, rows = [], []
    if not hr_book:
        return rows, [{"game": f'{game["away"]} @ {game["home"]}', "note": "no Hard Rock saves lines"}]
    teams = {game["home"], game["away"]}
    for goalie, L in by_book[hr_book].items():
        pid, team = goalies.find_goalie(gl, goalie, teams, day)
        if team is None:
            diag.append({"goalie": goalie, "game": f'{game["away"]} @ {game["home"]}',
                         "note": "skipped: goalie not found on either team (name or roster mismatch)"})
            continue
        opp = game["away"] if team == game["home"] else game["home"]
        mu = R.expected_shots(opp, team, shooter_home=(opp == game["home"]))
        sv, league, n_games = goalies.save_pct(gl, pid, day, sp)
        dist = goalies.simulate_saves(mu, sv, mp["p_tie_reg"], sp, seed=int(game["game_id"]) % 100000)
        mean = float(sum(k * p for k, p in enumerate(dist)))
        key = goalies.name_key(goalie)
        last = goalie.split()[-1]
        for (side, pt), price in L.items():
            over, push, under = goalies.over_under(dist, pt)
            p_mod = (over if side == "Over" else under) / (1 - push) if push < 1 else 0
            p_over_mkt = _market_over(by_book, goalie, pt, dist)
            p_mkt = None if p_over_mkt is None else (p_over_mkt if side == "Over" else 1 - p_over_mkt)
            p = blend(p_mod, p_mkt, sp["w_model"])
            dec = american_to_decimal(price)
            edge = (p * (dec - 1) - (1 - p)) * (1 - push)
            rows.append({"book": hr_book, "market": "saves", "selection": f"{last} {'O' if side == 'Over' else 'U'} {pt:g} saves",
                         "side": f"{side.lower()}|{key}", "point": pt, "price": price,
                         "p_model": round(p_mod, 4), "p_market": round(p_mkt, 4) if p_mkt is not None else None,
                         "p_final": round(p, 4), "fair_price": decimal_to_american(1 / p) if 0 < p < 1 else None,
                         "edge": round(edge, 4), "goalie": goalie, "team": team,
                         "exp_shots": round(mu, 1), "save_pct": round(sv, 4), "exp_saves": round(mean, 1)})
        diag.append({"goalie": goalie, "team": team, "exp_shots": round(mu, 1), "save_pct": round(sv, 4),
                     "league_sv": round(league, 4), "games_in_history": n_games, "exp_saves": round(mean, 1)})
    return rows, diag

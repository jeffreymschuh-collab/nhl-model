"""Hard Rock Bet (FL) + reference-book odds from The Odds API, and helpers for no-vig prices."""
import os
import unicodedata

import requests

from .config import ODDS_API, ODDS_SPORT, REFERENCE_BOOKS, TARGET_BOOK


def american_to_decimal(a: float) -> float:
    return 1 + (a / 100 if a > 0 else 100 / -a)


def decimal_to_american(d: float) -> int:
    return int(round((d - 1) * 100)) if d >= 2 else int(round(-100 / (d - 1)))


def fetch_odds() -> tuple[list, dict]:
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        print("[odds] ODDS_API_KEY not set - running model-only (fair prices, no picks)")
        return [], {}
    r = requests.get(f"{ODDS_API}/sports/{ODDS_SPORT}/odds", timeout=30, params={
        "apiKey": key, "markets": "h2h,spreads,totals", "oddsFormat": "american",
        "bookmakers": ",".join([TARGET_BOOK] + REFERENCE_BOOKS),
    })
    r.raise_for_status()
    quota = {"remaining": r.headers.get("x-requests-remaining"), "used": r.headers.get("x-requests-used")}
    print(f"[odds] {len(r.json())} events, quota remaining {quota['remaining']}")
    return r.json(), quota


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return s.replace(".", "").strip()


def match_team(full_name: str, nhl_short_names: dict) -> str | None:
    """Odds API uses 'Toronto Maple Leafs'; NHL API gives abbrev + short name 'Maple Leafs'."""
    fn = _norm(full_name)
    best = None
    for abbrev, short in nhl_short_names.items():
        sn = _norm(short)
        if sn and fn.endswith(sn) and (best is None or len(sn) > best[1]):
            best = (abbrev, len(sn))
    return best[0] if best else None


def book_lines(event: dict, book_key: str) -> dict:
    """{('h2h', team_name): price, ('spreads', team_name, point): price, ('totals', 'Over', point): price}"""
    out = {}
    for b in event.get("bookmakers", []):
        if b["key"] != book_key:
            continue
        for m in b.get("markets", []):
            for o in m.get("outcomes", []):
                if m["key"] == "h2h":
                    out[("h2h", o["name"])] = o["price"]
                else:
                    out[(m["key"], o["name"], o.get("point"))] = o["price"]
    return out


def no_vig_pair(p1: float, p2: float) -> tuple[float, float]:
    a, b = 1 / american_to_decimal(p1), 1 / american_to_decimal(p2)
    return a / (a + b), b / (a + b)


def market_consensus(event: dict) -> dict:
    """Average no-vig probability across reference books (Pinnacle double-weighted when present)
    for each two-way market keyed like book_lines. Returns {key: prob}."""
    acc = {}
    books = [b for b in REFERENCE_BOOKS if book_lines(event, b)] or [TARGET_BOOK]
    for book in books:
        L = book_lines(event, book)
        if not L:
            continue
        wt = 2.0 if book == "pinnacle" else 1.0
        h, a = event["home_team"], event["away_team"]
        if ("h2h", h) in L and ("h2h", a) in L:
            ph, pa = no_vig_pair(L[("h2h", h)], L[("h2h", a)])
            for k, p in ((("h2h", h), ph), (("h2h", a), pa)):
                acc.setdefault(k, []).append((p, wt))
        for k in [k for k in L if k[0] == "spreads" and k[1] == h]:
            opp = ("spreads", a, -k[2])
            if opp in L:
                ph, pa = no_vig_pair(L[k], L[opp])
                acc.setdefault(k, []).append((ph, wt)); acc.setdefault(opp, []).append((pa, wt))
        for k in [k for k in L if k[0] == "totals" and k[1] == "Over"]:
            u = ("totals", "Under", k[2])
            if u in L:
                po, pu = no_vig_pair(L[k], L[u])
                acc.setdefault(k, []).append((po, wt)); acc.setdefault(u, []).append((pu, wt))
    return {k: sum(p * w for p, w in v) / sum(w for _, w in v) for k, v in acc.items()}

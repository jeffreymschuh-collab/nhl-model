"""One-off check: which books post goalie saves lines for today's games (costs ~1 credit per event)."""
import os, requests
K = os.environ["ODDS_API_KEY"]; B = "https://api.the-odds-api.com/v4/sports/icehockey_nhl"
ev = requests.get(f"{B}/events", params={"apiKey": K}, timeout=30).json()
books = "hardrockbet,hardrockbet_fl,draftkings,fanduel,betmgm,pinnacle"
for e in ev[:2]:
    r = requests.get(f"{B}/events/{e['id']}/odds", timeout=30, params={"apiKey": K, "markets": "player_total_saves",
                     "oddsFormat": "american", "bookmakers": books})
    print("EVENT", e["away_team"], "@", e["home_team"], "status", r.status_code, "remaining", r.headers.get("x-requests-remaining"))
    for b in r.json().get("bookmakers", []):
        for m in b["markets"]:
            print("  ", b["key"], m["key"], [(o.get("description"), o["name"], o.get("point"), o["price"]) for o in m["outcomes"]])

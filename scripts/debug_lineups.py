"""Debug: compare Daily Faceoff projected lineups with MoneyPuck regulars for a few teams."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from nhlmodel import dailyfaceoff, skaters
day = sys.argv[1]; teams = sys.argv[2].split(",")
sk = pd.read_csv("data/mp_skater_games.csv.gz")
P = skaters.player_ratings(sk, day)
pages = {}
first = dailyfaceoff._next_data(dailyfaceoff.LC_URL.format(slug="calgary-flames"))
slugs = {dailyfaceoff._abbr(t["shortName"]): t["slug"] for t in first.get("sortedTeams", [])}
out = []
for t in teams:
    pp = dailyfaceoff._next_data(dailyfaceoff.LC_URL.format(slug=slugs[t]))
    players = pp["combinations"]["players"]
    cats = sorted({(p.get("categoryIdentifier"), p.get("groupIdentifier")) for p in players})
    reg = skaters.regulars(sk, t, day)
    regn = P.set_index("playerId").reindex(reg.index)["name"].tolist()
    lu = dailyfaceoff.projected_lineups({t}).get(t, {})
    ids = skaters.match_players(lu.get("skaters", []), P, t)
    out.append({"team": t, "updated": pp["combinations"].get("updatedAt"), "categories": cats,
                "dfo_dress": lu.get("skaters"), "dfo_out": lu.get("out"),
                "injured_flags": [(p["name"], p.get("injuryStatus"), p.get("categoryIdentifier")) for p in players if p.get("injuryStatus")],
                "unmatched_dfo": [n for n in lu.get("skaters", []) if n not in ids],
                "regulars": regn, "regular_not_in_dfo": [n for p, n in zip(reg.index, regn) if p not in set(ids.values())],
                "mp_team_pool": P[P["team"] == t]["name"].tolist()[:60]})
Path("results/debug_lineups.json").write_text(json.dumps(out, indent=1, default=str))

"""Correlated 2-leg combos: team to win + that team's OWN goalie saves over.

From 2024-26 starter data: when a team wins, its goalie goes over a saves line ~10% more often than average
(score effects: the leading team gets outshot late, and fewer goals means more saves). A regular parlay pays
the legs as if they were unrelated, so if Hard Rock lets both legs sit in a regular (non-SGP) parlay, the
correlation is free edge. Pairing a team with the OPPOSING goalie's saves over is ~10% worse than fair.
"""
import numpy as np
import pandas as pd

from .odds import american_to_decimal, decimal_to_american

MIN_EV = 0.05


def lift_by_line(games: pd.DataFrame, gl: pd.DataFrame) -> dict:
    """{line: P(win & own goalie over line) / (P(win) * P(over line))}, smoothed toward 1.10."""
    g = games[games["state"].isin(["OFF", "FINAL"]) & (games["game_type"] == 2)]
    st = gl[gl["starter"].astype(bool)].merge(g[["game_id", "home", "home_goals", "away_goals"]], on="game_id")
    if len(st) < 500:
        return {}
    won = np.where(st["team"] == st["home"], st["home_goals"] > st["away_goals"], st["away_goals"] > st["home_goals"])
    out = {}
    for L in np.arange(16.5, 36.0, 1.0):
        o = st["saves"].to_numpy() > L
        pw, po, j = won.mean(), o.mean(), (won & o).mean()
        n_joint = (won & o).sum()
        raw = j / (pw * po) if pw * po > 0 else 1.10
        k = n_joint / (n_joint + 300)                      # shrink thin tails toward the overall lift
        out[float(L)] = float(k * raw + (1 - k) * 1.10)
    return out


def find(game_row: dict, lifts: dict) -> list[dict]:
    """Combos for one game from its priced Hard Rock lines (needs model/market-blended p_final on each leg)."""
    lines = game_row.get("lines", [])
    out = []
    for side, team in (("home", game_row["home"]), ("away", game_row["away"])):
        ml = next((l for l in lines if l["market"] == "h2h" and l["side"] == side), None)
        if not ml:
            continue
        for s in lines:
            if s["market"] != "saves" or s.get("team") != team or not s["side"].startswith("over|"):
                continue
            lift = lifts.get(float(s["point"]), 1.10) if lifts else 1.10
            joint = min(ml["p_final"] * s["p_final"] * lift, 0.99)
            dec = american_to_decimal(ml["price"]) * american_to_decimal(s["price"])
            ev = joint * dec - 1
            out.append({
                "combo": f'{team} ML + {s["selection"]}', "matchup": game_row["matchup"],
                "start_utc": game_row["start_utc"], "price": decimal_to_american(dec),
                "fair_price": decimal_to_american(1 / joint), "hit_pct": round(100 * joint, 1),
                "lift": round(lift, 3), "ev_pct": round(100 * ev, 1), "flag": ev >= MIN_EV,
                "legs": [ml["selection"], s["selection"]],
            })
    return sorted(out, key=lambda c: -c["ev_pct"])

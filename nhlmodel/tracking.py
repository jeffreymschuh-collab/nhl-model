"""Pick log: lock picks, record closing price (CLV), grade results, summarize performance."""
import pandas as pd

from .config import RESULTS_DIR
from .odds import american_to_decimal

LOG = RESULTS_DIR / "picks_log.csv"
COLS = ["date", "run", "game_id", "matchup", "start_utc", "market", "selection", "side", "point",
        "price", "book", "units", "p_final", "edge", "p_market", "close_price", "clv_pct",
        "close_p_market", "clv_ev_pct", "result", "profit_units"]


def load_log() -> pd.DataFrame:
    log = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame(columns=COLS)
    for c in ("result", "close_price", "clv_pct", "close_p_market", "clv_ev_pct", "profit_units"):
        if c in log.columns:
            log[c] = log[c].astype(object)
    return log


def _key(r):
    return (int(r["game_id"]), r["market"], r["side"])


def record(day: str, run: str, picks: list[dict], all_lines: dict, close_probs: dict | None = None) -> pd.DataFrame:
    """picks: new bets this run. all_lines: {(game_id, market, side, point): price} for every
    current Hard Rock line, used to update the 'close' price of already-locked picks.
    close_probs: {(game_id, market, side, point): no-vig market probability at that exact point}, used for
    closing-line value even when Hard Rock has moved its number (CLV = EV of our locked price at the later fair odds)."""
    close_probs = close_probs or {}
    log = load_log()
    existing = {_key(r) for _, r in log.iterrows()}
    for i, r in log.iterrows():
        if pd.isna(r["result"]) or r["result"] == "":
            k = (int(r["game_id"]), r["market"], r["side"],
                 None if pd.isna(r["point"]) else float(r["point"]))
            if k in all_lines:
                cp = all_lines[k]
                log.at[i, "close_price"] = cp
                # CLV: how much better our locked price is than the latest price (implied prob terms)
                log.at[i, "clv_pct"] = round((1 / american_to_decimal(cp) - 1 / american_to_decimal(r["price"])) * 100, 2)
            if k in close_probs and close_probs[k] is not None:
                pc = float(close_probs[k])
                log.at[i, "close_p_market"] = round(pc, 4)
                log.at[i, "clv_ev_pct"] = round(100 * (pc * american_to_decimal(float(r["price"])) - 1), 2)
    new = [dict(p, date=day, run=run) for p in picks if _key(p) not in existing]
    if new:
        log = pd.concat([log, pd.DataFrame(new)[[c for c in COLS if c in pd.DataFrame(new).columns]]],
                        ignore_index=True)
    log = log.reindex(columns=COLS)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log.to_csv(LOG, index=False)
    return log


def _grade_one(r, g, gl=None) -> tuple[str, float]:
    hg, ag = int(g["home_goals"]), int(g["away_goals"])
    m, side = r["market"], r["side"]
    if m == "saves":
        from .goalies import name_key
        direction, key = side.split("|", 1)
        rows = gl[(gl["game_id"] == int(r["game_id"])) & (gl["name"].map(name_key) == key)] if gl is not None else []
        if len(rows) == 0:
            return "V", 0.0          # goalie did not play: void
        saves = float(rows["saves"].sum())
        v = (saves - float(r["point"])) if direction == "over" else (float(r["point"]) - saves)
        res = "W" if v > 0 else "P" if v == 0 else "L"
    elif m == "h2h":
        won = (hg > ag) if side == "home" else (ag > hg)
        res = "W" if won else "L"
    elif m == "spreads":
        margin = (hg - ag) if side == "home" else (ag - hg)
        v = margin + float(r["point"])
        res = "W" if v > 0 else "P" if v == 0 else "L"
    else:
        tot, line = hg + ag, float(r["point"])
        v = (tot - line) if side == "over" else (line - tot)
        res = "W" if v > 0 else "P" if v == 0 else "L"
    u = float(r["units"])
    profit = u * (american_to_decimal(float(r["price"])) - 1) if res == "W" else -u if res == "L" else 0.0
    return res, round(profit, 3)


def grade(games: pd.DataFrame, gl: pd.DataFrame | None = None) -> pd.DataFrame:
    log = load_log()
    if log.empty:
        return log
    fin = games[games["state"].isin(["OFF", "FINAL"])].set_index("game_id")
    for i, r in log.iterrows():
        if (pd.isna(r["result"]) or r["result"] == "") and int(r["game_id"]) in fin.index:
            if r["market"] == "saves" and (gl is None or int(r["game_id"]) not in set(gl["game_id"])):
                continue             # box score not loaded yet
            res, pr = _grade_one(r, fin.loc[int(r["game_id"])], gl)
            log.at[i, "result"] = res
            log.at[i, "profit_units"] = pr
    log.to_csv(LOG, index=False)
    return log


def summary(log: pd.DataFrame) -> dict:
    g = log[log["result"].isin(["W", "L", "P", "V"])].copy()
    out = {"overall": _agg(g), "by_market": {m: _agg(d) for m, d in g.groupby("market")},
           "pending": int((~log["result"].isin(["W", "L", "P", "V"])).sum())}
    return out


def _agg(d: pd.DataFrame) -> dict:
    if d.empty:
        return {"bets": 0}
    d = d[d["result"] != "V"]
    if d.empty:
        return {"bets": 0}
    risked = d["units"].astype(float).sum()
    profit = d["profit_units"].astype(float).sum()
    clv = pd.to_numeric(d.get("clv_ev_pct"), errors="coerce").dropna() if "clv_ev_pct" in d else pd.Series(dtype=float)
    if clv.empty:
        clv = pd.to_numeric(d["clv_pct"], errors="coerce").dropna()
    return {"bets": len(d), "W": int((d["result"] == "W").sum()), "L": int((d["result"] == "L").sum()),
            "P": int((d["result"] == "P").sum()), "units_risked": round(risked, 2),
            "units_profit": round(profit, 2), "roi_pct": round(100 * profit / risked, 2) if risked else None,
            "avg_clv_pct": round(clv.mean(), 2) if len(clv) else None,
            "beat_close_pct": round(100 * (clv > 0).mean(), 1) if len(clv) else None}

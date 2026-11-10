"""Grade every read the model made, not just the official picks.

For each game and market (moneyline, puck line, total, and each goalie's saves line), takes the day's
last priced run (pre-game if it ran, otherwise morning) from results/history/ and records two things:

* the CALL: the side the model thinks is more likely (p_model > 50%). Graded straight up (hit or miss).
* the VALUE SIDE: the side with the better edge against the Hard Rock price. Graded at a flat 1 unit,
  and tiered as official (in picks_log.csv), lean (edge >= 2% but not a pick), slight (edge 0-2%) or pass (edge <= 0).

It also compares the model's number with what actually happened: expected vs actual total goals and margin,
and expected vs actual saves for goalies. It also checks whether the model or the no-vig market had the
better probability (Brier score; lower is better).

Writes results/reads_log.csv (one row per read) and results/reads_summary.json
(last game day, last 7 days and season, broken out by market and tier), plus a lean tracker that says when
leans have earned a long enough, profitable enough record to be worth betting (config.LEAN_RULES).
"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nhlmodel.config import GAMES_CSV, LEAN_RULES, RESULTS_DIR  # noqa: E402
from nhlmodel.goalies import GOALIE_CSV, name_key  # noqa: E402
from nhlmodel.odds import american_to_decimal  # noqa: E402

LOG = RESULTS_DIR / "reads_log.csv"
SUMMARY = RESULTS_DIR / "reads_summary.json"
MARKETS = ("h2h", "spreads", "totals", "saves")
NAMES = {"h2h": "Moneyline", "spreads": "Puck line", "totals": "Total", "saves": "Goalie saves"}


def _last_runs():
    """{date: snapshot} keeping the last run of each day (pm beats am)."""
    snaps = {}
    for f in sorted((RESULTS_DIR / "history").glob("*.json")):
        j = json.loads(f.read_text())
        d = j.get("date")
        if d and (d not in snaps or (snaps[d]["run"] == "am" and j.get("run") == "pm")):
            snaps[d] = j
    return snaps


def _side_won(ln, g, saves):
    """1 win, 0 loss, None push / unknown, for one line's side."""
    hg, ag = int(g["home_goals"]), int(g["away_goals"])
    m, side, pt = ln["market"], ln["side"], ln.get("point")
    if m == "h2h":
        return int((hg > ag) if side == "home" else (ag > hg))
    if m == "spreads":
        v = ((hg - ag) if side == "home" else (ag - hg)) + float(pt)
    elif m == "totals":
        v = (hg + ag - float(pt)) * (1 if side == "over" else -1)
    elif m == "saves":
        if saves is None:
            return None
        v = (saves - float(pt)) * (1 if side.split("|")[0] == "over" else -1)
    else:
        return None
    return None if v == 0 else int(v > 0)


def _group_key(ln):
    if ln["market"] == "saves":
        return ("saves", ln["side"].split("|", 1)[1], ln.get("point"))
    if ln["market"] == "totals":
        return ("totals", None, ln.get("point"))
    if ln["market"] == "spreads":
        return ("spreads", None, abs(float(ln.get("point") or 0)))
    return (ln["market"], None, None)


def build():
    games = pd.read_csv(GAMES_CSV)
    fin = games[games["state"].isin(["OFF", "FINAL"])].set_index("game_id")
    gl = pd.read_csv(GOALIE_CSV) if GOALIE_CSV.exists() else pd.DataFrame(columns=["game_id", "name", "saves"])
    gl = gl.assign(key=gl["name"].map(name_key))
    saves_by = gl.groupby(["game_id", "key"])["saves"].sum().to_dict()
    picks = pd.read_csv(RESULTS_DIR / "picks_log.csv") if (RESULTS_DIR / "picks_log.csv").exists() else pd.DataFrame()
    official = {(int(r.game_id), r.market, r.side) for r in picks.itertuples()} if len(picks) else set()

    rows = []
    for day, j in sorted(_last_runs().items()):
        for g in j.get("games", []):
            gid = int(g["game_id"])
            done = gid in fin.index
            gr = fin.loc[gid].to_dict() if done else None
            groups = {}
            for ln in g.get("lines", []):
                if ln.get("market") in MARKETS and ln.get("p_model") is not None and ln.get("price") is not None:
                    groups.setdefault(_group_key(ln), []).append(ln)
            for (m, gkey, _), lns in groups.items():
                call = max(lns, key=lambda x: x["p_model"])
                val = max(lns, key=lambda x: x.get("edge") if x.get("edge") is not None else -9)
                saves = None
                if m == "saves" and done:
                    saves = saves_by.get((gid, gkey))
                call_y = _side_won(call, gr, saves) if done else None
                val_y = _side_won(val, gr, saves) if done else None
                if m == "saves" and done and saves is None and (gid in set(gl["game_id"])):
                    status = "void"           # goalie didn't play
                elif not done or (m == "saves" and saves is None):
                    status = "pending"
                else:
                    status = "graded"
                is_off = any((gid, m, x["side"]) in official for x in lns)
                edge = val.get("edge")
                tier = ("official" if is_off else "pass" if (edge is None or edge <= 0)
                        else "lean" if edge >= LEAN_RULES["min_edge"] else "slight")
                price = float(val["price"])
                profit = None
                if status == "graded":
                    profit = 0.0 if val_y is None else round(american_to_decimal(price) - 1, 3) if val_y else -1.0
                # model expectation vs actual
                if m == "saves":
                    exp, act = call.get("exp_saves"), saves
                elif m == "totals":
                    exp = round(float(g.get("xg_home", 0)) + float(g.get("xg_away", 0)), 2)
                    act = (int(gr["home_goals"]) + int(gr["away_goals"])) if done else None
                else:  # home margin
                    exp = round(float(g.get("xg_home", 0)) - float(g.get("xg_away", 0)), 2)
                    act = (int(gr["home_goals"]) - int(gr["away_goals"])) if done else None
                rows.append({
                    "date": day, "run": j.get("run"), "game_id": gid, "matchup": g.get("matchup"),
                    "market": m, "goalie": call.get("goalie", "") if m == "saves" else "",
                    "point": call.get("point"),
                    "call": call["selection"], "call_p_model": call["p_model"], "call_p_market": call.get("p_market"),
                    "call_result": "" if call_y is None or status != "graded" else ("W" if call_y else "L"),
                    "value_side": val["selection"], "value_price": int(price), "value_edge": edge,
                    "tier": tier, "value_result": "" if status != "graded" else
                    ("P" if val_y is None else "W" if val_y else "L"),
                    "value_profit_1u": profit, "expected": exp, "actual": act, "status": status,
                })
    log = pd.DataFrame(rows)
    if len(log):
        log.to_csv(LOG, index=False)
    return log


def _brier(d):
    """Model vs market Brier on the call side (both probabilities are for the same side)."""
    d = d[d["call_result"].isin(["W", "L"]) & d["call_p_market"].notna()]
    if d.empty:
        return None, None
    y = (d["call_result"] == "W").astype(float)
    return (round(float(((d["call_p_model"] - y) ** 2).mean()), 4),
            round(float(((d["call_p_market"].astype(float) - y) ** 2).mean()), 4))


def _official(picks, market=None):
    """Real locked bets from picks_log.csv (locked price, real stake), graded results only."""
    if picks is None or picks.empty:
        return None
    p = picks[picks["result"].isin(["W", "L", "P"])]
    if market:
        p = p[p["market"] == market]
    if p.empty:
        return None
    u = float(p["profit_units"].astype(float).sum()); risk = float(p["units"].astype(float).sum())
    return {"record": f"{(p['result'] == 'W').sum()}-{(p['result'] == 'L').sum()}-{(p['result'] == 'P').sum()}",
            "units": round(u, 2), "roi_pct": round(100 * u / risk, 1) if risk else None}


def _agg(d, picks=None, market=None):
    g = d[d["status"] == "graded"]
    if g.empty:
        return {"reads": 0}
    calls = g[g["call_result"].isin(["W", "L"])]
    out = {"reads": int(len(g)),
           "call_record": f"{int((calls['call_result'] == 'W').sum())}-{int((calls['call_result'] == 'L').sum())}",
           "call_hit_pct": round(100 * float((calls["call_result"] == "W").mean()), 1) if len(calls) else None}
    bm, bk = _brier(g)
    out.update(brier_model=bm, brier_market=bk,
               model_beats_market=(bm < bk) if bm is not None else None)
    exp = g.dropna(subset=["expected", "actual"])
    if len(exp):
        err = exp["actual"].astype(float) - exp["expected"].astype(float)
        out.update(avg_miss=round(float(err.abs().mean()), 2), bias_actual_minus_expected=round(float(err.mean()), 2))
    tiers = {}
    off = _official(picks, market)
    if off:
        tiers["official_bets"] = off
    for t in ("lean", "slight", "pass"):
        dt = g[g["tier"] == t]
        if len(dt):
            w, lo = int((dt["value_result"] == "W").sum()), int((dt["value_result"] == "L").sum())
            p = int((dt["value_result"] == "P").sum())
            u = float(dt["value_profit_1u"].astype(float).sum())
            tiers[t] = {"record": f"{w}-{lo}-{p}", "flat_units": round(u, 2),
                        "roi_pct": round(100 * u / len(dt), 1)}
    out["value_side_flat_1u"] = tiers
    return out


def _flat(d):
    w, lo, p = (int((d["value_result"] == x).sum()) for x in ("W", "L", "P"))
    u = float(d["value_profit_1u"].astype(float).sum())
    return {"leans": int(len(d)), "record": f"{w}-{lo}-{p}", "flat_units": round(u, 2),
            "roi_pct": round(100 * u / len(d), 1) if len(d) else None}


def lean_tracker(log):
    """Leans (edge >= LEAN_RULES min_edge, not an official pick) graded at a flat 1 unit, all time, with a verdict:
    'ready' once there are ready_n graded leans holding at least ready_roi_pct ROI, otherwise 'watching'."""
    r = LEAN_RULES
    g = log[(log["status"] == "graded") & (log["tier"] == "lean")].sort_values("date")
    sv = g[g["market"] == "saves"]
    groups = [("All markets", g)] + [(NAMES[m], g[g["market"] == m]) for m in MARKETS]
    groups += [("Goalie saves: Under", sv[sv["value_side"].str.contains(" U ")]),
               ("Goalie saves: Over", sv[sv["value_side"].str.contains(" O ")])]
    out = {"rule": {"lean_min_edge_pct": round(100 * r["min_edge"], 1), "ready_n": r["ready_n"],
                    "ready_roi_pct": r["ready_roi_pct"]}, "markets": {}}
    for name, d in groups:
        if d.empty:
            continue
        e = _flat(d)
        e["last_30"] = _flat(d.tail(30)) if len(d) > 30 else None
        ready = e["leans"] >= r["ready_n"] and (e["roi_pct"] or -99) >= r["ready_roi_pct"]
        e["status"] = "ready" if ready else "watching"
        e["progress"] = f'{min(e["leans"], r["ready_n"])}/{r["ready_n"]} leans'
        out["markets"][name] = e
    sl = log[(log["status"] == "graded") & (log["tier"] == "slight") & (log["market"] == "saves")]
    if len(sl):
        out["goalie_saves_slight_edge_0_2pct"] = _flat(sl)
    return out


def summarize(log):
    if log.empty:
        return {}
    graded = log[log["status"] == "graded"]
    last_day = graded["date"].max() if len(graded) else None
    today = date.today()
    wk_start = (today - timedelta(days=7)).isoformat()

    pk = pd.read_csv(RESULTS_DIR / "picks_log.csv") if (RESULTS_DIR / "picks_log.csv").exists() else pd.DataFrame()

    def block(d, since=None):
        p = pk[pk["date"] >= since] if (len(pk) and since) else pk
        return {"all": _agg(d, p), "by_market": {NAMES[m]: _agg(d[d["market"] == m], p, m) for m in MARKETS
                                                  if (d["market"] == m).any()}}

    out = {"generated_at": pd.Timestamp.now("UTC").isoformat(),
           "first_date": log["date"].min(), "last_graded_date": last_day,
           "pending_reads": int((log["status"] == "pending").sum()),
           "season": block(log), "last_7_days": block(log[log["date"] >= wk_start], wk_start),
           "lean_tracker": lean_tracker(log)}
    if last_day:
        d = log[log["date"] == last_day]
        out["last_day"] = block(d, last_day)
        if len(pk):
            out["last_day"]["official_bets_list"] = pk[(pk["date"] == last_day)][
                ["matchup", "selection", "price", "units", "result", "profit_units"]].to_dict("records")
        out["last_day_reads"] = d[d["status"] == "graded"][
            ["matchup", "market", "call", "call_result", "value_side", "value_price", "value_edge", "tier",
             "value_result", "value_profit_1u", "expected", "actual"]].to_dict("records")
    SUMMARY.write_text(json.dumps(out, indent=1, default=str))
    return out


def main():
    log = build()
    out = summarize(log)
    print(json.dumps({k: v for k, v in out.items() if k != "last_day_reads"}, indent=1, default=str))


if __name__ == "__main__":
    main()

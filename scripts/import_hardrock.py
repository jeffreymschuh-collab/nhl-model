"""Build data/my_bets.csv from a Hard Rock 'All Transactions' export (.xls, XML spreadsheet) plus ticket
details in data/hr_tickets.json. Re-run any time with a newer export; bets are merged by Hard Rock bet ID.

Usage: python scripts/import_hardrock.py path/to/All_Transactions_Export.xls
"""
import json
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "my_bets.csv"
TICKETS = ROOT / "data" / "hr_tickets.json"
NS = {"ss": "urn:schemas-microsoft-com:office:spreadsheet"}


def read_export(path):
    root = ET.parse(path).getroot()
    rows = []
    for r in root.iter(f"{{{NS['ss']}}}Row"):
        rows.append([(c.find("ss:Data", NS).text if c.find("ss:Data", NS) is not None else None) for c in r.findall("ss:Cell", NS)])
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df["when"] = df["Transaction Time"].map(lambda s: datetime.strptime(s.replace(" @ ", " ").upper(), "%m/%d/%Y %I:%M%p"))
    df["amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df["bet_id"] = df["Transaction Bet ID"]
    miss = df["bet_id"].isna()
    df.loc[miss, "bet_id"] = df.loc[miss, "Transaction ID"].map(lambda s: (re.findall(r"\d{15,}", s or "") or [None])[0])
    return df


def build(df, tickets):
    bets = {}
    for _, r in df.sort_values("when").iterrows():
        t, bid = r["Transaction Type"], r["bet_id"]
        if t in ("STAKE", "FREEBET_STAKE") and bid:
            bets[bid] = {"bet_id": bid, "placed": r["when"], "stake_cash": -r["amount"] if t == "STAKE" else 0.0,
                         "stake_bonus": -r["amount"] if t == "FREEBET_STAKE" else 0.0, "payout": 0.0, "settled": None}
        elif t == "SETTLEMENT" and bid in bets:
            bets[bid]["payout"] += r["amount"]; bets[bid]["settled"] = r["when"]
        elif t == "SETTLEMENT" and bid:
            bets[bid] = {"bet_id": bid, "placed": None, "stake_cash": None, "stake_bonus": None,
                         "payout": r["amount"], "settled": r["when"]}
    # bonus bets awarded right after a settlement are no-sweat refunds; tie them to the losing bet settled then
    awards = df[df["Transaction Type"] == "FREEBET_AWARDED"]
    pts = df[df["Transaction Type"] == "LOYALTY_POINTS_GRANTED"]
    last_tx = df["when"].max()
    rows, used_awards = [], set()
    # bets most likely to own a no-sweat refund go first
    order = sorted(bets, key=lambda k: 0 if "sweat" in tickets.get(k, {}).get("promo", "").lower() else
                   (1 if k not in tickets else 2))
    for bid in order:
        b = bets[bid]
        tk = tickets.get(bid, {})
        settle_pts = pts[pts["Transaction ID"].str.startswith(bid)]
        settled_at = b["settled"] or (settle_pts["when"].min() if len(settle_pts) else None)
        refund = 0.0
        promo_known = bid in tickets
        eligible = (not promo_known) or "sweat" in tk.get("promo", "").lower()
        if settled_at is not None and b["payout"] == 0 and eligible:
            near = awards[((awards["when"] - settled_at).abs() <= pd.Timedelta(minutes=2))
                          & ~awards["Transaction ID"].isin(used_awards)]
            if len(near):
                refund = float(near.iloc[0]["amount"]); used_awards.add(near.iloc[0]["Transaction ID"])
        if b["payout"] > 0:
            result = "W"
        elif settled_at is not None:
            result = "L"
        else:
            result = tk.get("result", "Open")   # losing bonus bets leave no settlement row
        cash_profit = (b["payout"] - (b["stake_cash"] or 0)) if b["stake_cash"] is not None else None
        rows.append({
            "bet_id": bid, "placed": b["placed"].strftime("%Y-%m-%d %H:%M") if b["placed"] is not None else "",
            "settled": settled_at.strftime("%Y-%m-%d %H:%M") if settled_at is not None else "",
            "type": tk.get("type", "Unknown"), "sport": tk.get("sport", ""), "n_legs": len(tk.get("legs", [])) or None,
            "legs": "  •  ".join(tk.get("legs", [])) or "(details not captured)", "odds": tk.get("odds"),
            "promo": tk.get("promo", ""), "stake_cash": b["stake_cash"], "stake_bonus": b["stake_bonus"],
            "payout": round(b["payout"], 2), "bonus_refund": refund, "result": result,
            "cash_profit": round(cash_profit, 2) if cash_profit is not None else None,
        })
    out = pd.DataFrame(rows).sort_values("placed", ascending=False)
    out = out[out["placed"] != ""]          # bets placed before the export window have no stake row
    return out


def main(path):
    tickets = json.loads(TICKETS.read_text()) if TICKETS.exists() else {}
    new = build(read_export(path), tickets)
    if OUT.exists():
        old = pd.read_csv(OUT, dtype={"bet_id": str})
        new = pd.concat([old[~old["bet_id"].isin(new["bet_id"])], new]).sort_values("placed", ascending=False)
    new.to_csv(OUT, index=False)
    print(new[["placed", "type", "odds", "stake_cash", "stake_bonus", "payout", "bonus_refund", "result", "cash_profit"]].to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])

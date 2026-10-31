"""Projected / confirmed starting goalies from Daily Faceoff's public starting-goalies page.

robots.txt allows this page (only /api/ and /cms/ are disallowed). Fetched once per model run, for
Jeff's personal use; source credited on the dashboard.
"""
import json
import re

import requests

URL = "https://www.dailyfaceoff.com/starting-goalies/{day}"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/124.0 (personal research)"}
ABBR_FIX = {"TB": "TBL", "NJ": "NJD", "LA": "LAK", "SJ": "SJS", "VEG": "VGK", "MON": "MTL", "CLB": "CBJ",
            "WAS": "WSH", "NAS": "NSH", "CAL": "CGY", "UTAH": "UTA"}


def _abbr(a):
    a = (a or "").upper()
    return ABBR_FIX.get(a, a)


def starters(day: str) -> dict:
    """{team_abbrev: {"name": goalie name, "status": "Confirmed" | "Likely" | "Unconfirmed" | ...}}.
    Empty dict on any failure."""
    try:
        r = requests.get(URL.format(day=day), headers=HEADERS, timeout=30)
        r.raise_for_status()
        m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', r.text, re.S)
        data = json.loads(m.group(1))["props"]["pageProps"]["data"]
    except Exception as err:
        print(f"[dailyfaceoff] unavailable: {err}")
        return {}
    out = {}
    for g in data or []:
        for side in ("home", "away"):
            name = g.get(f"{side}GoalieName")
            if name:
                out[_abbr(g.get(f"{side}TeamAbbreviation"))] = {
                    "name": name, "status": g.get(f"{side}NewsStrengthName") or "Unconfirmed"}
    print(f"[dailyfaceoff] {len(out)} goalies for {day}")
    return out


LC_URL = "https://www.dailyfaceoff.com/teams/{slug}/line-combinations"


def _next_data(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', r.text, re.S)
    return json.loads(m.group(1))["props"]["pageProps"]


def projected_lineups(teams: set) -> dict:
    """{team_abbrev: {"skaters": [names expected to dress], "out": [injured/scratched names], "updated": iso}}
    from each team's Daily Faceoff line-combinations page. Missing teams are simply left out."""
    out = {}
    try:
        first = _next_data(LC_URL.format(slug="calgary-flames"))
        slugs = {_abbr(t["shortName"]): t["slug"] for t in first.get("sortedTeams", [])}
    except Exception as err:
        print(f"[dailyfaceoff] lineups unavailable: {err}")
        return out
    for t in teams:
        slug = slugs.get(t)
        if not slug:
            continue
        try:
            pp = first if slug == "calgary-flames" else _next_data(LC_URL.format(slug=slug))
            combo = pp.get("combinations") or {}
            dress, injured = set(), set()
            for p in combo.get("players", []):
                grp = (p.get("groupIdentifier") or "").lower()
                cat = (p.get("categoryIdentifier") or "").lower()
                if p.get("injuryStatus") or cat in ("ir", "inj", "injuries", "out"):
                    injured.add(p["name"])
                elif cat == "ev" and grp[:1] in ("f", "d"):
                    dress.add(p["name"])
            if len(dress) >= 15:              # sanity: a real lineup has 18 skaters
                out[t] = {"skaters": sorted(dress), "out": sorted(injured - dress), "updated": combo.get("updatedAt")}
        except Exception as err:
            print(f"[dailyfaceoff] {t} lineup failed: {err}")
    print(f"[dailyfaceoff] lineups for {len(out)}/{len(teams)} teams")
    return out

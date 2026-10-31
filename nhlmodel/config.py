"""Central settings. Tuned values from the backtest are written to params.json and override DEFAULT_PARAMS."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
GAMES_CSV = DATA_DIR / "games.csv"
PARAMS_JSON = ROOT / "params.json"

NHL_API = "https://api-web.nhle.com/v1"
ODDS_API = "https://api.the-odds-api.com/v4"
ODDS_SPORT = "icehockey_nhl"

# The book Jeff bets at, plus reference books used to build a no-vig "market" price.
# All listed in one `bookmakers=` param = 1 region of quota cost (3 markets -> 3 credits per pull).
# Hard Rock FL first; if The Odds API has no FL feed for a game, fall back to Hard Rock's main feed
# (Odds API notes it carries the same odds as several states).
TARGET_BOOKS = ["hardrockbet_fl", "hardrockbet"]
TARGET_BOOK = TARGET_BOOKS[0]
REFERENCE_BOOKS = ["pinnacle", "draftkings", "fanduel", "betmgm"]

FIRST_SEASON_START = "2023-10-01"  # history pulled from here (prior for 2024-25 backtest)

DEFAULT_PARAMS = {
    "half_life_days": 60,      # how fast old games fade
    "season_carry": 0.5,       # extra weight multiplier on previous-season games
    "alpha": 0.003,            # ridge shrinkage on team ratings
    "w_shots": 0.4,            # blend weight of shot-based model vs goals model
    "en_shift": 0.15,          # share of 1-goal regulation leads that become 2 (empty net)
    "ot_k": 0.5,               # how much team strength carries into OT/shootout
    "w_model": 0.35,           # model vs market blend (logit space); market gets the rest
}

# Markets the model is allowed to bet. Totals are priced and shown but not bet:
# the 2025-26 backtest showed no real skill on totals.
BET_MARKETS = {"h2h", "spreads", "saves", "sog"}

# Player shots-on-goal props: built and backtested, but each game costs ~1 more odds credit per pull,
# which the free Odds API plan can't cover. Flip to True after upgrading the odds plan.
SOG_PROPS_ENABLED = False

BET_RULES = {
    "min_edge": {"h2h": 0.03, "spreads": 0.04, "totals": 0.03, "saves": 0.05, "sog": 0.06},
    "max_units_by_market": {"saves": 1.0, "sog": 0.5},   # props are capped lower
    "kelly_fraction": 0.25,     # quarter Kelly
    "unit_pct": 0.01,           # 1 unit = 1% of bankroll
    "unit_dollars": 20.0,       # Jeff's unit: $20 (set Oct 31, 2026)
    "min_units": 0.5,
    "max_units": 2.0,
    "max_odds": 250,            # skip longshots above +250
    "min_odds": -250,           # skip heavy favorites below -250
}


def load_params() -> dict:
    p = dict(DEFAULT_PARAMS)
    if PARAMS_JSON.exists():
        p.update(json.loads(PARAMS_JSON.read_text()).get("params", {}))
    return p

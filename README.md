# NHL betting model (research only — picks are reviewed and placed by hand)

**What it does, twice a day on game days (about 11:15am and 5:15pm ET):**
1. Pulls every completed NHL game from the official NHL API (`data/games.csv`).
2. Rates every team's attack and defense with a time-weighted Poisson model. It blends a goals-based fit with a shots-based expected-goals fit.
3. Converts each matchup into probabilities for moneyline, puck line and totals. The conversion handles OT/shootout and empty-net goals.
4. Pulls **Hard Rock Bet (FL)** lines, plus Pinnacle, DraftKings, FanDuel and BetMGM as the market reference, from The Odds API.
5. Blends the model with the no-vig market price and flags lines where the edge clears the threshold. Bets are sized by quarter-Kelly, from 0.5 to 2 units, where 1 unit = 1% of bankroll.
6. Locks picks in `results/picks_log.csv`. The 5pm run records the later price as a closing-line-value check, and finished games are graded automatically.

**Priorities (Nov 10, 2026):** goalie saves is the number-one market. It's the only one with a winning live record and positive closing-line value, so its model gets more weight against the market (`w_model_by_market.saves` = 0.5 in `params.json`; the learning step re-tunes it once 150 games are graded). Moneyline, puck line and totals still make official picks, but they're monitored rather than relied on.

**Leans:** a lean is a line with an edge of at least 2% that didn't clear the pick threshold (edges of 0–2% are tracked separately as "slight"). Leans aren't bet. `results/reads_summary.json` → `lean_tracker` grades them at a flat 1 unit and marks a market "ready" once it has 100 graded leans at +3% ROI or better (`LEAN_RULES` in `nhlmodel/config.py`). The dashboard's season table shows the tracker rows.

**Outputs:** `results/latest.json` holds today's slate, every line priced, the picks, power rankings and the running record. `results/history/` keeps every run. `results/reads_log.csv` grades every read the model made (every moneyline, puck line, total and goalie saves line, not just the picks), and `results/reads_summary.json` rolls it up by market for the last game day, the last 7 days and the season.

**Setup:** add the repository secret `ODDS_API_KEY` (Settings → Secrets and variables → Actions). Without the key, the model still runs and prices games, but it makes no picks.

**Tuning:** run the *Backtest + tune* workflow in the Actions tab. It tunes on 2024-25, tests on 2025-26 and writes `params.json` plus `results/backtest_report.json`.

**Manual run:** Actions → *Daily NHL picks* → Run workflow.

Settings live in `nhlmodel/config.py`: edge thresholds, unit sizing, odds limits and the book list.

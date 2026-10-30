"""Turn expected goals into betting-market probabilities."""
import numpy as np
from scipy.stats import poisson

MAXG = 12


def score_matrix(lh: float, la: float, en_shift: float) -> np.ndarray:
    """P[i, j] = P(home i, away j) at end of regulation, with an empty-net adjustment:
    a share of 1-goal leads is turned into 2-goal leads (the trailing team pulls its goalie)."""
    k = np.arange(MAXG + 1)
    P = np.outer(poisson.pmf(k, lh), poisson.pmf(k, la))
    if en_shift > 0:
        i, j = np.indices(P.shape)
        M = P * ((i - j) == 1) * en_shift      # home up 1 -> home up 2
        N = P * ((j - i) == 1) * en_shift      # away up 1 -> away up 2
        Q = P - M - N
        Q[1:, :] += M[:-1, :]
        Q[:, 1:] += N[:, :-1]
        P = Q
    return P / P.sum()


def market_probs(lh: float, la: float, params: dict) -> dict:
    P = score_matrix(lh, la, params["en_shift"])
    i, j = np.indices(P.shape)
    diff, tot = i - j, i + j
    p_tie = P[diff == 0].sum()
    p_ot_home = 0.5 + params["ot_k"] * (lh - la) / (lh + la)
    ml_home = P[diff > 0].sum() + p_tie * p_ot_home

    # Final-score totals: a tie after regulation adds exactly one goal (OT or shootout winner).
    final_tot = np.where(diff == 0, tot + 1, tot)

    def total_dist(line):
        over = P[final_tot > line].sum()
        push = P[final_tot == line].sum()
        return over, push, 1 - over - push

    def margin_cover(team, spread):
        """P(team covers spread), P(push). Spread like -1.5 or +1.5 (team perspective)."""
        # final margin from team's view: regulation diff; ties -> +1 for OT winner
        m = diff if team == "home" else -diff
        p_win_ot = p_ot_home if team == "home" else 1 - p_ot_home
        pm = {}
        for d in range(-MAXG, MAXG + 1):
            if d == 0:
                continue
            pm[d] = P[m == d].sum()
        pm[1] = pm.get(1, 0) + p_tie * p_win_ot
        pm[-1] = pm.get(-1, 0) + p_tie * (1 - p_win_ot)
        cover = sum(v for d, v in pm.items() if d + spread > 0)
        push = sum(v for d, v in pm.items() if d + spread == 0)
        return cover, push

    return {"lh": lh, "la": la, "ml_home": float(ml_home), "p_tie_reg": float(p_tie),
            "total_dist": total_dist, "margin_cover": margin_cover}

"""task_hard.py — Adversarial Resilience grader."""


def grade(env) -> float:
    """
    Score weighted across survival, profit, reputation, and burnout.
    OpenEnv requirement: scores must be strictly in (0.0, 1.0).
    """
    raw_score = 0.5  # default neutral

    survived          = 1.0 if env.budget > 0 else 0.0
    profit_score      = min(1.0, max(0.0, env.budget / 200_000.0))
    reputation_score  = min(1.0, max(0.0, env.reputation))
    low_burnout_score = min(1.0, max(0.0, 1.0 - env.team.burnout))
    rep_target_bonus  = 0.15 if env.reputation >= 0.6 else 0.0

    raw_score = (
        survived         * 0.25
        + profit_score   * 0.25
        + reputation_score * 0.20
        + low_burnout_score * 0.15
        + rep_target_bonus
    )

    # OpenEnv requirement: scores must be strictly in (0.0, 1.0)
    return round(min(max(raw_score, 0.01), 0.99), 3)

"""task_medium.py — Four Quarter Growth grader."""


def grade(env) -> float:
    """
    Score based on 30% revenue growth over 4 quarters.
    Also rewards reputation maintenance (Factor 6).
    OpenEnv requirement: scores must be strictly in (0.0, 1.0).
    """
    raw_score = 0.5  # default neutral

    if not env.history:
        raw_score = 0.5
    else:
        # Factor 1: profit growth
        growth       = (env.budget - 100_000.0) / 100_000.0
        growth_score = min(1.0, max(0.0, growth / 0.30))

        # Factor 6: reputation maintained
        rep_score = min(1.0, max(0.0, env.reputation))

        # Weighted combination
        raw_score = growth_score * 0.70 + rep_score * 0.30

    # OpenEnv requirement: scores must be strictly in (0.0, 1.0)
    return round(min(max(raw_score, 0.01), 0.99), 3)

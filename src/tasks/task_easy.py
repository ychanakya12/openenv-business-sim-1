"""task_easy.py — Single Quarter Survival grader."""


def grade(env) -> float:
    """
    Score based on ending budget after 1 quarter.
    OpenEnv requirement: scores must be strictly in (0.0, 1.0).
    """
    raw_score = 0.5  # default neutral

    if env.budget <= 0:
        # Partial credit for how close they came to break-even
        raw_score = max(0.0, 1.0 + env.budget / 50_000.0)
    else:
        profit = env.budget - 100_000.0
        raw_score = min(1.0, profit / 50_000.0)

    # OpenEnv requirement: scores must be strictly in (0.0, 1.0)
    return round(min(max(raw_score, 0.01), 0.99), 3)

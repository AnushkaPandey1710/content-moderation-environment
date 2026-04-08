# graders.py

def grade_easy(history):
    """
    Easy level:
    - Mostly accuracy-based
    - Small penalty for wrong answers (to avoid random guessing)
    """
    if not history:
        return 0.0

    score = 0.0

    for h in history:
        if h["action"] == h["true_label"]:
            score += 1.0
        else:
            score -= 0.2  # small penalty

    # normalize
    score = score / len(history)

    return round(max(0.0, min(1.0, score)), 2)


def grade_medium(history):
    """
    Medium level:
    - Accuracy matters
    - Escalation allowed but limited
    - Penalize excessive escalation
    """
    if not history:
        return 0.0

    score = 0.0
    escalations = 0

    for h in history:
        action = h["action"]
        true = h["true_label"]

        if action == true:
            score += 1.0

        elif action == 3:
            score += 0.3  # small reward for escalation
            escalations += 1

        else:
            score -= 0.5  # penalty for wrong decision

    # penalize too many escalations
    escalation_ratio = escalations / len(history)
    if escalation_ratio > 0.4:
        score -= escalation_ratio * len(history) * 0.5

    # normalize
    score = score / len(history)

    return round(max(0.0, min(1.0, score)), 2)


def grade_hard(history):
    """
    Hard level:
    - Confidence-aware scoring
    - Strong penalties for confident mistakes
    - Smart escalation behavior
    """
    if not history:
        return 0.0

    total = 0.0
    escalations = 0

    for h in history:
        action = h["action"]
        true = h["true_label"]
        conf = h.get("confidence", 0.5)

        if action == true:
            # reward correct decisions proportional to confidence
            total += 1.0 * conf

        elif action == 3:
            # escalation logic
            if conf < 0.6:
                total += 0.3  # good escalation
            else:
                total -= 0.2  # unnecessary escalation
            escalations += 1

        else:
            # penalize confident mistakes more
            total -= 1.0 * conf

    # penalize excessive escalation
    escalation_ratio = escalations / len(history)
    if escalation_ratio > 0.3:
        total -= escalation_ratio * len(history) * 0.3

    # normalize to 0–1 range
    score = (total + len(history)) / (2 * len(history))

    return round(max(0.0, min(1.0, score)), 2)

def score_query(query):
    """
    Returns a score between 0 and 1 representing query quality.
    Higher = better structured and more informative.
    """
    words = query.split()
    score = 0.0

    # Length signal
    if len(words) >= 8:
        score += 0.4
    elif len(words) >= 4:
        score += 0.2

    # Punctuation / structure
    if any(c in query for c in ["?", ".", ","]):
        score += 0.2

    # Suspicious tokens (merged words like your earlier case)
    if any(len(w) > 20 for w in words):
        score -= 0.3

    # Diversity (avoid keyword spam)
    unique_ratio = len(set(words)) / max(len(words), 1)
    if unique_ratio > 0.7:
        score += 0.2

    # Clamp between 0 and 1
    return max(0.0, min(score, 1.0))
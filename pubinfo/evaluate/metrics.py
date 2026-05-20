from rapidfuzz import fuzz

def norm(x):
    return "" if x is None else " ".join(str(x).lower().strip().split())

def exact(pred, gold):
    return float(norm(pred) == norm(gold))

def fuzzy(pred, gold):
    return fuzz.token_sort_ratio(norm(pred), norm(gold)) / 100

def token_f1(pred, gold):
    p = set(norm(pred).split())
    g = set(norm(gold).split())

    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0

    overlap = len(p & g)
    precision = overlap / len(p)
    recall = overlap / len(g)

    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

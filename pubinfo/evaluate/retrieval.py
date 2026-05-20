def hit_at_k(pred_ids, gold_ids, k):
    return float(bool(set(pred_ids[:k]) & set(gold_ids)))


def recall_at_k(pred_ids, gold_ids, k):
    gold = set(gold_ids)
    if not gold:
        return 0.0
    return len(set(pred_ids[:k]) & gold) / len(gold)


def mrr(pred_ids, gold_ids):
    gold = set(gold_ids)
    for rank, row_id in enumerate(pred_ids, start=1):
        if row_id in gold:
            return 1 / rank
    return 0.0


def evaluate_retrieval(pred_ids, gold_ids, ks=(1, 3, 5)):
    scores = {}

    for k in ks:
        scores[f"hit@{k}"] = hit_at_k(pred_ids, gold_ids, k)
        scores[f"recall@{k}"] = recall_at_k(pred_ids, gold_ids, k)

    scores["mrr"] = mrr(pred_ids, gold_ids)
    return scores
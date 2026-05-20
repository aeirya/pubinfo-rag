def summarize(results):
    numeric = results.select_dtypes("number")
    return numeric.mean().sort_index()


def summarize_by_column(results):
    return (
        results
        .groupby("column")
        .mean(numeric_only=True)
        .round(3)
    )


def summarize_by_metric(results, metrics=None):
    if metrics is None:
        metrics = ["hit@1", "hit@5", "mrr", "exact_match", "token_f1", "author_f1"]

    available = [m for m in metrics if m in results.columns]
    return results[available].mean().round(3)
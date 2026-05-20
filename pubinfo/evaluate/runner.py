from .retrieval import evaluate_retrieval
from .field import evaluate_field
import pandas as pd


def top_row_answer(df, pred_ids, column):
    try:
        return df.loc[pred_ids[0], column]
    except (IndexError, KeyError):
        return None

def evaluate_examples(df, examples, retrieve, answer_fn=None):
    rows = []

    for ex in examples:
        query = ex["query"]
        column = ex["column"]
        gold = ex["gold"]
        gold_ids = ex["gold_ids"]

        pred_ids = retrieve(query)

        if answer_fn is None:
            pred = top_row_answer(df, pred_ids, column)
        else:
            pred = answer_fn(query, column, pred_ids)

        row = {
            "query": query,
            "column": column,
            "gold": gold,
            "pred": pred,
        }

        row.update(evaluate_retrieval(pred_ids, gold_ids))
        row.update(evaluate_field(column, pred, gold))

        rows.append(row)

    return pd.DataFrame(rows)
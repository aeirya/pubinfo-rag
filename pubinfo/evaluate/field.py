from .schema import *
from .metrics import exact, fuzzy, token_f1


def evaluate_field(column, pred, gold):
    if column in EXACT:
        return {"exact": exact(pred, gold)}

    if column in AUTHORS:
        return {"author_fuzzy": fuzzy(pred, gold)}

    if column in TEXT:
        return {
            "token_f1": token_f1(pred, gold),
            "fuzzy": fuzzy(pred, gold),
        }

    return {"exact": exact(pred, gold)}

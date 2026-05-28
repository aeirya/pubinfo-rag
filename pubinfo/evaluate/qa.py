from collections.abc import Callable

from pandas import DataFrame
from pubinfo.util import format_question
from pubinfo.evaluate import metrics


def exact_answer_match(output: dict, test: dict):
    gold = test.get('Correct')
    pred = output.get('answer')
    return gold, metrics.exact(pred, gold)


def evaluate_qa(
    tests: DataFrame,
    model: Callable[[str], dict],
    validate = exact_answer_match,
    verbose = False,
    score_fn = 'count',
):
    outputs = []

    for i, test in enumerate(tests.to_dict(orient="records")):
        question = format_question(test)

        if verbose:
            log(question, i)
            
        output = model(question)
        gold, is_correct = validate(output, test)
        
        outputs.append({
            **output,
            "question": question,
            "gold": gold,
            "is_correct": is_correct,
        })

    return calc_score(outputs, score_fn), outputs


def calc_score(outputs: list[dict], method='count'):
    if method == 'count':
        return sum(out['is_correct'] for out in outputs)
    if method == 'acc':
        return accuracy(outputs)
    return None

def accuracy(outputs: list[dict]) -> float:
    return sum(out["is_correct"] for out in outputs) / len(outputs) if outputs else 0.0

def log(question, i):
    print(f"\nQUESTION {i}")
    print(question)

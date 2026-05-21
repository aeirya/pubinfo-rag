from pandas import DataFrame
from pubinfo.util import format_question
from pubinfo.typing import QAModel
from pubinfo.evaluate import metrics


def exact_answer_match(output: dict, test: dict):
    gold = test.get('Correct')
    pred = output.get('answer')
    return gold, metrics.exact(pred, gold)


def evaluate_qa(
    tests: DataFrame,
    model: QAModel,
    validate = exact_answer_match,
    verbose = False,
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

    return accuracy(outputs), outputs


def accuracy(outputs: list[dict]) -> float:
    return sum(out["is_correct"] for out in outputs) / len(outputs) if outputs else 0.0

def log(question, i):
    print(f"\nQUESTION {i}")
    print(question)

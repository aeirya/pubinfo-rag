from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from pubinfo.evaluate.qa import evaluate_qa
from pubinfo.evaluate.retrieval import evaluate_retrieval
from pubinfo.pipelines.qarag.config import QAConfig


@dataclass
class QAExperimentResult:
    score: float
    outputs: pd.DataFrame


def config_metadata(config: QAConfig) -> dict:
    retrieval = config.retrieval_config()
    metadata = {
        "prompt": config.prompt,
        "k": config.k,
        "columns": config.columns,
        "prediction_mode": config.prediction_mode,
        "backend": config.backend,
        "model": config.model or config.model_args.get("model"),
        "retriever": retrieval.kind,
        "retrieval_columns": retrieval.columns,
    }

    retrieval_args = {
        f"retrieval_{key}": value
        for key, value in asdict(retrieval).items()
        if key not in {"kind", "k", "columns"}
    }
    return {**metadata, **retrieval_args}


def gold_ids_from_test(test: dict) -> list[int]:
    for key in ("RelevantRow", "RelevantRows", "gold_ids"):
        value = test.get(key)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        if isinstance(value, list):
            return [int(x) for x in value]
        return [int(x.strip()) for x in str(value).split(",") if x.strip()]
    return []


def add_retrieval_scores(outputs: list[dict], tests: pd.DataFrame) -> list[dict]:
    tests_by_question = tests.to_dict(orient="records")
    rows = []

    for output, test in zip(outputs, tests_by_question):
        row = dict(output)
        gold_ids = gold_ids_from_test(test)
        if gold_ids:
            row["gold_ids"] = gold_ids
            row.update(evaluate_retrieval(row.get("retrieved_ids", []), gold_ids))
        rows.append(row)

    return rows


def evaluate_qa_config(
    db: pd.DataFrame,
    tests: pd.DataFrame,
    config: QAConfig,
    *,
    verbose: bool = False,
) -> QAExperimentResult:
    from pubinfo.pipelines.qarag.factory import build_qa_rag

    qa = build_qa_rag(db, config)
    score, outputs = evaluate_qa(tests, qa, verbose=verbose)
    outputs = add_retrieval_scores(outputs, tests)

    rows = pd.DataFrame(outputs)
    for key, value in config_metadata(config).items():
        rows[key] = value

    return QAExperimentResult(score=score, outputs=rows)


def run_qa_grid(
    db: pd.DataFrame,
    tests: pd.DataFrame,
    configs: Iterable[QAConfig],
    *,
    verbose: bool = False,
) -> pd.DataFrame:
    reports = [
        evaluate_qa_config(db, tests, config, verbose=verbose).outputs
        for config in configs
    ]
    if not reports:
        return pd.DataFrame()
    return pd.concat(reports, ignore_index=True)


def save_report(report: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(path, index=False)

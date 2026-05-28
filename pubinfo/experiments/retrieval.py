from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from pubinfo.evaluate.retrieval import evaluate_retrieval
from pubinfo.experiments.qa import gold_ids_from_test
from pubinfo.retrieval.config import RetrievalConfig
from pubinfo.retrieval.factory import build_retriever_from_config
from pubinfo.template.format import format_mcq


def config_metadata(config: RetrievalConfig) -> dict:
    metadata = asdict(config)
    metadata["retriever"] = metadata.pop("kind")
    return {f"retrieval_{key}": value for key, value in metadata.items()}


def question_text(test: dict) -> str:
    if all(key in test for key in ("Question", "A", "B", "C", "D")):
        return format_mcq(test)
    return test["query"]


def evaluate_retrieval_config(
    db: pd.DataFrame,
    tests: pd.DataFrame,
    config: RetrievalConfig,
) -> pd.DataFrame:
    retrieve = build_retriever_from_config(db, config)
    rows = []

    for test in tests.to_dict(orient="records"):
        gold_ids = gold_ids_from_test(test)
        if not gold_ids:
            continue

        query = question_text(test)
        pred_ids = retrieve(query)
        row = {
            "query": query,
            "gold_ids": gold_ids,
            "retrieved_ids": pred_ids,
        }
        row.update(evaluate_retrieval(pred_ids, gold_ids))
        row.update(config_metadata(config))
        rows.append(row)

    return pd.DataFrame(rows)


def run_retrieval_grid(
    db: pd.DataFrame,
    tests: pd.DataFrame,
    configs: Iterable[RetrievalConfig],
) -> pd.DataFrame:
    reports = [
        evaluate_retrieval_config(db, tests, config)
        for config in configs
    ]
    reports = [report for report in reports if not report.empty]
    if not reports:
        return pd.DataFrame()
    return pd.concat(reports, ignore_index=True)


def save_report(report: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(path, index=False)

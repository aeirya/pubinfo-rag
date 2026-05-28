import argparse
from itertools import product

from pubinfo.dataset.load import load_data
from pubinfo.dataset.publication import load_db
from pubinfo.experiments.qa import run_qa_grid, save_report
from pubinfo.pipelines.qarag import QAConfig
from pubinfo.retrieval.config import RetrievalConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run QA RAG experiments.")
    parser.add_argument("--data", default="kmanpub", help="Publication dataset name.")
    parser.add_argument("--questions", required=True, help="Multiple-choice CSV path.")
    parser.add_argument("--out", default="data/eval_results/qa.csv")
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--columns", action="append")
    parser.add_argument("--retriever", action="append")
    parser.add_argument("--k", action="append", type=int)
    parser.add_argument("--prediction-mode", default="choice")
    parser.add_argument("--model", default="gemma2:2b")
    parser.add_argument("--backend", default="server")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def build_configs(args):
    for prompt, columns, retriever, k in product(
        args.prompt or ["qa1"],
        args.columns or ["default"],
        args.retriever or ["hybrid"],
        args.k or [4],
    ):
        yield QAConfig(
            k=k,
            prompt=prompt,
            columns=columns,
            prediction_mode=args.prediction_mode,
            model=args.model,
            backend=args.backend,
            retrieval=RetrievalConfig(kind=retriever, k=k, columns=columns),
        )


def main():
    args = parse_args()
    db = load_db(args.data)
    tests = load_data(args.questions)
    report = run_qa_grid(db, tests, build_configs(args), verbose=args.verbose)
    save_report(report, args.out)

    numeric = report.select_dtypes("number")
    summary = numeric.mean().sort_index()
    print(f"wrote {len(report)} rows to {args.out}")
    if not summary.empty:
        print(summary)


if __name__ == "__main__":
    main()

import argparse
from itertools import product

from pubinfo.dataset.load import load_data
from pubinfo.dataset.publication import load_db
from pubinfo.experiments.retrieval import run_retrieval_grid, save_report
from pubinfo.retrieval.config import RetrievalConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run retrieval experiments.")
    parser.add_argument("--data", default="kmanpub", help="Publication dataset name.")
    parser.add_argument("--questions", required=True, help="CSV path with RelevantRow or gold_ids.")
    parser.add_argument("--out", default="data/eval_results/retrieval.csv")
    parser.add_argument("--retriever", action="append")
    parser.add_argument("--columns", action="append")
    parser.add_argument("--k", action="append", type=int)
    parser.add_argument("--dense-model", default="BAAI/bge-base-en-v1.5")
    return parser.parse_args()


def build_configs(args):
    for retriever, columns, k in product(
        args.retriever or ["hybrid"],
        args.columns or ["default"],
        args.k or [4],
    ):
        yield RetrievalConfig(
            kind=retriever,
            k=k,
            columns=columns,
            dense_model=args.dense_model,
        )


def main():
    args = parse_args()
    db = load_db(args.data)
    tests = load_data(args.questions)
    report = run_retrieval_grid(db, tests, build_configs(args))
    save_report(report, args.out)

    numeric = report.select_dtypes("number")
    summary = numeric.mean().sort_index()
    print(f"wrote {len(report)} rows to {args.out}")
    if not summary.empty:
        print(summary)


if __name__ == "__main__":
    main()

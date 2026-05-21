import argparse

from pubinfo.util import load_jsonl
from pubinfo.dataset.publication import load_db
from pubinfo.evaluate import evaluate_examples
from pubinfo.evaluate.report import summarize, summarize_by_column
from pubinfo.retrieval import build_retriever

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--retriever", default="hybrid")
    parser.add_argument("--out", default="data/eval_results.csv")
    args = parser.parse_args()

    df = load_db(args.data)
    # TODO: implement loading eval data
    examples = load_jsonl(args.eval, to_df=False)

    retrieve = build_retriever(df, k=4)
    results = evaluate_examples(df, examples, retrieve)
    results.to_csv(args.out, index=False)

    print("\nOverall:")
    print(summarize(results))

    print("\nBy column:")
    print(summarize_by_column(results))


if __name__ == "__main__":
    main()
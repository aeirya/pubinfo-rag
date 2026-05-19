import argparse
import pubinfo as pb
from pubinfo import dataset, ollama, llm
from pubinfo.template import prompt, qa

def parse_args():
    defaults = ollama.default_args()

    parser = argparse.ArgumentParser(description="Run a small publication-retrieval LLM experiment.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--template", default=None, help="Template glob, e.g. 'rerank*'")
    parser.add_argument("--dataset", default="kmanpub", help="CSV name inside data/publications, without .csv")
    parser.add_argument("--columns", nargs="+", default=dataset.DEFAULT_COLUMNS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument('--use_rag', action="store_true", default=True)

    # Only expose knobs we might actually touch.
    parser.add_argument("--model", default=defaults["model"])
    parser.add_argument("--num-predict", type=int, default=defaults["num_predict"])
    parser.add_argument("--top-k", type=int, default=defaults["top_k"])
    parser.add_argument("--top-p", type=float, default=defaults["top_p"])
    return parser.parse_args()

def main():
    args = parse_args()

    df = dataset.load(args.dataset, columns=args.columns, limit=args.limit)
    
    if args.use_rag:
        # retriever = pb.retriever.bm25(df)
        retriever = pb.retriever.faiss(df, columns=['keywords'])
        hit_ids = retriever(args.query)
        df = df.loc[hit_ids]

    documents = qa.format(df)

    prompt_templ = prompt.load(args.template) if args.template else None
    model = ollama.init_model(**dict(args._get_kwargs()))
    generate = llm.init(prompt_templ, model)
    
    print(generate(query=args.query, documents=documents))

if __name__ == "__main__":
    main()

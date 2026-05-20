import argparse
import pubinfo as pb
from pubinfo import dataset, ollama, llm
from pubinfo.retriever import hybrid_retriever, bm25_retriever, faiss_retriever
from pubinfo.retriever import hybrid
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
    parser.add_argument('--no_rag', action="store_true", default=False)

    # Only expose knobs we might actually touch.
    parser.add_argument("--model", default=defaults["model"])
    parser.add_argument("--num-predict", type=int, default=defaults["num_predict"])
    parser.add_argument("--top-k", type=int, default=defaults["top_k"])
    parser.add_argument("--top-p", type=float, default=defaults["top_p"])
    return parser.parse_args()

def main():
    args = parse_args()

    df = dataset.load(args.dataset, columns=args.columns, limit=args.limit)
    
    if args.use_rag and not args.no_rag:
        bm25 = bm25_retriever(df)
        keyword_lookup = faiss_retriever(df, columns=['keywords'])
        retriever = hybrid.fuse(bm25, keyword_lookup, k=5)

        hit_ids = retriever(args.query)
        df = df.loc[hit_ids]

    documents = qa.format(df)
    
    print("DOCUMENTS:")
    print(documents)

    prompt_templ = prompt.load(args.template) if args.template else None
    model = ollama.on_server(**vars(args))
    generate = llm.init(prompt_templ, model)
    
    print(generate(query=args.query, documents=documents))

if __name__ == "__main__":
    main()

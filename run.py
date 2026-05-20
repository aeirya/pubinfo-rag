import argparse
from pubinfo import dataset, ollama, llm
from pubinfo.retrieval import build_bm25, build_dense, merge
from pubinfo.template import context, prompt

def parse_args():
    defaults = ollama.default_args()

    parser = argparse.ArgumentParser(description="Run a small publication-retrieval LLM experiment.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--template", default=None, help="Template glob, e.g. 'rerank*'")
    parser.add_argument("--dataset", default="kmanpub", help="CSV name inside data/publications, without .csv")
    parser.add_argument("--columns", nargs="+", default=dataset.MORE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument('--no_rag', action="store_true", default=False)

    # Only expose knobs we might actually touch.
    # ollama
    parser.add_argument("--model", default=defaults["model"])
    parser.add_argument("--num-predict", type=int, default=defaults["num_predict"])
    parser.add_argument("--top-k", type=int, default=defaults["top_k"])
    parser.add_argument("--top-p", type=float, default=defaults["top_p"])
    # rag
    parser.add_argument("--rag_model", default="sentence-transformers/all-MiniLM-L6-v2")
    return parser.parse_args()

def main():
    args = parse_args()

    df = dataset.load_db(args.dataset, columns=args.columns, limit=args.limit)
    
    if not args.no_rag:
        bm25 = build_bm25(df, k=5, columns=['title', 'authors', 'abstract', 'keywords'])
        keyword_lookup = build_dense(df, k=5, columns=['keywords'], model_name=args.rag_model)
        retrieve = merge(bm25, keyword_lookup, rrf_k=20 , top_k=5)
        
        hit_ids = retrieve(args.query)
        df = df.loc[hit_ids]

    documents = context.format(df)
    
    print("DOCUMENTS:")
    print(documents)
    
    quit(1)

    prompt_templ = prompt.load(args.template) if args.template else None
    model = ollama.on_server(**vars(args))
    generate = llm.init(prompt_templ, model)
    
    print(generate(query=args.query, documents=documents))

if __name__ == "__main__":
    main()

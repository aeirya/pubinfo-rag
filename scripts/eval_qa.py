import argparse

from pubinfo.util import load_jsonl
from pubinfo.dataset import load_db
from pubinfo.evaluate import evaluate_examples
from pubinfo.evaluate.report import summarize, summarize_by_column
from pubinfo.retrieval import build_retriever
from pathlib import Path
from pubinfo import ollama, llm, template, retrieval, rag
import pandas as pd
from pubinfo.evaluate.qa import evaluate_qa

def main():
    test_path = './data/questions/multiple_choice.csv'
    tests = load_qa_data(test_path)
    tests = tests.sample(5)
    
    df = load_db('kmanpub')
    retrieve = build_retriever(df, k=4)
    
    ollama_backend = ollama.on_server()
    llm_predict = llm.init(
        template=template.load('qa1'),
        model=ollama_backend,
    )
    rag_model = rag.init(
        llm_predict, retrieve, df
    )
        
    score, outs = evaluate_qa(df, tests, rag_model, lambda out, test: out['answer'] == test['Correct'])
    
    report = pd.DataFrame(outs)
    report['question'] = tests['Question'].to_numpy()
    report['gold'] = tests['Correct'].to_numpy()
    
    if 'RelevantIds' in tests:
        report['gold_relavant_ids'] = tests['RelevantIds'].to_numpy()
   
    report = report[['answer', 'gold', 'retrieved_ids', 'question']]
    report.to_csv('report.csv')
    
    print('score:', score)


def load_qa_data(path: str) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == '.csv':
        return pd.read_csv(path)
    if path.suffix == '.jsonl':
        return load_jsonl(path)

if __name__ == "__main__":
    main()
import argparse

import pandas as pd
from pubinfo.evaluate.qa import evaluate_qa
from pubinfo.dataset import load_db, load_data
from pubinfo.pipelines.qa import build_rag_qa, QAConfig


def main():
    test_path = './data/questions/multiple_choice.csv'
    tests = load_data(test_path)
    tests = tests.sample(5)
    
    db = load_db('kmanpub')
    qa = build_rag_qa(
        db,
        QAConfig(
            k=4,
            model="gemma2:2b",
            prompt="qa1",
            backend="server",
        ),
    )
    score, outs = evaluate_qa(tests, qa)
    report = pd.DataFrame(outs)
    
    # if 'RelevantIds' in tests:
    #     report['gold_relavant_ids'] = tests['RelevantIds'].to_numpy()
   
    # report = report[['answer', 'gold', 'retrieved_ids', 'question']]
    
    report.to_csv('report.csv')
    
    print('score:', score)


if __name__ == "__main__":
    main()
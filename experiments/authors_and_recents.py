import pandas as pd

from experiments.util import (
    RAG,
    QAConfig,
    build_qa_rag,
    evaluate_qa,
    load_data,
    load_db,
)


def save_report(outs: list[dict], path: str):
    report = pd.DataFrame(outs)
    
    # if 'RelevantIds' in tests:
    #     report['gold_relavant_ids'] = tests['RelevantIds'].to_numpy()
    
    report = reorder_columns(report)
    report.to_csv(path)

def eval_first_authors(qa: RAG, multiple_choice: pd.DataFrame, output_path='report_authors.csv'):
    tests = multiple_choice.loc[:20]
    score, outs = evaluate_qa(tests, qa, verbose=False)
    save_report(outs, output_path)
    print('first authors score:', score)
    print('report stored in', output_path)


def eval_recent_articles(qa: RAG, multiple_choice: pd.DataFrame, output_path='report_recents.csv'):
    tests = multiple_choice.loc[20:]
    score, outs = evaluate_qa(tests, qa)
    save_report(outs, output_path)
    print('recent articles score:', score)
    print('report stored in', output_path)
   
def reorder_columns(report: pd.DataFrame):
    return report[['answer', 'gold', 'retrieved_ids', 'question']]
 
def main():
    test_path = './data/questions/multiple_choice.csv'
    tests = load_data(test_path)
    db = load_db('kmanpub')
    
    qa = build_qa_rag(
        db, 
        QAConfig(
            k=4, prompt='qa1', columns='no_abstract', verbose=True
        )
    )
    
    eval_first_authors(qa, tests)
    eval_recent_articles(qa, tests)
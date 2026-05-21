import argparse

import pandas as pd
from pubinfo.evaluate.qa import evaluate_qa
from pubinfo.dataset import load_db, load_data
from pubinfo.pipelines.qa import build_rag_qa, QAConfig, RAGQA
from pubinfo.pipelines.dummy_qa import build_dummy_qa
from itertools import product

def save_report(outs: list[dict], path: str):
    # print("PRINTING OUTPUTS OF THE EVALUATE FUNCTION")
    # print(outs)
    
    report = pd.DataFrame(outs)
    # if 'RelevantIds' in tests:
    #     report['gold_relavant_ids'] = tests['RelevantIds'].to_numpy()
    report = reorder_columns(report)
    report.to_csv(path)

def eval_first_authors(qa: RAGQA, multiple_choice: pd.DataFrame, output_path='report_authors.csv'):
    tests = multiple_choice.loc[:20]
    score, outs = evaluate_qa(tests, qa, verbose=False)
    save_report(outs, output_path)
    print('first authors score:', score)
    print('report stored in', output_path)


def eval_recent_articles(qa: RAGQA, multiple_choice: pd.DataFrame, output_path='report_recents.csv'):
    tests = multiple_choice.loc[20:]
    score, outs = evaluate_qa(tests, qa)
    save_report(outs, output_path)
    print('recent articles score:', score)
    print('report stored in', output_path)
   
def store_output_details(outputs: list[dict]):
    return pd.DataFrame([{
                "prompt": prompt,
                "columns": columns,
                "k": k,
                **out,
            } for out in outputs])
    
   
def run_abstract_test(
    db: pd.DataFrame, 
    tests: pd.DataFrame, 
    config: QAConfig):
    
    qa = build_rag_qa(db, config)
    score, outputs = evaluate_qa(tests, qa, verbose=False)    
    # print("abstract score", score)
    # save_report(outs, 'report_abstracts.csv')
    return {
            "prompt": config.prompt,
            "columns": config.columns,
            "k": config.k,
            'score': score
        }

def abstract_exp_configs(
    prompts = ['qa1', 'qa2'],
    columns = ['default', 'no_abstract'],
    ks = [4, 10],
):
    for prompt, cols, k in product(prompts, columns, ks):
        config = QAConfig(
            prompt=prompt,
            columns=cols,
            k=k,
        )
        yield config

def reorder_columns(report: pd.DataFrame):
    return report[['answer', 'gold', 'retrieved_ids', 'question']]
    
def abstract_experiment():
    tests = load_data('./data/questions/abstract_questions.csv')
    db = load_db('kmanpub')

    results = [
        run_abstract_test(db, tests, config)
        for config in abstract_exp_configs()
    ]

    report = pd.DataFrame(results)
    report.to_csv("qa_abstracts_report.csv", index=False)
    return report
    
    
def main():
    abstract_experiment()
    quit(0)
    
    test_path = './data/questions/multiple_choice.csv'
    tests = load_data(test_path)
    
    qa = build_rag_qa(
        db, 
        QAConfig(
            k=4, prompt='qa1', columns='no_abstract', verbose=True
        )
    )
    
    eval_first_authors(qa, tests)
    eval_recent_articles(qa, tests)
    quit(0)
    
    score, outs = evaluate_qa(tests, qa, verbose=False)
    report.to_csv('report.csv')
    
    print('score:', score)


    
def full_rag(db):
    qa = build_rag_qa(
        db,
        QAConfig(
            k=4,
            model="gemma2:2b",
            prompt="qa1",
            backend="server",
            verbose=False
        ),
    )
    return qa

def dummy_qa(db):
    qa = build_dummy_qa(db, verbose=False)
    return qa


if __name__ == "__main__":
    main()
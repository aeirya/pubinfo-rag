import os

os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from transformers.utils import logging as hf_logging
hf_logging.disable_progress_bar()
hf_logging.set_verbosity_error()


import argparse

import pandas as pd
from pubinfo.evaluate.qa import evaluate_qa
from pubinfo.dataset import load_db, load_data
from pubinfo.pipelines.qa import build_rag_qa, QAConfig, RagQA
from pubinfo.pipelines.qa import build_dummy_qa
from itertools import product

def save_report(outs: list[dict], path: str):
    # print("PRINTING OUTPUTS OF THE EVALUATE FUNCTION")
    # print(outs)
    
    report = pd.DataFrame(outs)
    # if 'RelevantIds' in tests:
    #     report['gold_relavant_ids'] = tests['RelevantIds'].to_numpy()
    report = reorder_columns(report)
    report.to_csv(path)

def eval_first_authors(qa: RagQA, multiple_choice: pd.DataFrame, output_path='report_authors.csv'):
    tests = multiple_choice.loc[:20]
    score, outs = evaluate_qa(tests, qa, verbose=False)
    save_report(outs, output_path)
    print('first authors score:', score)
    print('report stored in', output_path)


def eval_recent_articles(qa: RagQA, multiple_choice: pd.DataFrame, output_path='report_recents.csv'):
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
    
    result = {
            "prompt": config.prompt,
            "columns": config.columns,
            "k": config.k,
            'score': score,
            **config.model_args,
        }
    print(repr(result))
    return result

def model_args_list():
    return [
        {
            'model': 'gemma2:2b',
            'num_ctx': 2048,
            'num_predict': 5,
            'reasoning': False,
        },
        # {
        #     'model': 'gemma2:2b',
        #     "num_ctx": 4096,
        #     'num_predict': 128,
        #     'reasoning': False,
        # },
        # {
        #     'model': 'qwen2.5:7b',
        #     "num_ctx": 4096,
        #     'num_predict': 256,
        #     'reasoning': False
        # },
        # {
        #     'model': "qwen3:8b",
        #     'num_ctx': 2048*2,
        #     'reasoning': True,
        #     'num_predict': 256,
        # }
    ]


def limit_options():
    pass

def abstract_exp_configs(
    prompts = [
        'qa1', 'qa2'
        ],
    columns = [
        'default', 
        # 'no_abstract'
        ],
    ks = [4,
        #   10
          ],
    models = model_args_list()
):
    for model_args, prompt, cols, k in product(models, prompts, columns, ks):
        config = QAConfig(
            prompt=prompt,
            columns=cols,
            k=k,
            model_args=model_args,
        )
        yield config

def reorder_columns(report: pd.DataFrame):
    return report[['answer', 'gold', 'retrieved_ids', 'question']]
    
def abstract_experiment():
    tests = load_data('./data/questions/abstract_questions.csv')
    db = load_db('kmanpub')
    # configs = abstract_exp_configs(models=[model_args_list()[-1]])
    configs = abstract_exp_configs()
    results = [
        run_abstract_test(db, tests, config)
        for config in configs
    ]

    report = pd.DataFrame(results)
    report.to_csv("qa_abstracts_report_full_postproc", index=False)
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
            model_args={"model": "gemma2:2b"},
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
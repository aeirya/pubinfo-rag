
from itertools import product

import pandas as pd

from experiments.args import model_args_list
from pubinfo.dataset.load import load_data
from pubinfo.dataset.publication import load_db
from pubinfo.evaluate.qa import evaluate_qa
from pubinfo.pipelines.qarag.config import QAConfig
from pubinfo.pipelines.qarag.factory import build_qa_rag


def run_abstract_test(
    db: pd.DataFrame, 
    tests: pd.DataFrame, 
    config: QAConfig):
    
    qa = build_qa_rag(db, config)
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

def abstract_experiment():
    tests = load_data('./data/questions/abstract_questions.csv')
    db = load_db('kmanpub')
    configs = abstract_exp_configs()
    results = [
        run_abstract_test(db, tests, config)
        for config in configs
    ]

    report = pd.DataFrame(results)
    report.to_csv("qa_abstracts_report_full_postproc", index=False)
    return report
    
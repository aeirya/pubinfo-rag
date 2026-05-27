from pubinfo.dataset.load import load_data
from pubinfo.dataset.publication import load_db
from pubinfo.evaluate.qa import evaluate_qa
from pubinfo.pipelines.qarag.config import QAConfig
from pubinfo.pipelines.qarag.dummy_qa import build_dummy_qa
from pubinfo.pipelines.qarag.factory import build_qa_rag
from pubinfo.pipelines.qarag.rag import RAG

__all__ = [
    "load_data", "load_db", "evaluate_qa", "RAG", 
    "build_qa_rag", "QAConfig", 
    "full_rag", "dummy_qa",
    "disable_hf_log",
]

def disable_hf_log():
    import os
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    from transformers.utils import logging as hf_logging
    hf_logging.disable_progress_bar()
    hf_logging.set_verbosity_error()
    
def full_rag(db):
    qa = build_qa_rag(
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
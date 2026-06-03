# pubinfo/retrieval/cache.py

import pickle
from pathlib import Path

from langchain.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

CACHE_ROOT = Path(".cache/retrieval")


def safe_name(name: str) -> str:
    return name.replace("/", "__")


def path_for(corpus: str, kind: str, name: str | None = None) -> Path:
    path = CACHE_ROOT / corpus / kind
    if name is not None:
        path = path / safe_name(name)
    return path


def exists(path: Path) -> bool:
    return path.exists()


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_faiss(path: Path, embeddings: Embeddings):
    return FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def save_faiss(path: Path, db: FAISS):
    path.mkdir(parents=True, exist_ok=True)
    db.save_local(str(path))


def cached(path: Path, load, build, save, rebuild: bool = False):
    if path.exists() and not rebuild:
        return load()

    obj = build()
    save(obj)
    return obj
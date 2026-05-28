from dataclasses import dataclass
from typing import Literal


RetrievalKind = Literal["bm25", "dense", "hybrid", "tfidf", "semantic"]

DEFAULT_COLUMNS = [
    "title",
    "keywords",
    "abstract",
    "authors",
    "date_published",
    "context_name",
]
DEFAULT_COLUMNS_NO_ABSTRACT = [
    "title",
    "keywords",
    "authors",
    "date_published",
    "context_name",
]

COLUMN_PRESETS = {
    "default": DEFAULT_COLUMNS,
    "no_abstract": DEFAULT_COLUMNS_NO_ABSTRACT,
}

RETRIEVAL_ALIASES = {
    "tfidf": "bm25",
    "semantic": "dense",
}


@dataclass
class RetrievalConfig:
    kind: RetrievalKind
    k: int
    columns: str | list[str] | None
    bm25_k: int | None = None
    dense_k: int | None = None
    rrf_k: int | None = None
    bm25_columns: str | list[str] | None = None
    dense_columns: str | list[str] | None = None
    dense_model: str = "BAAI/bge-base-en-v1.5"


def resolve_columns(columns: str | list[str] | None) -> list[str] | None:
    if isinstance(columns, str) and columns:
        try:
            return COLUMN_PRESETS[columns]
        except KeyError as exc:
            known = ", ".join(sorted(COLUMN_PRESETS))
            raise ValueError(f"Unknown column preset {columns!r}. Known presets: {known}") from exc
    return columns


def canonical_kind(kind: str) -> str:
    return RETRIEVAL_ALIASES.get(kind, kind)

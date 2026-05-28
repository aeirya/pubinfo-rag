from dataclasses import dataclass, field
from typing import Any, Literal

from pubinfo import prompts
from pubinfo.retrieval.config import RetrievalConfig


GenerationMode = Literal["constrained", "text", "choice"]


@dataclass
class QAConfig:
    k: int
    prompt: str
    columns: str | list[str] | None
    retrieval: RetrievalConfig | None = None
    prediction_mode: GenerationMode = "choice"
    model: str | None = None
    model_args: dict[str, Any] = field(default_factory=dict)
    backend: str = "server"
    verbose: bool = False

    def generation_args(self) -> dict[str, Any]:
        args = {
            "template": prompts.load(self.prompt),
            "backend": self.backend,
            **self.model_args,
        }
        if self.model is not None:
            args["model"] = self.model
        return args

    def retrieval_config(self) -> RetrievalConfig:
        if self.retrieval is None:
            return RetrievalConfig(kind="hybrid", k=self.k, columns=self.columns)

        config = self.retrieval
        config.k = self.k
        if self.columns and config.columns in (None, "default"):
            config.columns = self.columns
        return config

from dataclasses import dataclass, field
from typing import Any

from pubinfo import template
from pubinfo.retrieval.config import RetrievalConfig
from pubinfo.typing import GenerationMode


@dataclass
class QAConfig:
    k: int = 5
    prompt: str = 'qa1'
    retrieval: RetrievalConfig | None = None
    
    columns: str = 'default'
    verbose: bool = False
    
    column_list: list[str] = field(default_factory=list)
    prediction_mode: GenerationMode = "choice"
    
    model: str | None = None
    model_args: dict[str, Any] = field(default_factory=dict)
    backend: str = 'server'

    def generation_args(self) -> dict[str, Any]:
        args = {
            "template": template.load(self.prompt),
            "backend": self.backend,
            **self.model_args,
        }
        if self.model is not None:
            args["model"] = self.model
        return args

    def retrieval_config(self) -> RetrievalConfig:
        if self.retrieval is None:
            return RetrievalConfig(k=self.k, columns=self.columns)

        config = self.retrieval
        config.k = self.k
        if self.columns and config.columns in (None, "default"):
            config.columns = self.columns
        return config

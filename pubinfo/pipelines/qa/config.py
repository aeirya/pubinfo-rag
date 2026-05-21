from dataclasses import dataclass, field
from pubinfo import template


@dataclass
class QAConfig:
    k: int = 5
    prompt: str = 'qa1'
    
    columns: str = 'default'
    verbose: bool = False
    
    column_list: list[str] = field(default_factory=list)
    prediction_mode: str = "choice"
    
    model: str | None = None
    model_args: dict[str, Any] = field(default_factory=dict)
    backend: str = 'server'
    
    def gen_kwargs(self):
        return {
            'template': template.load(self.prompt),
            'model': self.model,
            'backend': self.backend,
            **self.model_args
        }
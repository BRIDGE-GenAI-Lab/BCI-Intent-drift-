from dataclasses import dataclass, asdict, field
from typing import Optional, Literal

LABELS = ("faithful", "degraded", "drift")
Label = Literal["faithful", "degraded", "drift"]

@dataclass
class AttemptRecord:
    message_id: str
    corpus: str
    category: str
    intended_text: str
    cer_target: float
    seed: int
    source_subject: str
    noisy_text: str
    actual_cer: float
    model_id: str
    model_class: str
    depth: str
    temperature: float
    prompt_id: str
    output_text: str
    logprob: Optional[float] = None
    verbalized_conf: Optional[float] = None
    nli_label: Optional[str] = None
    judge_label: Optional[str] = None
    human_label: Optional[str] = None
    final_label: Optional[str] = None
    critical: bool = False

    def to_dict(self):
        return asdict(self)

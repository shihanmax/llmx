from typing import Optional
from dataclasses import dataclass, field


@dataclass
class GeneratingArguments:
    r""""""
    do_sample: Optional[bool] = field(
        default=True,
        metadata={"help": ""}
    )
    temperature: Optional[float] = field(
        default=0.95,
        metadata={"help": ""}
    )
    top_p: Optional[float] = field(
        default=0.7,
        metadata={"help": ""}
    )
    top_k: Optional[int] = field(
        default=30,
        metadata={"help": ""}
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={"help": ""}
    )
    max_length: Optional[int] = field(
        default=512,
        metadata={"help": ""}
    )
    max_new_tokens: Optional[int] = field(
        default=512,
        metadata={"help": ""}
    )
    repetition_penalty: Optional[float] = field(
        default=1.0,
        metadata={"help": ""}
    )
    length_penalty: Optional[float] = field(
        default=1.0,
        metadata={"help": ""}
    )


def __post_init__(self):
    if self.max_length and self.max_new_tokens:
        del self.max_length

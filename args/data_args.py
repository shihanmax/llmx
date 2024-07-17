import logging
import os
import json
from typing import List, Literal, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DataArguments:
    r""""""
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": ""}
    )
    
    split: Optional[str] = field(
        default="train",
        metadata={"help": ""}
    )
    max_seq_len: Optional[int] = field(
        default=2048,
        metadata={"help": "max input + output len"}
    )
    chat_format: Optional[str] = field(
        default=None,
        metadata={"help": "format to follow when training a multi-turn model"}
    )

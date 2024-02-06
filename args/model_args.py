import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    cache_dir: Optional[str] = field(
        default="~/.llmx_cache/",
        metadata={"help": ""}
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": ""}
    )
    use_fast_tokenizer: Optional[bool] = field(
        default=True,
        metadata={"help": ""}
    )
    split_special_tokens: Optional[bool] = field(
        default=False,
        metadata={"help": ""}
    )
    flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": ""}
    )

    s2_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "ref. https://github.com/dvlab-research/LongLORA"}
    )

    def __post_init__(self):
        if "yi-" in self.model_name_or_path. lower():
            # TODO: 需要更正式
            self.use_fast_tokenizer = False  # YI
            logger.info(
                "turned 'use_fast_tokenizer' to 'False' for Yi:"
                f"{self.model_name_or_path}"
            )

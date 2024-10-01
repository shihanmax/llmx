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
    
    rope_scaling: Optional[str] = field(
        default=None,
        metadata={"help": "rope scaling method, options: ['linear', 'dynamic'], ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/configuration_llama.py#L29"}  # noqa
    )
    
    rope_scaling_factor: Optional[float] = field(
        default=None,
        metadata={"help": "rope scaling factor"}
    )

    flash_attn: Optional[bool] = field(
        default=True,
        metadata={"help": ""}
    )

    s2_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "ref. https://github.com/dvlab-research/LongLORA"}
    )

    sequence_parallel_size: Optional[int] = field(
        default=1,
        metadata={"help": "set > 1 to enable sequence parallel for training tasks with extreme long context"}
    )

    def __post_init__(self):
        if "yi-" in self.model_name_or_path.lower():
            # TODO: 需要更正式
            self.use_fast_tokenizer = False  # YI
            logger.info(
                "turned 'use_fast_tokenizer' to 'False' for Yi:"
                f"{self.model_name_or_path}"
            )

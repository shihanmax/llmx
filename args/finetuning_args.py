import json
import logging
import os
import re
from typing import List, Literal, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FinetuningArguments:
    r""""""
    # dpo_* are for dpo training
    dpo_beta: Optional[float] = field(
        default=0.1,
        metadata={"help": "The beta factor in DPO loss. Higher beta means less divergence from the initial policy."}
    )

    dpo_loss_type: Optional[str] = field(
        default="sigmoid",
        metadata={
            "help": "The type of DPO loss to use. Either 'sigmoid' the default DPO loss"
        }
    )
    
    training_stage: Optional[str] = field(
        default="sft",
        metadata={"help": "pt, sft, dpo"}
    )
    
    parameter_mode: Optional[str] = field(
        default="lora",
        metadata={"help": "full, lora"}
    )
    
    qlora: Optional[bool] = field(
        default=False,
        metadata={"help": "enable QLoRA"}
    )

    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "QLoRA: bnb_4bit_quant_type"}
    )
    
    bnb_4bit_use_double_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "QLoRA: use_double_quant"}
    )
    
    checkpoint_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save the checkpoints"}
    )
    
    merged_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save the merged model"}
    )
    
    max_shard_size: Optional[str] = field(
        default="10GB",
        metadata={"help": "max shard size when merging peft model"}
    )
    
    # for multi-gpu inference
    device_ids: Optional[str] = field(
        default="0",
        metadata={"help": "cuda device ids: format:['0', '1-3', '1,3,5']"}
    )
    
    predict_file_path: Optional[str] = field(
        default=None,
        metadata={"help": ""}
    )
    
    predict_dry_run_size: Optional[int] = field(
        default=None,
        metadata={"help": "if test on part of the dataset"}
    )
    
    save_predict_result_to: Optional[str] = field(
        default=None,
        metadata={"help": ""}
    )
    
    per_device_predict_batch_size: Optional[int] = field(
        default=8,
        metadata={"help": ""}
    )
        
    with_debugging_model: Optional[bool] = field(
        default=False,
        metadata={"help": "Load a smaller size model for debugging"}
    )

    def __post_init__(self):
        if re.findall(r"^\d+$", self.device_ids):
            self.device_ids = [int(self.device_ids)]
        elif re.findall(r"\d+\-\d+$", self.device_ids):
            start, end = self.device_ids.split("-")
            self.device_ids = [i for i in range(int(start), int(end) + 1)]
        else:
            self.device_ids = [int(i) for i in self.device_ids.split(",")]

        # parameter checking
        if self.qlora and self.parameter_mode != "lora":
            logger.warning("qlora requres parameter_mode == 'lora'")
            self.parameter_mode = "lora"
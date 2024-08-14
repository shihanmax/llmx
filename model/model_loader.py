import logging
import math

import torch
from accelerate import Accelerator
from peft import (
    LoraConfig, TaskType, get_peft_model, PeftModel,
    prepare_model_for_kbit_training,
)

from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)


def scaling_rope(config, model_args, data_args):
    """
    ref: 
        - https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/
        - https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/model/model_utils/rope.py
    """
    if not model_args.rope_scaling:
        logger.info(f"will not scale RoPE")
        return
    
    if not hasattr(config, "rope_scaling"):
        logger.warning(f"curr model not support RoPE scaling")
        return 

    scaling_factor = 1.0
    if data_args.max_seq_len:
        # curr_max_len: max position embedding base model supports
        curr_max_len = getattr(config, "max_position_embedding", None)
        
        if curr_max_len:
            if data_args.max_seq_len > curr_max_len:
                logger.info(f"extend model max len to:{data_args.max_seq_len}")
                setattr(
                    config, "max_position_embedding", data_args.max_seq_len,
                )
                scaling_factor = float(
                    math.ceil(data_args.max_seq_len / curr_max_len)
                )
            else:
                logger.warning(
                    "curr model already support max len: "
                    f"{data_args.max_seq_len}"
                )
    setattr(
        config, "rope_scaling", 
        {"type": model_args.rope_scaling, "factor": scaling_factor},
    )
    logger.info(f"adopt scaling factor: {scaling_factor}")
    
        
class ModelLoader(object):

    @classmethod
    def patch_tokenizer(cls, tokenizer):
        """inplace op"""
        if tokenizer.eos_token_id is None:
            tokenizer.eos_token = "<|endoftext|>"
            
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    @classmethod
    def patch_config(cls, config, model_args, data_args):
        scaling_rope(config, model_args, data_args)
    
    @classmethod
    def load_config_and_tokenizer(cls, model_args, data_args, default_args):

        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **default_args
        )

        cls.patch_config(config, model_args, data_args)
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side="left",
            **default_args,
        )
        
        cls.patch_tokenizer(tokenizer)
        
        return config, tokenizer
    
    @classmethod
    def _load(
        cls, model_args, training_args, finetuning_args, data_args, peft_args,
    ):
        default_args = {
            "trust_remote_code": True,
            "cache_dir": model_args.cache_dir,
        }

        config, tokenizer = cls.load_config_and_tokenizer(
            model_args, data_args, default_args,
        )

        if finetuning_args.qlora:
            logger.info(f"qlora enabled!")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=finetuning_args.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=finetuning_args.bnb_4bit_use_double_quant,  # noqa
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                quantization_config=bnb_config,
                **default_args,
            )

            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing,
            )

        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                torch_dtype=torch.float16,
                # empty_init=False,
                device_map={"": Accelerator().process_index},  # ignored by deepspeed
                **default_args,
            )

        # patch LoRA
        if finetuning_args.parameter_mode == "lora":
            # prepare peft config
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=not training_args.do_train,
                r=peft_args.lora_rank,
                lora_alpha=peft_args.lora_alpha,
                lora_dropout=peft_args.lora_dropout,
                target_modules=peft_args.lora_target.split(","),
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return model, config, tokenizer

    @classmethod
    def load(
        cls, model_args, training_args, finetuning_args, data_args, peft_args, 
    ):
        """Prepare model and tokenizer."""

        do_train = training_args.do_train
        
        if model_args.flash_attn:    
            from ..utils.patches.llama_attention_patch import patch_llama_attn
            patch_llama_attn(
                use_flash_attn=True, use_full=True, inference=not do_train
            )
            
        if model_args.s2_attn:
            from ..utils.patches.llama_attention_patch import patch_llama_attn
            patch_llama_attn(
                use_flash_attn=True, use_full=False, inference=not do_train
            )
        
        model, config, tokenizer = cls._load(
            model_args, training_args, finetuning_args, data_args, peft_args,
        )

        if finetuning_args.training_stage == "dpo":
            # create a ref model for dpo
            ref_model, *_ = cls._load(
                model_args, training_args, finetuning_args, data_args, peft_args,
            )
        else:
            ref_model = None
    
        if not training_args.do_train:
            model.requires_grad_(False)
            
        return ref_model, model, tokenizer

    @classmethod
    def merge_adapter(cls, model_args, finetuning_args):
        default_args = {
            "trust_remote_code": True,
            "cache_dir": model_args.cache_dir,
            "device_map": "auto",
        }
        
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **default_args
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side="left",
            **default_args,
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.float16,
            config=config,
            **default_args,
        )
        
        peft_model = PeftModel.from_pretrained(
            base_model, finetuning_args.checkpoint_dir,
        )
        
        logger.info("start merging model, wait..")
        
        peft_model = peft_model.merge_and_unload()
        peft_model.save_pretrained(
            f"{finetuning_args.merged_dir}",
            max_shard_size=finetuning_args.max_shard_size,
        )
        
        tokenizer. save_pretrained(f"{finetuning_args.merged_dir}")
        logger.info(f"peft model merged & saved to {finetuning_args.merged_dir}")

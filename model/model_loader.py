import logging

import torch
from accelerate import Accelerator
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def patch_tokenizer(tokenizer):
    """inplace op"""
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = "<|endoftext|>"
        
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def prepare_model(model_args, training_args, peft_args, finetuning_args):
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

    default_args = {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
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
    
    patch_tokenizer(tokenizer)
    
    if finetuning_args.training_stage == "dpo":
        # create a ref model for dpo
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            torch_dtype=torch.float16,
            device_map={"": Accelerator().process_index},
            **default_args,
        )
    else:
        ref_model = None
        
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch.float16,
        device_map={"": Accelerator().process_index},
        **default_args,
    )
    
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
        
    elif finetuning_args.parameter_mode == "freeze":
        # Todo: implement
        pass
    
    if not training_args.do_train:
        model.requires_grad_(False)
        
    return ref_model, model, tokenizer


def merge_adapter(model_args, finetuning_args):
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

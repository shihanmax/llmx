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
from transformers.utils.import_utils import is_flash_attn_2_available

from ..utils.patches.rope_patch.patch import adopt_rope_config
from ..utils.patches.rope_patch.qwen2_rope_patch import patch_qwen2_rope_scaling
from ..utils.patches.sequence_parallel import patch_attention_for_qwen2

logger = logging.getLogger(__name__)


class ModelLoader(object):

    @classmethod
    def patch_tokenizer(cls, tokenizer):
        """inplace op"""
        logger.info("patching tokenizer..")
        if tokenizer.eos_token_id is None:
            tokenizer.eos_token = "<|endoftext|>"
            
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    @classmethod
    def patch_config(cls, config, model_args, data_args):
        logger.info("patching config..")

        # patch RoPE scaling
        adopt_rope_config(config, model_args, data_args)

        # patch fa2 TODO: 更合理的 FA2设置方式
        if hasattr(config, "_attn_implementation"):
            setattr(config, "_attn_implementation", "flash_attention_2")

    @classmethod
    def patch_model(cls, model, config, model_args, training_args):
        logger.info("patching model..")
        
        # patch rope scaling
        # patch_qwen2_rope_scaling(model, config)  # inplace op

        # patch attention forward for sequence parallel
        if model_args.sequence_parallel_size > 1:  
            logger.info(f"patching model for sequence parallel")
            patch_attention_for_qwen2(model)

        if training_args.gradient_checkpointing:
            # enable gradient checkpointing
            model.gradient_checkpointing_enable()
            logger.info(f"gradient checkpointing enabled")

        return model

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
        
        logger.info("config and tokenizer loaded")
        return config, tokenizer
    
    @classmethod
    def load_model(
        cls, config, model_args, training_args, finetuning_args, peft_args,
        default_args,
    ):
        logger.info(f"loading model..")

        if finetuning_args.qlora:
            logger.info("qlora enabled!")
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

            model = cls.patch_model(model, config, model_args, training_args)
            model = prepare_model_for_kbit_training(
                model, 
                use_gradient_checkpointing=training_args.gradient_checkpointing,  # noqa
            )

        else:
            device_map = {"": Accelerator().process_index} if training_args.deepspeed is None \
                else None

            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                torch_dtype=torch.float16,
                device_map=device_map,
                low_cpu_mem_usage=False,  # TODO: zero3时设置为 False，同时删去 device_map参数/或者设置为 None
                **default_args,
            )

            model = cls.patch_model(model, config, model_args, training_args)

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

        return model

    @classmethod
    def load_model_for_debugging(cls, model_args, data_args, training_args):
        logger.debug(f"Qwen2 model for DEBUGGING loaded!")

        from transformers import Qwen2ForCausalLM, Qwen2Config

        conf = Qwen2Config(
            hidden_size=160,
            intermediate_size=100,
            num_hidden_layers=1,
            num_attention_heads=8,
            num_key_value_heads=1,
            hidden_act="silu",
            max_position_embeddings=8192,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            use_sliding_window=False,
            max_window_layers=1,
            attention_dropout=0.0,
            rope_scaling=None,
            _attn_implementation="flash_attention_2",  # flash_attention_2
            torch_dtype=torch.float16,
        )

        cls.patch_config(conf, model_args, data_args)
        # conf.torch_dtype = torch.float16

        model = Qwen2ForCausalLM(conf)
        model = cls.patch_model(model, conf, model_args, training_args)

        return model

    @classmethod
    def load(
        cls, model_args, training_args, finetuning_args, data_args, peft_args, 
    ):
        """Prepare model and tokenizer."""

        do_train = training_args.do_train
        
        # WARN: maybe deprecated
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

        config, tokenizer = cls.load_config_and_tokenizer(
            model_args, data_args, default_args,
        )

        if finetuning_args.with_debugging_model:
            logger.debug("loading a smaller size model for debugging")
            model = cls.load_model_for_debugging(model_args, data_args, training_args)
        else:
            model = cls.load_model(
                config, model_args, training_args, finetuning_args, peft_args,
                default_args,
            )

        if finetuning_args.training_stage == "dpo":
            # create a ref model for dpo
            ref_model, *_ = cls.load_model(
                model_args, training_args, finetuning_args, data_args, 
                peft_args,
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

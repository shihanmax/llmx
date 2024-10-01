import os
import logging
from itertools import chain

from datasets import load_dataset, load_from_disk
from transformers import (
    DataCollatorForSeq2Seq, DataCollatorForLanguageModeling,
)
import torch.distributed as dist
from trl.trainer.utils import DPODataCollatorWithPadding

from .formatter import CHAT_FORMAT_MAPPER
from ..utils import is_rank_zero


logger = logging.getLogger(__name__)


def post_process_dataset(dataset, data_args, tokenizer, stage):
    """
    def merge_to_one_column(examples):
    Warning: not used for now
    将部分字段合并为一个新字段，用于language modeling
    ＃ TODO: 用于SFTTrainer，待实现，效果待验证
    ref.: https://github.com/hiyouga/LLaMA-Factory/issues/1017
    
    inputs = {"text": []}
    for i in range(len(examples["instruction"])):
    query = examples["instruction"][i] + examples["input"][i]
    response = examples["output"][i]
    """

    def process_fn_sft(examples, tokenizer, max_seq_len):
        """for sft"""
        chat_format = data_args.chat_format
        inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        
        for i in range(len(examples["instruction"])):
            query = examples["instruction"][i] + examples["input"][i]
            response = examples["output"][i]
            history = examples["history"][i]
            
            encoded_pairs = CHAT_FORMAT_MAPPER[chat_format].format_to_ids(
                tokenizer, max_seq_len=max_seq_len, query=query, 
                history=history, response=response,
            )
            
            for (query_ids, response_ids) in encoded_pairs:
                inputs["input_ids"].append(query_ids)
                inputs["attention_mask"].append([1] * len(query_ids))
                inputs["labels"]. append(response_ids)

        return inputs

    def process_fn_pt(examples, tokenizer, max_seq_len):
        """for pre-training.
        We suggest to store the original sequence in a field named "raw".
        """
        # TODO: support tokenizer for other models like llama_factory does
        tokenizer_args = {"add_special_tokens": True}
        inputs = {"input_ids": []}
        
        concat_input_ids = []
        concat_length = 0
        
        for i in range(len(examples["raw"])):
            raw_sequence = examples["raw"][i]
            curr_input_ids = tokenizer.encode(raw_sequence, **tokenizer_args)
            concat_length += len(curr_input_ids)
            concat_input_ids.append(curr_input_ids)
            
        concat_input_ids = chain(*concat_input_ids)
        num_samples = concat_length // max_seq_len  # drop the last chunk
        
        for i in range(num_samples):
            curr_sample_input_ids = [
                next(concat_input_ids) for _ in range(max_seq_len)
            ]
            inputs["input_ids"].append(curr_sample_input_ids)
            
        return inputs
    
    def process_fn_dpo(examples, tokenizer, max_seq_len):
        chat_format = data_args.chat_format
        
        inputs = {
            "chosen_input_ids": [],
            "chosen_attention_mask": [],
            "chosen_labels": [],
            "rejected_input_ids": [],
            "rejected_attention_mask": [],
            "rejected_labels": [],
        }
        
        for resp_key in ["chosen", "rejected"]:
            for i in range(len(examples["instruction"])):
                query = examples["instruction"][i] + examples["input"][i]
                response = examples[resp_key][i]
                history = examples["history"][i]
                
                encoded_pairs = CHAT_FORMAT_MAPPER[chat_format].format_to_ids(
                    tokenizer, max_seq_len=max_seq_len, query=query, 
                    history=history, response=response,
                )
                
                for (query_ids, response_ids) in encoded_pairs:
                    length = len(query_ids)
                    inputs[f"{resp_key}_input_ids"].append(query_ids)
                    inputs[f"{resp_key}_attention_mask"].append([1] * length)
                    inputs[f"{resp_key}_labels"].append(response_ids)
                    
        return inputs
    
    if stage == "sft":
        dataset = dataset.map(
            process_fn_sft, batched=True,
            remove_columns=["instruction", "input", "output", "history"],
            fn_kwargs={
                "tokenizer": tokenizer, "max_seq_len": data_args.max_seq_len,
            }
        )
        
    elif stage == "pt":
        dataset = dataset.map(
            process_fn_pt, batched=True,
            remove_columns=["raw"],
            fn_kwargs={
                "tokenizer": tokenizer, "max_seq_len": data_args.max_seq_len,
            }
        )
        
    elif stage == "dpo":
        dataset = dataset.map(
            process_fn_dpo, batched=True,
            remove_columns=[
                "instruction", "input", "chosen", "rejected", "history",
            ],
            fn_kwargs={
                "tokenizer": tokenizer, "max_seq_len": data_args.max_seq_len,
            }
        )
        
    else:
        raise Exception(f"Unknown stage {stage}")
    
    return dataset


def prepare_data(model_args, data_args, finetuning_args, tokenizer):
    """Prepare dataset, collator for sft / pt / dpo"""

    if True:
        stage = finetuning_args.training_stage
        assert stage in ["sft", "pt", "dpo"], f"Invalid training stage: {stage}"
        
        base_dir = os.path.abspath(os.path.dirname(__file__))
        dataset_path = os.path.join(
            base_dir, "../resource/data", data_args.dataset_name,
        )
        
        if not os.path.exists(dataset_path):
            raise Exception(
                f"there exists no dataset named {data_args.dataset_name} at "
                f"{dataset_path}"
            )
        
        logger.info(f"loading dataset from {dataset_path} ..")

        train_dataset = load_dataset(
            path=dataset_path,
            name=data_args.dataset_name,
            split=data_args.split,
            cache_dir=model_args.cache_dir,
        )
        
        train_dataset = post_process_dataset(
            train_dataset, data_args, tokenizer, stage=stage,
        )

        # pad for sequence parallel
        if model_args.sequence_parallel_size > 1:
            pad_to_multiple_of = model_args.sequence_parallel_size
        else:
            pad_to_multiple_of = None

        if stage == "sft":
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                label_pad_token_id=-100,
                pad_to_multiple_of=pad_to_multiple_of,
            )
            
        elif stage == "pt":
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                pad_to_multiple_of=pad_to_multiple_of,
            )
            
        elif stage == "dpo":
            # note: DPODataCollatorWithPadding does not (or not support) 
            # pad_to_multiple_of: 
            # https://github.com/huggingface/trl/blob/0cda2f2f019a8e1f7d97742fc3e5634a7a727791/trl/trainer/utils.py#L359
            data_collator = DPODataCollatorWithPadding()
            
        else:
            raise Exception(f"Invalid training stage: {stage}")
            
        data = [train_dataset, None, data_collator]
    # else:
    #     data = [None, None, None]
    
    # if dist.is_initialized():
    #     dist.barrier()
    #     dist.broadcast_object_list(data, src=0)
        
    return data

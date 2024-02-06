from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from .data_args import DataArguments
from .finetuning_args import FinetuningArguments
from .generating_args import GeneratingArguments
from .model_args import ModelArguments
from .peft_args import PeftArguments


def parse_args(on_train=True):
    arg_classes = [
        DataArguments, FinetuningArguments, GeneratingArguments,
        ModelArguments, PeftArguments,
    ]
    
    if on_train:
        arg_classes.insert(0, Seq2SeqTrainingArguments)
        
    parser = HfArgumentParser(arg_classes)

    # TODO: yaml, json supports
    return parser.parse_args_into_dataclasses()


def parse_args_for_merge_adapter():
    """for merging peft models"""
    arg_classes = [FinetuningArguments, ModelArguments]
    parser = HfArgumentParser(arg_classes)
    
    return parser.parse_args_into_dataclasses()
    
    
def parse_args_for_batch_inference():
    """for batch inference"""
    arg_classes = [
        FinetuningArguments, ModelArguments, GeneratingArguments, 
        DataArguments,
    ]
    
    parser = HfArgumentParser(arg_classes)
    return parser.parse_args_into_dataclasses()

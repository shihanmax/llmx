import logging
import sys
sys.path.append("../..")
from llmx.args.parser import parse_args_for_merge_adapter
from llmx.model.model_loader import merge_adapter


logging. basicConfig(level=logging.INFO)
finetuning_args, model_args = parse_args_for_merge_adapter()
merge_adapter(model_args, finetuning_args)

import logging
import sys

sys.path.append("../..")
from llmx.args.parser import parse_args_for_batch_inference
from llmx.inference.inference import batch_inference

logging. basicConfig(level=logging.INFO)


if __name__ == "__main__":
    # mp.set start method( 'spawn')
    finetuning_args, model_args, generating_args, data_args = (
        parse_args_for_batch_inference()
    )
    batch_inference(finetuning_args, model_args, generating_args, data_args)

import logging
import sys
from dataclasses import asdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.append("../..")

from llmx.args.parser import parse_args_for_batch_inference
from llmx.data.formatter import CHAT_FORMAT_MAPPER
from llmx.model.model_loader import ModelLoader

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# TODO: 合并至ModelLoader
def prepare_model_and_tokenizer(model_args, device):
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        trust_remote_code=True,
    )

    ModelLoader.patch_tokenizer(tokenizer)

    model.half()
    model.eval()
    logger.info("model and tokenizer loaded")
    model.to(device)
    
    return model, tokenizer


def predict_single(
    data_args, generating_args, tokenizer, model, input_, history, device,
):
    chat_format = data_args.chat_format
    
    id_pairs = CHAT_FORMAT_MAPPER[chat_format].format_to_ids(
        tokenizer, max_seq_len=data_args.max_seq_len,
        idx=0, query=input_, history=history,
        response="",  # set response to ""
    )
    print(f"id_pairs: {id_pairs}")

    input_ids, _ = id_pairs[-1]  # 取最后一对query-resp pair
    curr_batch_input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(curr_batch_input_ids)
    
    print(f"curr_batch_input_ids: {curr_batch_input_ids}")
    
    generate_args = {
        **asdict(generating_args),
        "eos_token_id": [tokenizer.eos_token_id],
        "pad_token_id": tokenizer.pad_token_id,
    }
    print(tokenizer.eos_token_id)
    print(tokenizer.eos_token)
    if generate_args.get("max_new_tokens", 0) > 0 and \
            generate_args.get("max_length", 0) > 0:
        generate_args.pop("max_length")
     
    logits = model.generate(
        input_ids=curr_batch_input_ids, attention_mask=attention_mask,
        **generate_args
    )
    print(f"logits: {logits}")

    answer = tokenizer.batch_decode(logits.cpu(), skip_special_tokens=True)[0]
    print(f"answer: {answer}")
    input_tokens = tokenizer.decode(
        curr_batch_input_ids[0], skip_special_tokens=True,
    )
    print(f"input_tokens : {input_tokens}")

    actual_answer = answer[len(input_tokens):].strip()
    print(f"actual answer: {actual_answer}")
    history.append([input_, actual_answer])
    print(f"history: {history}")
    
    return actual_answer, history


if __name__ == "__main__":
    # mp.set_start_method('spawn')
    device = torch.device("cuda:0")
    finetuning_args, model_args, generating_args, data_args = \
        parse_args_for_batch_inference()
        
    model, tokenizer = prepare_model_and_tokenizer(model_args, device)

    history = []  # tmp
    
    while True:
        instruction = ""
        try:
            # input_bytes = sys.stdin.buffer.readline()
            # input_= input_bytes.decode('utf-8', 'replace').strip()
            input_ = input("> ").strip()
        except:
            print("illegal input, try again")
            continue
        
        if input_ == "bye":
            print("bye")
            break
        
        if input_ == "cls":
            history.clear()
            print("history cleared")
            continue
        
        actual_answer, history = predict_single(
            data_args, generating_args, tokenizer, model, input_, history, 
            device,
        )
        
        print(f"\n{actual_answer}\n\n")
        # print(f"history: {history}\n\n")

import logging
import os
import sys
import time

from dataclasses import asdict
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append("../..")
from llmx.data.formatter import CHAT_FORMAT_MAPPER
from llmx.utils.utils import load_json, save_json

logger = logging.getLogger(__name__)


def encode_one_data(data, tokenizer, data_args):
    instruction = data.get("instruction")
    input_ = data.get("input")
    history = data.get("history")
    
    chat_format = data_args.chat_format
    
    id_pairs = CHAT_FORMAT_MAPPER[chat_format].format_to_ids(
        tokenizer, max_seq_len=data_args.max_seq_len,
        idx=0, query=instruction + input_, history=history,
        response="",  # set response to "" on inference mode
    )
    input_ids, _ = id_pairs[-1]  # 取最后一对query-resp pair
    
    return input_ids


def encode_multi_data(data_list, tokenizer, data_args):
    if not data_list:
        raise Exception("invalid empty`data_list`!")
    
    batch_input_ids = []
    max_len = 0
    
    for data in data_list:
        input_ids = encode_one_data(data, tokenizer, data_args)
        batch_input_ids.append(input_ids)
        max_len = max(max_len, len(input_ids))
        
    assert max_len > 0, "data list contains no valid sequence"
        
    for idx, input_ids in enumerate(batch_input_ids):
        padding = [tokenizer.pad_token_id] * (max_len - len(input_ids))
        batch_input_ids[idx] = padding + input_ids
    return batch_input_ids


def _encode_and_predict(
    device_id, queue, result_queue, model, tokenizer, generating_args,
):
    """worker"""
    device = torch.device(f"cuda:{device_id}")
    model.to(device)
    
    while True:
        idx, total_batch_num, curr_batch_input_ids = queue.get()
        
        if idx is None:
            logging.info(f"device_id: {device_id} done and exiting now")
            break
        
        curr_batch_input_ids = torch.tensor(
            curr_batch_input_ids, device=device,
        )
        
        logits = model.generate(
            input_ids=curr_batch_input_ids, **generating_args
        )
        
        answers = tokenizer.batch_decode(
            logits.cpu(), skip_special_tokens=True,
        )
        
        final_answers = []
        for curr_answer, curr_input_ids in zip(answers, curr_batch_input_ids):
            # for debugging
            input_tokens = tokenizer.decode(
                curr_input_ids, skip_special_tokens=True,
            )
            actual_answer = curr_answer[len(input_tokens):].strip()
            final_answers.append(actual_answer)
            
        result_queue.put((idx, final_answers))
        result_queue_size = result_queue.qsize()
        
        percent = f"{round(result_queue_size / total_batch_num * 100, 1)}"
        print(
            f"batch %{percent}({result_queue_size} / {total_batch_num})",
            end="\r"
        )


def batch_inference(finetuning_args, model_args, generating_args, data_args):
    # prepare model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        trust_remote_code=True,
    )
    
    model.half()
    model.eval()
    logger.info("model and tokenizer loaded")
    
    device_ids = finetuning_args.device_ids
    logger.info(f"will inference on device: {device_ids}")
    
    # prepare data
    data_to_predict = load_json(finetuning_args.predict_file_path)
    
    if finetuning_args.predict_dry_run_size:
        # dry run
        logger.info("dry run mode on")
        data_to_predict = data_to_predict[
            :finetuning_args.predict_dry_run_size
        ]
        
    num_predict_data = len(data_to_predict)
    logger.info(f"will predict: {num_predict_data} data")
    time_start = time.time()
    
    queue = mp.Queue()
    # create result queue with Manager to avoid dead-lock
    result_queue = mp.Manager().Queue()  
    processes = []
    
    generate_args = {
        **asdict(generating_args),
        "eos_token_id": [tokenizer.eos_token_id],
        "pad_token_id": tokenizer.pad_token_id,
    }
    
    if generate_args.get("max_new_tokens", 0) > 0 and \
            generate_args.get("max_length", 0) > 0:
        generate_args.pop("max_length")
        
    batch_size = finetuning_args.per_device_predict_batch_size
    total_batch_num = (num_predict_data // batch_size) + 1
    
    # TODO: add support
    if total_batch_num < len(device_ids):
        raise Exception(
            f"total batch num {total_batch_num} < device num, "
            "will use part of them"
        )
        
    for device_id in device_ids:
        # https://huggingface.co/docs/diffusers/training/distributed_inference
        p = mp.Process(
            target=_encode_and_predict,
            args=(
                device_id, queue, result_queue, model, tokenizer,
                generate_args,
            ),
        )
        
        p.start()
        processes.append(p)
        logger.info(f"process: {p} started")
        
    # note: batch num < gpu num may cause bug
    logging.info(f"start inference, total batch: {total_batch_num}, wait..")

    for batch_idx in range(total_batch_num):
        start = batch_idx * batch_size
        end = start + batch_size
        data_curr_batch = data_to_predict[start: end]
        
        if data_curr_batch:
            curr_batch_input_ids = encode_multi_data(
                data_curr_batch, tokenizer, data_args,
            )
            queue.put((batch_idx, total_batch_num, curr_batch_input_ids))
            
    # send signal to break the loop
    for _ in device_ids:
        queue.put((None, None, None))
        
    for p in processes:
        p.join()  # wait for all subprocess end
        
    logger.info("done batch inference, start merge results...")
    
    merged_answers = []
    # all answers were put into the queue
    while not result_queue.empty():
        merged_answers.append(result_queue.get())  # (idx, final_ans)
        
    merged_answers = sorted(merged_answers, key=lambda x: x[0])
    
    # merge all batch answers
    flatten_answers = []
    for i in merged_answers:
        flatten_answers.extend(i[1])
        
    assert len(flatten_answers) == num_predict_data, \
        f"size mismatch!{len(merged_answers)}: {num_predict_data}"
        
    time_end = time.time()
    cost = time_end - time_start
    avg_cost_per_data = cost / num_predict_data
    
    logger.info(
        f"run {num_predict_data} instance cost {round(cost, 3)}s,"
        f"avg {round(avg_cost_per_data, 3)} s per instance"
    )
    
    if finetuning_args.save_predict_result_to:
        answers = []
        
        for pred, gold in zip(flatten_answers, data_to_predict):
            answers.append(
                {"pred": pred, "gold": gold.get("output", "")}
            )
            
        save_to_dir = os.path.dirname(finetuning_args.save_predict_result_to)
        if not os.path.exists(save_to_dir):
            os.makedirs(save_to_dir)
            
        save_json(answers, finetuning_args.save_predict_result_to)
        logger.info(
            f"predictions saved to: {finetuning_args.save_predict_result_to}"
        )

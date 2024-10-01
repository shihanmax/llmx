# https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#command-line-arguments-for-the-server
current_date=$(date +%Y%m%d)
# /ptm/Qwen2_5-14B-Instruct

export CUDA_VISIABLE_DEVICES=0,1 && python -m vllm.entrypoints.openai.api_server \
    --model /experiment/llmx/debugging/0927_risk_qwen25-14B_full_maxlen128k-32Cd-nrank-0/checkpoint-200 \
    --served-model-name qwen2_5_14B_sft \
    --dtype=auto \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.95 \
    --max-log-len 100 \
    --port 9800 \
    --enforce-eager \
    --tensor-parallel-size 2
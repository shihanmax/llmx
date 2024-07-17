export CUDA_VISIBLE_DEVICES=0

python ../runner/run_merge_adapter.py \
    --model_name_or_path /path/to/chatalm2-6b \
    --checkpoint_dir /path/to/ckpt/checkpoint-140 \
    --merged_dir /path/to/merged
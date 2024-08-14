python ../task/run_inference.py \
    --model_name_or_path path/to/merged_model \
    --device_ids "0-8" \
    --chat_format baichuan2 \
    --predict_file_path path/to/predict_data.json \
    --save_predict_result_to ../debugging/baichuan2.json \
    --per_device_predict_batch_size 12 \
    --predict_dry_run_size 0
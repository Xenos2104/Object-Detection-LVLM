accelerate launch \
    --num_processes 2 \
    --main_process_port 25001 \
    --config_file config/deepspeed_bf16_zero2.yaml \
    sft.py \
    --config config/Qwen2_5-VL-3B-Instruct.yaml

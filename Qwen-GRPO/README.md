## cuda version 
uv add huggingface_hub
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv add trl[vllm] 

## launch reward model
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --host 127.0.0.2 --model Qwen/Qwen3-14B --dtype bfloat16 --api-key empty 

## launch training 
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc-per-node=1 train_grpo_distributed.py  --config_file grpo_config.json 
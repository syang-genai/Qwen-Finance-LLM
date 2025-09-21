# create enviroment
uv init 
uv venv 
source .venv/bin/activate

# install package
uv add huggingface_hub
uv add accelerate
uv add vllm

# fixed bug
sudo apt install build-essential python3.10-dev

# start vllm service for evaluation
## SFT 
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --host 127.0.0.2 --model Qwen/Qwen3-8B --dtype bfloat16 --enable-lora --lora-modules  qlora_adapter=../Qwen-SFT/Qwen-OutputDir/SFT/checkpoint-256 --max-model-len 4096
## GRPO 
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --host 127.0.0.2 --model Qwen/Qwen3-8B --dtype bfloat16 --enable-lora --lora-modules  qlora_adapter=/root/qwen-finance-llm/Qwen-GRPO/Qwen-OutputDir/GRPO/GRPO-neg-5/checkpoint-12  --max-model-len 4096

# start vllm service for reward function
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --host 127.0.0.2 --model Qwen/Qwen3-14B --dtype bfloat16 --api-key empty 

# start vllm service for agent
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --host 127.0.0.2 --model Qwen/Qwen3-8B --dtype bfloat16 --enable-lora --lora-modules  qlora_adapter=/root/qwen-finance-llm/Qwen-GRPO/Qwen-OutputDir/GRPO/GRPO-neg-5/checkpoint-12  --max-model-len 4096 --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 
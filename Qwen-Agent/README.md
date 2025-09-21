Enviroment
---
* uv init agent
* cd agent
* uv venv
* source .venv/bin/activate
* uv add google-adk litellm
* uv add vllm

Google ADK
---
Start Agent: uv run adk web --port 8080

Local vLLM Host LLM Model Debug
---
1. Load local vLLM host LLM model through LiteLlm.

CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --host 127.0.0.2 --model Qwen/Qwen3-8B --dtype bfloat16 --enable-lora --lora-modules  qlora_adapter=/root/qwen-finance-llm/Qwen-GRPO/Qwen-OutputDir/GRPO/GRPO-neg-5/checkpoint-12  --max-model-len 4096 --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 

2. Provide the following to LiteLlm:  
* Endpoint URL provided by your vLLM deployment  
  api_base_url = "http://0.0.0.0:8000/v1"  
* Model name as recognized by *your* vLLM endpoint configuration, and add **'hosted_vllm/'** before vLLM model id.  
  model_name_at_endpoint = "hosted_vllm/Qwen/Qwen3-8B" 
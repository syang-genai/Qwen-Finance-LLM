# Qwen-Finance-LLM
#### Description
Developed an intelligent financial agent leveraging the Qwen3-8B large language model to deliver advanced financial knowledge Q&A, investment analysis, instantaneous stock lookups, and real-time financial news insights. Implemented Supervised Fine-Tuning (SFT) under constrained GPU environments using QLoRA combined with FlashAttention, achieving more than an 80% reduction in memory utilization while doubling token throughput. Further enhanced model alignment through GRPO-based reinforcement learning, utilizing the Qwen3-14B model as a reward model (deployed via vLLM), resulting in a 13% performance uplift on financial domain benchmarks. Designed and deployed real-time agent capabilities by integrating the Alpha Vantage MCP Server with Google AgentADK, enabling seamless access to live market data and financial news feeds.

#### Contents
Dataset: dataset cleaning, dataset formatting and dataset mixing  
Qwen-SFT: supervised finetuning  
Qwen-GRPO: grpo reinforcement learning  
Qwen-Agent: Google ADK agent, which relies on Alpha Vantage MCP Server  
Qwen-VLLM:   
    * deploy Qwen-14B model as GRPO reward model.  
    * deploy Qwen-0.6B model as evaluation model.  
    * deploy Qwen-8B SFT and GRPO post training model for evaluation.   
    * deploy Qwen-8B post training model for Qwen-Agent.  
Qwen-Eval: EvalScope(model) & GoogleADK Evaluation(agent)

#### Google ADK Agent Demo
![Google ADK Demo](asset/google-adk.png)

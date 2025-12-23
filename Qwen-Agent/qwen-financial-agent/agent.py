import os
import sys

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
from . import prompt
from dotenv import load_dotenv

api_key = os.getenv("apikey") 

# endpoint URL provided by your vLLM deployment
api_base_url = "http://127.0.0.2:8000/v1"

# model name as recognized by *your* vLLM endpoint configuration
model_name_at_endpoint = "hosted_vllm/Qwen/Qwen3-0.6B"  # "hosted_vllm/Qwen/Qwen3-8B" 

MODEL=LiteLlm(
        model=model_name_at_endpoint,
        api_base=api_base_url,
    )


root_agent = LlmAgent(
    name="qwen_financial_agent",
    model=MODEL,
    description=(
        "Navigate the world of finance with confidence. This agent is designed to be your all-in-one partner for managing and understanding your money. From real-time market updates to complex financial analysis, it provides the tools and insights you need to make informed decisions."
    ),
    instruction=prompt.FINANCIAL_ASSISTANT_PROMPT,
    output_key="assistant_output",
    tools=[MCPToolset(connection_params=StreamableHTTPConnectionParams(url=f"https://mcp.alphavantage.co/mcp?apikey={api_key}"))]
)

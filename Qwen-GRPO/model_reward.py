import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


device="cuda"
reward_model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
model = AutoModelForCausalLM.from_pretrained(reward_model_name, device_map="auto")
model.eval() 


def get_llm_judgment_reward(
    first: str,
    second: str,
    llm_model, # Your general purpose LLM
    llm_tokenizer,
    device
) -> float:
    """
        Uses a general-purpose LLM to 'judge' a response and extracts a score.
        This is more complex as it involves prompt engineering for the LLM and parsing its output.
    """
    
    evaluation_prompt = f"""
        Please rate the similarity between the following two sentense.
        First sentense: {first}
        Second sentense: {second}
        Answer the question with Rating (1-5).
        Rating:
    """
    
    # Apply chat template if your LLM supports it (e.g., Llama, Mistral)
    messages = [{"role": "user", "content": evaluation_prompt}]
    encoded_input = llm_tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True, enable_thinking=False).to(device)

    with torch.no_grad():
        # Generate a short response (e.g., "5" or "Rating: 4")
        outputs = llm_model.generate(
            encoded_input,
            max_new_tokens=100, # Expecting a short rating
            num_return_sequences=1,
            do_sample=False
        )
        generated_text = llm_tokenizer.decode(outputs[0][encoded_input.shape[1]:], skip_special_tokens=True).strip()
    
    
    pattern = r"[-+]?\d*\.\d+|\d+"
    match = re.findall(pattern, generated_text)
    
    if len(match)>0:
        try:
            rating = float(match[0])
            if rating<1:
                return 0
            elif rating>5:
                return 5
            return rating
        except ValueError:
            return 0
    else:
        return 0


score=get_llm_judgment_reward("give me a english sentense", "give me a chinese sentense", model,tokenizer,device)
print(score)
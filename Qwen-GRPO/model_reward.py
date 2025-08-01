reward_model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
model = AutoModelForCausalLM.from_pretrained(reward_model_name)
model.eval() 


def get_llm_judgment_reward(
    first: str,
    second: str,
    llm_model, # Your general purpose LLM
    llm_tokenizer
) -> float:
    """
        Uses a general-purpose LLM to 'judge' a response and extracts a score.
        This is more complex as it involves prompt engineering for the LLM and parsing its output.
    """
    
    messages=list()
    for f, s in zip(first,second):
        evaluation_prompt = f"""
            Please rate the similarity between the following two sentense.
            First sentense: {first}
            Second sentense: {second}
            Answer the question with Rating (1-5).
            Rating:
        """
        messages.append([{"role": "user", "content": evaluation_prompt}])
    

    encoded_input = llm_tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True, enable_thinking=False)

    with torch.no_grad():
        outputs = llm_model.generate(
            encoded_input,
            max_new_tokens=100,
            num_return_sequences=1,
            do_sample=False
        )
        
        generated_text = llm_tokenizer.batch_decode(outputs[:,encoded_input.shape[1]:], skip_special_tokens=True)
        
    
    reward=list()
    pattern = r"[-+]?\d*\.\d+|\d+"
    for gt in generated_text:
        match = re.findall(pattern, gt)
        
        if len(match)>0:
            try:
                rating = float(match[0])
                # print("rating",rating)
                if rating<0:
                    reward.append(0)
                elif rating>5:
                    reward.append(5)
                else:
                    reward.append(rating)
            except ValueError:
                reward.append(0)
        else:
            reward.append(0)
    
    return reward 



score=get_llm_judgment_reward(["give me a english sentense"]*3, ["give me a chinese sentense"]*3, model,tokenizer)
print(score)
def reformat(example, tokenizer, enable_think):
    instruction=tokenizer.apply_chat_template(
        example["prompt"],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_think
    )
    
    response=tokenizer.apply_chat_template(
        example["completion"],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )
    
    instruction_ids=tokenizer(instruction,  add_special_tokens=False)
    response_ids=tokenizer(response, add_special_tokens=False)

    # create input_ids, attention and labels
    input_ids=instruction_ids["input_ids"]+response_ids["input_ids"]
    attention_mask=instruction_ids['attention_mask']+response_ids['attention_mask']
    labels=[-100]*len(instruction_ids["input_ids"])+response_ids["input_ids"]
    
    example["input_ids"]=input_ids
    example["attention_mask"]=attention_mask
    example["labels"]=labels
    return example
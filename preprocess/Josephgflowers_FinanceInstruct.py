import torch

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


def preprocess_format(example, tokenizer):
    """
        example={"system":, "user":, "assistant":}
    """
    system_prompt= "You are a financial assistant. Answer the user's question accurately but keep it brief." if example["system"]=='\n' else example["system"] 
    
    instruction=tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": example["user"]}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )
    
    response=tokenizer.apply_chat_template(
        [{"role": "assistant", "content": example["assistant"]}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )

    instruction_ids=tokenizer(instruction,  add_special_tokens=False)
    response_ids=tokenizer(response,  add_special_tokens=False)
    
    # create input_ids, attention and labels
    input_ids=instruction_ids["input_ids"]+response_ids["input_ids"]
    attention_mask=instruction_ids['attention_mask']+response_ids['attention_mask']
    labels=[-100]*len(instruction_ids["input_ids"])+response_ids["input_ids"]
    
    example["input_ids"]=input_ids
    example["attention_mask"]=attention_mask
    example["labels"]=labels

    return example
    

## use tokenizer in data collector function, dynamical padding
## data collector function

def main():
    model_name = "Qwen/Qwen3-0.6B"
    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    # data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    
    
    dataset = load_dataset("Josephgflowers/Finance-Instruct-500k",split="train")
    dataset = dataset.map(preprocess_format,fn_kwargs=dict(tokenizer=tokenizer), remove_columns=["system","user","assistant"])
    
    # save dataset
    dataset.save_to_disk("/root/llm_finetune/dataset/Josephgflowers/Finance-Instruct-500k-Formated")
    dataset=load_from_disk("/root/llm_finetune/dataset/Josephgflowers/Finance-Instruct-500k-Formated")
    
    print("first example \n", dataset[0])
    

if __name__ == "__main__":
    main()

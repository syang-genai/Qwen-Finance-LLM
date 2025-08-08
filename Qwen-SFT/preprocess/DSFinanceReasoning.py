from utils import reformat

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


def dataset_reformat(example):
    example["prompt"]=[{"role": "user", "content": example["prompt"]}]
    example["completion"]=[{"role": "assistant", "content": example["completion"]}]
    return example
    

def FinanceReasoning(count):
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")

    dataset = load_dataset("Diweanshu/Finance-Reasoning",split="train")
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(count))
    
    dataset = dataset.map(dataset_reformat, remove_columns=["system_prompt"])
    dataset = dataset.map(reformat, fn_kwargs=dict(tokenizer=tokenizer, enable_think=False), remove_columns=["prompt","completion"])

    # save dataset
    dataset.save_to_disk("../dataset/Diweanshu/Finance-Reasoning")
    return dataset
    

if __name__ == "__main__":
    FinanceReasoning(400)
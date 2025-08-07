from utils import reformat

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


def dataset_reformat(example):
    """
        example={"system":, "user":, "assistant":}
    """
    example["prompt"]=[{"role": "user", "content": example["user"]}]
    example["completion"]=[{"role": "assistant", "content": example["assistant"]}]
    return example
    

def main():
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    
    dataset = load_dataset("Josephgflowers/Finance-Instruct-500k",split="train")
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(5000))
    
    dataset = dataset.map(dataset_reformat, remove_columns=["system","user","assistant"])
    dataset = dataset.map(reformat, fn_kwargs=dict(tokenizer=tokenizer, enable_think=False), remove_columns=["prompt","completion"])
    
    # save dataset
    dataset.save_to_disk("../dataset/Josephgflowers/Finance-Instruct-500k-Formated")
    # dataset=load_from_disk("../dataset/Josephgflowers/Finance-Instruct-500k-Formated")


if __name__ == "__main__":
    main()
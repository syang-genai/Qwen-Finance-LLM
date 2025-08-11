from utils import reformat

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


def dataset_reformat(example):
    example["prompt"]=[{"role": "user", "content": example["instruction"]}]
    example["completion"]=[{"role": "assistant", "content": example["output"]}]
    return example


def HC3Instruct(count):
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")

    
    dataset = load_dataset("causal-lm/hc3-instruct",split="train")
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(count))
    
    # select prompt
    dataset = dataset.map(dataset_reformat, remove_columns=["instruction","input","output","annotator"])
    dataset = dataset.map(reformat, fn_kwargs=dict(tokenizer=tokenizer, enable_think=True), remove_columns=["prompt","completion"])
    
    # save dataset
    dataset.save_to_disk("../dataset/H3Instruct")
    return dataset


if __name__ == "__main__":
    HC3Instruct(count=2500)
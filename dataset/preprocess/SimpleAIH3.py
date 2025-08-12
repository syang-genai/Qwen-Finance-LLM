from utils import reformat

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


def train_dataset_reformat(example):
    example["prompt"]=[{"role": "user", "content": example["instruction"]}]
    example["completion"]=[{"role": "assistant", "content": example["output"]}]
    return example


def eval_dataset_reformat(example):
    example["message"]=[{"role": "user", "content": example["instruction"]}]
    example["response"]=example["output"]
    return example


def HC3Instruct(train_count, eval_count, sublist, model_name, train_save_path, eval_save_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    
    dataset = load_dataset("causal-lm/hc3-instruct",split=sublist)
    dataset = dataset.shuffle(seed=42)
    
    train_dataset = dataset.select(range(train_count))
    train_dataset = train_dataset.map(train_dataset_reformat, remove_columns=["instruction","input","output","annotator"])
    print(train_dataset[0])
    train_dataset = train_dataset.map(reformat, fn_kwargs=dict(tokenizer=tokenizer, enable_think=False), remove_columns=["prompt","completion"])
    train_dataset.save_to_disk(train_save_path)


    eval_dataset = dataset.select(range(train_count, train_count+eval_count))
    eval_dataset = eval_dataset.map(eval_dataset_reformat, remove_columns=["instruction","input","output","annotator"])
    eval_dataset.to_json(eval_save_path)
    return train_dataset


if __name__ == "__main__":
    HC3Instruct(
            train_count=2500, \
            eval_count=500, \
            sublist="train", \
            model_name="Qwen/Qwen3-0.6B", \
            train_save_path="../train_dataset/H3Instruct", \
            eval_save_path="../eval_dataset/H3Instruct/H3I.jsonl"
        )
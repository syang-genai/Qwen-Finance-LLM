from utils import reformat

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


def train_dataset_reformat(example):
    example["prompt"]=[{"role": "user", "content": example["prompt"]}]
    example["completion"]=[{"role": "assistant", "content": example["completion"]}]
    return example


def eval_dataset_reformat(example):
    example["message"]=[{"role": "user", "content": example["prompt"]}]
    example["response"]=example["completion"]
    return example


def FinanceReasoning(train_count, eval_count, sublist, model_name, train_save_path, eval_save_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")

    dataset = load_dataset("Diweanshu/Finance-Reasoning",split=sublist)
    dataset = dataset.shuffle(seed=42)


    train_dataset = dataset.select(range(train_count))
    train_dataset = train_dataset.map(train_dataset_reformat, remove_columns=["system_prompt"])
    print(train_dataset[0])
    train_dataset = train_dataset.map(reformat, fn_kwargs=dict(tokenizer=tokenizer, enable_think=False), remove_columns=["prompt","completion"])
    train_dataset.save_to_disk(train_save_path)
    
    eval_dataset = dataset.select(range(train_count,train_count+eval_count))
    eval_dataset = eval_dataset.map(eval_dataset_reformat, remove_columns=["system_prompt"])
    eval_dataset.to_json(eval_save_path)
    return train_dataset


if __name__ == "__main__":
    FinanceReasoning(train_count=400, eval_count=100, sublist="train", model_name="Qwen/Qwen3-0.6B", train_save_path="../train_dataset/Diweanshu/Finance-Reasoning", eval_save_path="../eval_dataset/Diweanshu/Finance-Reasoning/FR.jsonl")
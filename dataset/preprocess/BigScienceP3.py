from utils import reformat

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


def train_dataset_reformat(example):
    example["prompt"]=[{"role": "user", "content": example["inputs_pretokenized"]}]
    example["completion"]=[{"role": "assistant", "content": example["targets_pretokenized"]}]
    return example

def eval_dataset_reformat(example):
    example["message"]=[{"role": "user", "content": example["inputs_pretokenized"]}]
    example["response"]=example["targets_pretokenized"]
    return example

def grpo_dataset_reformat(example):
    example["prompt"]=[{"role": "user", "content": example["inputs_pretokenized"]}]
    example["completion"]=[{"role": "assistant", "content": example["targets_pretokenized"]}]
    example["reference_answer"]=[{"role": "assistant", "content": example["targets_pretokenized"]}]
    return example



def BigScience(train_count, eval_count, grpo_count, sublist, model_name, train_save_path, eval_save_path, grpo_save_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    
    dataset = load_dataset("bigscience/P3", sublist, split="train")
    dataset = dataset.shuffle(seed=42)
    
    train_dataset = dataset.select(range(train_count))
    train_dataset = train_dataset.map(train_dataset_reformat, remove_columns=["inputs","inputs_pretokenized","targets","targets_pretokenized"])
    print(train_dataset[0])
    train_dataset = train_dataset.map(reformat, fn_kwargs=dict(tokenizer=tokenizer, enable_think=False), remove_columns=["prompt","completion"])
    train_dataset.save_to_disk(train_save_path)
    
    eval_dataset = dataset.select(range(train_count,train_count+eval_count))
    eval_dataset = eval_dataset.map(eval_dataset_reformat, remove_columns=["inputs","inputs_pretokenized","targets","targets_pretokenized"])
    eval_dataset.to_json(eval_save_path)

    grpo_dataset = dataset.select(range(train_count+eval_count,train_count+eval_count+grpo_count))
    grpo_dataset = grpo_dataset.map(grpo_dataset_reformat, remove_columns=["inputs","inputs_pretokenized","targets","targets_pretokenized"])
    print("grpo dataset \n", grpo_dataset[0])
    grpo_dataset.save_to_disk(grpo_save_path)

    return train_dataset, eval_dataset, grpo_dataset


if __name__ == "__main__":
    BigScience(
        5000, 
        1000, 
        1000, 
        "adversarial_qa_dbert_based_on", 
        "Qwen/Qwen3-0.6B", 
        train_save_path="../train_dataset/BigScienceP3/QA", 
        eval_save_path="../eval_dataset/BigScienceP3/QA/BSP3_QA.jsonl",
        grpo_save_path="../grpo_dataset/BigScienceP3/QA")
   
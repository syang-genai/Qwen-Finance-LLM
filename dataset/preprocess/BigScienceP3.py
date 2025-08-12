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


def BigScience(train_count, eval_count, sublist, model_name, train_save_path="../train_dataset/BigScienceP3/QA", eval_save_path="../eval_dataset/BigScienceP3/QA"):
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
    return train_dataset


if __name__ == "__main__":
    BigScience(2500, 500, "adversarial_qa_dbert_based_on", "Qwen/Qwen3-0.6B", train_save_path="../train_dataset/BigScienceP3/QA", eval_save_path="../eval_dataset/BigScienceP3/BSP3_QA.jsonl")
    # BigScience(2500, 500, "ag_news_classify_with_choices_question_first", "Qwen/Qwen3-0.6B", train_save_path="../train_dataset/BigScienceP3/Classify", eval_save_path="../eval_dataset/BigScienceP3/Classify/BSP3_Classify.json")
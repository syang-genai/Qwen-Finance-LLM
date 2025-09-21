from utils import reformat

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


def train_dataset_reformat(example):
    """
        example={"system":, "user":, "assistant":}
    """
    example["prompt"]=[{"role": "user", "content": example["instruction"]+":"+example["input"]}]
    example["completion"]=[{"role": "assistant", "content": example["output"]}]
    return example


def eval_dataset_reformat(example):
    """
        example={"system":, "user":, "assistant":}
    """
    example["message"] = [{"role": "user", "content": example["instruction"]+":"+example["input"]}]
    example["response"] = example["output"]
    return example
    

def grpo_dataset_reformat(example):
    example["prompt"]=[{"role": "user", "content": example["instruction"]+":"+example["input"]}]
    example["completion"]=[{"role": "assistant", "content": example["output"]}]
    example["reference_answer"]=[{"role": "assistant", "content": example["output"]}]
    return example


def BloombergFinancialNews(train_count, eval_count, grpo_count, model_name, train_save_path, eval_save_path, grpo_save_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    
    dataset = load_dataset("genloop/bloomberg_financial_news_120k-alpaca",split="train")
    # dataset = dataset.shuffle(seed=42)
    print("original dataset \n", dataset[0])
    
    train_dataset = dataset.select(range(train_count))
    train_dataset = train_dataset.map(train_dataset_reformat, remove_columns=["input","instruction","output"])
    print("train dataset \n", train_dataset[0])
    train_dataset = train_dataset.map(reformat, fn_kwargs=dict(tokenizer=tokenizer, enable_think=False), remove_columns=["prompt","completion"])
    train_dataset.save_to_disk(train_save_path)

    eval_dataset = dataset.select(range(train_count, train_count+eval_count))
    eval_dataset = eval_dataset.map(eval_dataset_reformat, remove_columns=["input","instruction","output"])
    print("eval dataset \n", eval_dataset[0])
    eval_dataset.to_json(eval_save_path)
    
    grpo_dataset = dataset.select(range(train_count+eval_count,train_count+eval_count+grpo_count))
    grpo_dataset = grpo_dataset.map(grpo_dataset_reformat, remove_columns=["input","instruction","output"])
    print("grpo dataset \n", grpo_dataset[0])
    grpo_dataset.save_to_disk(grpo_save_path)

    return train_dataset, eval_dataset, grpo_dataset



if __name__ == "__main__":
    BloombergFinancialNews(
        train_count=5000, 
        eval_count=1000, 
        grpo_count=1000,  
        model_name="Qwen/Qwen3-0.6B", 
        train_save_path="../train_dataset/Genloop/BloombergFinancialNew", 
        eval_save_path="../eval_dataset/Genloop/BloombergFinancialNew/BFN.jsonl", 
        grpo_save_path="../grpo_dataset/Genloop/BloombergFinancialNew"
    )
    
    
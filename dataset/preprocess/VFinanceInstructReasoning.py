from utils import reformat

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


def train_dataset_reformat(example):
    example["prompt"]=[{"role": "user", "content": example["user"]}]
    example["completion"]=[{"role": "assistant", "content": "<think> {} </think> {}".format(example["generation"], example["assistant"])}]
    return example


def SynFinanceInstructReason(train_count, eval_count, sublist, model_name, train_save_path, eval_save_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")

    dataset = load_dataset("vamshirvk/Finance-Instruct-500k-reasoning",split=sublist)
    dataset = dataset.shuffle(seed=42)
    train_dataset=dataset.select(range(train_count))
    
    train_dataset = train_dataset.map(train_dataset_reformat, remove_columns=["system","user","assistant","generation","distilabel_metadata","model_name"])
    print(train_dataset[0])
    train_dataset = train_dataset.map(reformat, fn_kwargs=dict(tokenizer=tokenizer, enable_think=True), remove_columns=["prompt","completion"])
    train_dataset.save_to_disk(train_save_path)
    
    
    return train_dataset


if __name__ == "__main__":
    SynFinanceInstructReason(
        train_count=40, \
        eval_count=0, \
        sublist="train", \
        model_name="Qwen/Qwen3-0.6B", \
        train_save_path="../train_dataset/Vamshirvk/Finance-Instruct-500k-reasoning", \
        eval_save_path=""
        )
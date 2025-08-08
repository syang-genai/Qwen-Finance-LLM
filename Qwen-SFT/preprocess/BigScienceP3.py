from utils import reformat

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


def dataset_reformat(example):
    example["prompt"]=[{"role": "user", "content": example["inputs_pretokenized"]}]
    example["completion"]=[{"role": "assistant", "content": example["targets_pretokenized"]}]
    return example


def BigScience(qa_count,classify_count):
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")

    
    dataset = load_dataset("bigscience/P3",  "adversarial_qa_dbert_based_on", split="train")
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(qa_count))
    
    # select prompt
    dataset = dataset.map(dataset_reformat, remove_columns=["inputs","inputs_pretokenized","targets","targets_pretokenized"])
    dataset = dataset.map(reformat, fn_kwargs=dict(tokenizer=tokenizer, enable_think=True), remove_columns=["prompt","completion"])
    
    # save dataset
    dataset.save_to_disk("../dataset/BigScienceP3/QA")


    dataset = load_dataset("bigscience/P3",  "ag_news_classify_with_choices_question_first", split="train")
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(classify_count))
    
    # select prompt
    dataset = dataset.map(dataset_reformat, remove_columns=["inputs","inputs_pretokenized","targets","targets_pretokenized","answer_choices"])
    dataset = dataset.map(reformat, fn_kwargs=dict(tokenizer=tokenizer, enable_think=True), remove_columns=["prompt","completion"])
    
    # save dataset
    dataset.save_to_disk("../dataset/BigScienceP3/Classify")
    return dataset

if __name__ == "__main__":
    BigScience(2500, 2500)
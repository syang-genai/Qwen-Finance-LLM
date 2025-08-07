from utils import reformat

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


def dataset_reformat(example):
    example["prompt"]=[{"role": "user", "content": example["user"]}]
    example["completion"]=[{"role": "assistant", "content": "<think> {} </think> {}".format(example["generation"],example["assistant"])}]
    return example


def main():
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")

    dataset = load_dataset("vamshirvk/Finance-Instruct-500k-reasoning",split="train")
    dataset = dataset.shuffle(seed=42)
    
    dataset = dataset.map(dataset_reformat, remove_columns=["system","user","assistant","generation","distilabel_metadata","model_name"])
    print(dataset[0])

    dataset = dataset.map(reformat, fn_kwargs=dict(tokenizer=tokenizer, enable_think=True), remove_columns=["prompt","completion"])
    
    # save dataset
    dataset.save_to_disk("../dataset/Vamshirvk/Finance-Instruct-500k-reasoning")
    dataset=load_from_disk("../dataset/Vamshirvk/Finance-Instruct-500k-reasoning")
    print("first example \n", dataset[0])

if __name__ == "__main__":
    main()
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


def dataset_reformat(example):
    example["prompt"]=[{"role": "user", "content": example["prompt"]}]
    example["completion"]=[{"role": "assistant", "content": example["completion"]}]
    return example
    

def main():
    dataset = load_dataset("Diweanshu/Finance-Reasoning",split="train")
    dataset = dataset.map(dataset_reformat, remove_columns=["system_prompt"])
    
    # save dataset
    dataset.save_to_disk("/root/Qwen-Finance-LLM/dataset/Diweanshu/Finance-Reasoning")
    dataset=load_from_disk("/root/Qwen-Finance-LLM/dataset/Diweanshu/Finance-Reasoning")
    print("first example \n", dataset[:2])


if __name__ == "__main__":
    main()
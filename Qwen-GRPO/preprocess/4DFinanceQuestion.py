import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


def dataset_reformat(example):
    example["prompt"]=[{"role": "user", "content": example["instruction"]}]
    example["completion"]=[{"role": "assistant", "content": example["output"]}]
    return example
    

def main():
    dataset = load_dataset("4DR1455/finance_questions",split="train")
    dataset = dataset.map(dataset_reformat, remove_columns=["instruction","output","input"])
    
    # save dataset
    dataset.save_to_disk("/root/Qwen-Finance-LLM/dataset/4D/FinanceQuestion")
    dataset=load_from_disk("/root/Qwen-Finance-LLM/dataset/4D/FinanceQuestion")
    print("first example \n", dataset[:2])


if __name__ == "__main__":
    main()
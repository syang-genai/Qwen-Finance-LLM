import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


def dataset_reformat(example):
    example["prompt"]=[{"role": "user", "content": example["question"]}]
    example["completion"]=[{"role": "assistant", "content": example["responses"][0]["response"]}]
    return example
    

def main():
    dataset = load_dataset("neoyipeng/natural_reasoning_finance",split="train")
    dataset = dataset.map(dataset_reformat, remove_columns=["question","responses"])
    
    # save dataset
    dataset.save_to_disk("/root/Qwen-Finance-LLM/dataset/NeoYiPeng/NaturalReasoningFinance")
    dataset=load_from_disk("/root/Qwen-Finance-LLM/dataset/NeoYiPeng/NaturalReasoningFinance")
    print("first example \n", dataset[:2])


if __name__ == "__main__":
    main()
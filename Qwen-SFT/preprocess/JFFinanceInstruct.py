import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


def dataset_reformat(example):
    """
        example={"system":, "user":, "assistant":}
    """
    example["prompt"]=[{"role": "user", "content": example["user"]}]
    example["completion"]=[{"role": "assistant", "content": example["assistant"]}]
    return example
    

def main():
    dataset = load_dataset("Josephgflowers/Finance-Instruct-500k",split="train")
    dataset = dataset.map(dataset_reformat, remove_columns=["system","user","assistant"])
    
    
    # save dataset
    dataset.save_to_disk("../dataset/Josephgflowers/Finance-Instruct-500k-Formated")
    dataset=load_from_disk("../dataset/Josephgflowers/Finance-Instruct-500k-Formated")


if __name__ == "__main__":
    main()
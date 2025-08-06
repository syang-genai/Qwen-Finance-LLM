import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


def dataset_reformat(example):
    example["prompt"]=[{"role": "user", "content": example["user"]}]
    example["completion"]=[{"role": "assistant", "content": "<think> {} </think> {}".format(example["generation"],example["assistant"])}]
    return example


def main():
    dataset = load_dataset("vamshirvk/Finance-Instruct-500k-reasoning",split="train")
    dataset = dataset.map(dataset_reformat, remove_columns=["system","user","assistant","generation","distilabel_metadata","model_name"])
    
    # save dataset
    dataset.save_to_disk("/root/Qwen-Finance-LLM/dataset/Vamshirvk/Finance-Instruct-500k-reasoning")
    dataset=load_from_disk("/root/Qwen-Finance-LLM/dataset/Vamshirvk/Finance-Instruct-500k-reasoning")
    print("first example \n", dataset[0])


if __name__ == "__main__":
    main()
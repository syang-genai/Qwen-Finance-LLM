from utils import reformat

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


def dataset_reformat(example):
    example["prompt"]=[{"role": "user", "content": example["problem"]}]
    example["completion"]=[{"role": "assistant", "content": example["solution"]}]
    return example


def OpenR1(count):
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")

    dataset = load_dataset("open-r1/OpenR1-Math-220k", split="train")
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(count))

    # select prompt
    dataset = dataset.map(dataset_reformat, remove_columns=['problem', 'solution', 'answer', 'problem_type', 'question_type', 'source', 
    'uuid', 'is_reasoning_complete', 'generations', 'correctness_math_verify', 'correctness_llama', 
    'finish_reasons', 'correctness_count', 'messages'])
    dataset = dataset.map(reformat, fn_kwargs=dict(tokenizer=tokenizer, enable_think=False), remove_columns=["prompt","completion"])
    
    # save dataset
    dataset.save_to_disk("../dataset/Open-R1/OpenMathReasoning")
    return dataset 


if __name__ == "__main__":
    OpenR1(2500)
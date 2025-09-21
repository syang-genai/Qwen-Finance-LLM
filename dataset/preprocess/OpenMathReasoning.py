from utils import reformat

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


def train_dataset_reformat(example):
    example["prompt"]=[{"role": "user", "content": example["problem"]}]
    example["completion"]=[{"role": "assistant", "content": example["solution"]}]
    return example


def eval_dataset_reformat(example):
    example["message"]=[{"role": "user", "content": example["problem"]}]
    example["response"]=example["solution"]
    return example


def grpo_dataset_reformat(example):
    example["prompt"]=[{"role": "user", "content": example["problem"]}]
    example["completion"]=[{"role": "assistant", "content": example["solution"]}]
    example["reference_answer"]=[{"role": "assistant", "content": example["solution"]}]
    return example


def OpenR1(train_count, eval_count, grpo_count, model_name, train_save_path, eval_save_path, grpo_save_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")

    dataset = load_dataset("open-r1/OpenR1-Math-220k", split="train")
    dataset = dataset.shuffle(seed=42)
    

    train_dataset = dataset.select(range(train_count))
    train_dataset = train_dataset.map(train_dataset_reformat, remove_columns=['problem', 'solution', 'answer', 'problem_type', 'question_type', 'source', 
    'uuid', 'is_reasoning_complete', 'generations', 'correctness_math_verify', 'correctness_llama', 
    'finish_reasons', 'correctness_count', 'messages'])
    print(train_dataset[0])
    train_dataset = train_dataset.map(reformat, fn_kwargs=dict(tokenizer=tokenizer, enable_think=False), remove_columns=["prompt","completion"])
    train_dataset.save_to_disk(train_save_path)


    eval_dataset=dataset.select(range(train_count,train_count+eval_count))
    eval_dataset = eval_dataset.map(eval_dataset_reformat, remove_columns=['problem', 'solution', 'answer', 'problem_type', 'question_type', 'source', 
    'uuid', 'is_reasoning_complete', 'generations', 'correctness_math_verify', 'correctness_llama', 
    'finish_reasons', 'correctness_count', 'messages'])
    eval_dataset.to_json(eval_save_path)


    grpo_dataset = dataset.select(range(train_count+eval_count,train_count+eval_count+grpo_count))
    grpo_dataset = grpo_dataset.map(grpo_dataset_reformat, remove_columns=['problem', 'solution', 'answer', 'problem_type', 'question_type', 'source', 
    'uuid', 'is_reasoning_complete', 'generations', 'correctness_math_verify', 'correctness_llama', 
    'finish_reasons', 'correctness_count', 'messages'])
    print("grpo dataset \n", grpo_dataset[0])
    grpo_dataset.save_to_disk(grpo_save_path)

    return train_dataset, eval_dataset, grpo_dataset 


if __name__ == "__main__":
    OpenR1(train_count=5000, 
        eval_count=1000, 
        grpo_count=1000, 
        model_name="Qwen/Qwen3-0.6B", 
        train_save_path="../train_dataset/Open-R1/OpenMathReasoning", 
        eval_save_path="../eval_dataset/Open-R1/OpenMathReasoning/OMR.jsonl",
        grpo_save_path="../grpo_dataset/Open-R1/OpenMathReasoning")
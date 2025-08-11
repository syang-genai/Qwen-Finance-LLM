from utils import reformat
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


def dataset_format(example):
    prompt=f"""
            Evaluating whether to invest in the following startup or not, and provide a final single decision with explaination. 
            The startup idea: {example["idea"]}
            Your task including response in the following format:
            1. DECISION: [Invest] or [Do not invest]
            2. EXPLANATION: A very short 1–2 sentence explanation why you decided to invest or not.
        """
    
    example["prompt"]=[{"role":"user","content": prompt}]
    
    assistant= f"""            
            Decision: {example["manager_partner_decision"]}
            Explanation: {example["manager_partner_explanation"]}
        """ 
    
    example["completion"]=[{"role":"assistant","content": assistant}]
    return example


def FinancialDecisionsReasoning(count):
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")

    dataset = load_dataset("ZennyKenny/synthetic_vc_financial_decisions_reasoning_dataset",split="test")
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(count))
    
    dataset = dataset.map(dataset_format, remove_columns=["index","idea","junior_partner_pitch","hawk_reasoning","fin_reasoning","fit_reasoning","manager_partner_think","manager_partner_decision","manager_partner_explanation"])    
    dataset = dataset.map(reformat, fn_kwargs=dict(tokenizer=tokenizer, enable_think=False), remove_columns=["prompt","completion"])
    
    dataset.save_to_disk("../dataset/ZennyKenny/SyntheticFinancialDecisionsReasoningDataset")
    return dataset 


if __name__ == "__main__":
    FinancialDecisionsReasoning(count=180)
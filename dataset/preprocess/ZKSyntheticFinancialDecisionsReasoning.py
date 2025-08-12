from utils import reformat
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


def train_dataset_format(example):
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
    example["completion"]= [{"role":"assistant","content": assistant}] 
    return example


def eval_dataset_format(example):
    prompt=f"""
            Evaluating whether to invest in the following startup or not, and provide a final single decision with explaination. 
            The startup idea: {example["idea"]}
            Your task including response in the following format:
            1. DECISION: [Invest] or [Do not invest]
            2. EXPLANATION: A very short 1–2 sentence explanation why you decided to invest or not.
        """
    example["message"]=[{"role":"user","content": prompt}]

    assistant= f"""            
            Decision: {example["manager_partner_decision"]}
            Explanation: {example["manager_partner_explanation"]}
        """ 
    example["response"]=assistant
    return example


def FinancialDecisionsReasoning(train_count, eval_count, sublist, model_name, train_save_path, eval_save_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")

    dataset = load_dataset("ZennyKenny/synthetic_vc_financial_decisions_reasoning_dataset",split=sublist)
    dataset = dataset.shuffle(seed=42)
    
    train_dataset = dataset.select(range(train_count))
    train_dataset = train_dataset.map(train_dataset_format, remove_columns=["index","idea","junior_partner_pitch","hawk_reasoning","fin_reasoning","fit_reasoning","manager_partner_think","manager_partner_decision","manager_partner_explanation"])    
    print(train_dataset[0])
    train_dataset = train_dataset.map(reformat, fn_kwargs=dict(tokenizer=tokenizer, enable_think=False), remove_columns=["prompt","completion"])
    train_dataset.save_to_disk(train_save_path)

    eval_dataset = dataset.select(range(train_count,train_count+eval_count))
    eval_dataset = eval_dataset.map(eval_dataset_format, remove_columns=["index","idea","junior_partner_pitch","hawk_reasoning","fin_reasoning","fit_reasoning","manager_partner_think","manager_partner_decision","manager_partner_explanation"])    
    eval_dataset.to_json(eval_save_path)
    return train_dataset 


if __name__ == "__main__":
    FinancialDecisionsReasoning(
            train_count=180, 
            eval_count=20, 
            sublist="test", 
            model_name="Qwen/Qwen3-0.6B", 
            train_save_path="../train_dataset/ZennyKenny/SyntheticFinancialDecisionsReasoningDataset", 
            eval_save_path="../eval_dataset/ZennyKenny/SyntheticFinancialDecisionsReasoningDataset/SFDR.jsonl"
        )
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
            <think>
            Manager Partner Thoughts:
            {example["manager_partner_think"]}
            </think>
            
            Decision: {example["manager_partner_decision"]}
            Explanation: {example["manager_partner_explanation"]}
        """ 
    
    example["completion"]=[{"role":"assistant","content": assistant}]
    return example


def main():
    dataset = load_dataset("ZennyKenny/synthetic_vc_financial_decisions_reasoning_dataset",split="test")
    dataset = dataset.map(dataset_format, remove_columns=["index","idea","junior_partner_pitch","hawk_reasoning","fin_reasoning","fit_reasoning","manager_partner_think","manager_partner_decision","manager_partner_explanation"])
    
    # save dataset
    dataset.save_to_disk("/root/Qwen-Finance-LLM/dataset/ZennyKenny/SyntheticFinancialDecisionsReasoningDataset")
    dataset=load_from_disk("/root/Qwen-Finance-LLM/dataset/ZennyKenny/SyntheticFinancialDecisionsReasoningDataset")
    print("first example \n", dataset[0])

if __name__ == "__main__":
    main()
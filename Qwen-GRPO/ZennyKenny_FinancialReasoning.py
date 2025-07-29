from datasets import load_dataset

def preprocess_format(example):
    """
        example=["system":..., "user":..., "assistant":...]
    """
    
    assistant= f"""
        <think>
        Manager Partner Thoughts:
        {example["manager_partner_think"]}
        </think>

        Decision: {example["manager_partner_decision"]}
        Explanation: {example["manager_partner_explanation"]}
    """
    example["prompt"]=[{"role":"user","content": example["idea"]}]
    example["completion"]=[{"role":"assistant","content": assistant}]
    return example

def main():
    # Login using e.g. `huggingface-cli login` to access this dataset
    dataset=load_dataset("ZennyKenny/synthetic_vc_financial_decisions_reasoning_dataset",split="test")
    print(dataset)    
    
    dataset = dataset.map(preprocess_format, remove_columns=["index","idea","junior_partner_pitch","hawk_reasoning","fin_reasoning","fit_reasoning","manager_partner_think","manager_partner_decision","manager_partner_explanation"])
    
    # save dataset
    dataset.save_to_disk("/root/Qwen-Finance-LLM/Qwen-GRPO/preprocess/Financial_Decisions_Reasoning_Dataset")
    print("first example \n", dataset[0])
    
if __name__ == "__main__":
    main()
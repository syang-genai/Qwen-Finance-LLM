from datasets import load_dataset

def preprocess_format(example):
    """
        example=["system":..., "user":..., "assistant":...]
    """
    system_prompt=f"""
            You are evaluating whether to invest in the following startup. The user will give the startup idea. 
            Your task includes: 
            1. FIRST, REASON internally inside a <think>...</think> block. DO NOT include any decision or explanation here.
            2. AFTER the </think> block, WRITE:
            DECISION: [Invest] or [Do not invest]
            EXPLANATION: A very short 1–2 sentence explanation why you decided to invest or not.
            IMPORTANT: 
            - Keep DECISION and EXPLANATION outside the <think> block.
            - Follow the exact format shown.
            """
    
    example["prompt"]=[{"role":"system","content":system_prompt}, {"role":"user","content": example["idea"]}]
    
    assistant= f"""
        <think>
        Manager Partner Thoughts:
        {example["manager_partner_think"]}
        </think>

        Decision: {example["manager_partner_decision"]}
        Explanation: {example["manager_partner_explanation"]}
    """ 
    example["completion"]=[{"role":"assistant","content": assistant}]
    example["decision"]=example["manager_partner_decision"]
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
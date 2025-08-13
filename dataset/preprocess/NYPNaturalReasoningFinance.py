from datasets import load_dataset

def dataset_reformat(example):
    example["prompt"]=[{"role": "user", "content": example["question"]}]
    example["completion"]=[{"role": "assistant", "content": example["responses"][0]["response"]}]
    example["reference_answer"]=[{"role": "assistant", "content": example["reference_answer"]}]
    return example
    

def NYRF(save_path="../train_dataset/NeoYiPeng/NaturalReasoningFinance"):
    dataset = load_dataset("neoyipeng/natural_reasoning_finance",split="train")
    dataset = dataset.map(dataset_reformat, remove_columns=["question","responses"])
    dataset.save_to_disk(save_path)
    print(dataset[:1])
    return dataset 


if __name__ == "__main__":
    NYRF(save_path="../train_dataset/NeoYiPeng/NaturalReasoningFinance")

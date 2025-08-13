from datasets import load_dataset

def dataset_reformat(example):
    example["prompt"]=[{"role": "user", "content": example["instruction"]}]
    example["completion"]=[{"role": "assistant", "content": example["output"]}]
    example["reference_answer"]=[{"role": "assistant", "content": example["output"]}]
    return example
    

def FQ4D(split, save_path="../train_dataset/4D/FinanceQuestion"):
    dataset = load_dataset("4DR1455/finance_questions",split=split)
    dataset = dataset.map(dataset_reformat, remove_columns=["instruction","output","input"])
    dataset.save_to_disk(save_path)
    print(dataset[:1])
    return dataset 


if __name__ == "__main__":
    FQ4D(split="train", save_path="../train_dataset/4D/FinanceQuestion")

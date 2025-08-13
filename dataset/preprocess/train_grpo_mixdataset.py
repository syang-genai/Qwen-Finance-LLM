from FinanceQuestion4D import FQ4D
from NYPNaturalReasoningFinance import NYRF
from datasets import concatenate_datasets


def main():
    fq4d=FQ4D(split="train", save_path="../train_dataset/4D/FinanceQuestion")
    nyrf=NYRF(save_path="../train_dataset/NeoYiPeng/NaturalReasoningFinance")
    
    dataset=concatenate_datasets([fq4d, nyrf])
    dataset = dataset.shuffle(seed=42)
    dataset.save_to_disk("../train_dataset/train_grpo_mixdataset")


if __name__ == "__main__":
    main()

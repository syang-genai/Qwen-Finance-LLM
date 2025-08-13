from BigScienceP3 import BigScience
from DSFinanceReasoning import FinanceReasoning
from JFFinanceInstruct import FinanceInstruct
from OpenMathReasoning import OpenR1
from SimpleAIH3 import HC3Instruct
from VFinanceInstructReasoning import SynFinanceInstructReason
from ZKSyntheticFinancialDecisionsReasoning import FinancialDecisionsReasoning

from datasets import concatenate_datasets


def main():
    bs=BigScience(train_count=2500, \
        eval_count=500, \
        sublist="adversarial_qa_dbert_based_on", \
        model_name="Qwen/Qwen3-0.6B", \
        train_save_path="../train_dataset/BigScienceP3/QA", \
        eval_save_path="../eval_dataset/BSP3_QA.jsonl")


    op=OpenR1(train_count=2500, \
        eval_count=500, \
        sublist="train",\
        model_name="Qwen/Qwen3-0.6B", \
        train_save_path="../train_dataset/Open-R1/OpenMathReasoning", \
        eval_save_path="../eval_dataset/Open-R1/OpenMathReasoning/OMR.jsonl")

    
    hc=HC3Instruct(
            train_count=2500, \
            eval_count=500, \
            sublist="train", \
            model_name="Qwen/Qwen3-0.6B", \
            train_save_path="../train_dataset/H3Instruct", \
            eval_save_path="../eval_dataset/H3Instruct/H3I.jsonl"
        )

    
    
    fr=FinanceReasoning(
        train_count=400, \
        eval_count=100, \
        sublist="train", \
        model_name="Qwen/Qwen3-0.6B", \
        train_save_path="../train_dataset/Diweanshu/Finance-Reasoning", \
        eval_save_path="../eval_dataset/Diweanshu/Finance-Reasoning/FR.jsonl")

    
    sfr=SynFinanceInstructReason(
            train_count=40, \
            eval_count=0, \
            sublist="train", \
            model_name="Qwen/Qwen3-0.6B", \
            train_save_path="../train_dataset/Vamshirvk/Finance-Instruct-500k-reasoning", \
            eval_save_path="")


    
    fi= FinanceInstruct(
            train_count=5000, \
            eval_count=1000, \
            sublist="train", \
            model_name="Qwen/Qwen3-0.6B", \
            train_save_path="../train_dataset/Josephgflowers/Finance-Instruct-500k-Formated", \
            eval_save_path="../eval_dataset/Josephgflowers/Finance-Instruct-500k-Formated/FIF.jsonl", 
        )

    
    fdr=FinancialDecisionsReasoning(
            train_count=180, 
            eval_count=20, 
            sublist="test", 
            model_name="Qwen/Qwen3-0.6B", 
            train_save_path="../train_dataset/ZennyKenny/SyntheticFinancialDecisionsReasoningDataset", 
            eval_save_path="../eval_dataset/ZennyKenny/SyntheticFinancialDecisionsReasoningDataset/SFDR.jsonl"
        )
    
    dataset=concatenate_datasets([bs,op,hc,fr,sfr,fi,fdr])
    dataset = dataset.shuffle(seed=42)
    dataset.save_to_disk("../train_dataset/train_sft_mixdataset")


if __name__ == "__main__":
    main()

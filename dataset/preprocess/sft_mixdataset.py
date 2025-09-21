from BigScienceP3 import BigScience
from OpenMathReasoning import OpenR1
from FinancialInstructionAQ22 import Financial_Instruction_AQ22
from JFFinanceInstruct import FinanceInstruct
from MdFinancialAlpaca import FinanceAlpacaLlama
from AdFinancialQuestions import FinancialQuestions
from NYPNaturalReasoningFinance import NYRF
from BloombergFinancialNews import BloombergFinancialNews
from JFinancialQA import FinancialQA
from SKFinancialQuestions import SKFinancialQuestions

from datasets import concatenate_datasets

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset, load_from_disk
# Set the plot style
sns.set_style("whitegrid")


def main():
    bs,_,_=BigScience(
        5000, 
        1000, 
        1000, 
        "adversarial_qa_dbert_based_on", 
        "Qwen/Qwen3-0.6B", 
        train_save_path="../train_dataset/BigScienceP3/QA", 
        eval_save_path="../eval_dataset/BigScienceP3/QA/BSP3_QA.jsonl",
        grpo_save_path="../grpo_dataset/BigScienceP3/QA")
    print("bs")


    op,_,_=OpenR1(train_count=5000, 
        eval_count=1000, 
        grpo_count=1000, 
        model_name="Qwen/Qwen3-0.6B", 
        train_save_path="../train_dataset/Open-R1/OpenMathReasoning", 
        eval_save_path="../eval_dataset/Open-R1/OpenMathReasoning/OMR.jsonl",
        grpo_save_path="../grpo_dataset/Open-R1/OpenMathReasoning")
    print("op")


    fiaq, _, _ =Financial_Instruction_AQ22(
        5000, 
        1000, 
        5000, 
        "Qwen/Qwen3-0.6B", 
        train_save_path="../train_dataset/DeividasM/FinancialInstruction", 
        eval_save_path="../eval_dataset/DeividasM/FinancialInstruction/FIAQ22.jsonl", 
        grpo_save_path="../grpo_dataset/DeividasM/FinancialInstruction/FIAQ22")
    print("fiaq")
    

    fi,_,_=FinanceInstruct(
        train_count=2500, 
        eval_count=1000, 
        grpo_count=1000,  
        model_name="Qwen/Qwen3-0.6B", 
        train_save_path="../train_dataset/Josephgflowers/FinanceInstruct500kFormated", 
        eval_save_path="../eval_dataset/Josephgflowers/FinanceInstruct500kFormated/FIF.jsonl", 
        grpo_save_path="../grpo_dataset/Josephgflowers/FinanceInstruct500kFormated"
    )
    print("fi")


    fal,_,_=FinanceAlpacaLlama(
        train_count=5000, 
        eval_count=1000, 
        grpo_count=1000,  
        model_name="Qwen/Qwen3-0.6B", 
        train_save_path="../train_dataset/Madanarnav/FinancialAlpacaLlama", 
        eval_save_path="../eval_dataset/Madanarnav/FinancialAlpacaLlama/FAL.jsonl", 
        grpo_save_path="../grpo_dataset/Madanarnav/FinancialAlpacaLlama"
    )
    print("fal")


    fq,_,_=FinancialQuestions(
        train_count=2500, 
        eval_count=1000, 
        grpo_count=1000,  
        model_name="Qwen/Qwen3-0.6B", 
        train_save_path="../train_dataset/Adityaaaa/FinancialQuestions", 
        eval_save_path="../eval_dataset/Adityaaaa/FinancialQuestions/FQ.jsonl", 
        grpo_save_path="../grpo_dataset/Adityaaaa/FinancialQuestions"
    )
    print("fa")


    nyrf,_,_=NYRF(
        train_count=5000, 
        eval_count=1000, 
        grpo_count=1000,  
        model_name="Qwen/Qwen3-0.6B", 
        train_save_path="../train_dataset/NeoYiPeng/NaturalReasoningFinance", 
        eval_save_path="../eval_dataset/NeoYiPeng/NaturalReasoningFinance/NRF.jsonl", 
        grpo_save_path="../grpo_dataset/NeoYiPeng/NaturalReasoningFinance"
    )
    print("nyrf")

    bfn,_,_=BloombergFinancialNews(
        train_count=5000, 
        eval_count=1000, 
        grpo_count=1000,  
        model_name="Qwen/Qwen3-0.6B", 
        train_save_path="../train_dataset/Genloop/BloombergFinancialNew", 
        eval_save_path="../eval_dataset/Genloop/BloombergFinancialNew/BFN.jsonl", 
        grpo_save_path="../grpo_dataset/Genloop/BloombergFinancialNew"
    )
    print("bfn")


    fqa,_,_=FinancialQA(
        train_count=5000, 
        eval_count=1000, 
        grpo_count=1000,  
        model_name="Qwen/Qwen3-0.6B", 
        train_save_path="../train_dataset/JollyPrasad/FinancialQA", 
        eval_save_path="../eval_dataset/JollyPrasad/FinancialQA/JPFQA.jsonl", 
        grpo_save_path="../grpo_dataset/JollyPrasad/FinancialQA"
    )
    print("fqa")


    skfq,_,_=SKFinancialQuestions(
        train_count=3000, 
        eval_count=1000, 
        grpo_count=1000,  
        model_name="Qwen/Qwen3-0.6B", 
        train_save_path="../train_dataset/Sud/SKFinancialQuestions", 
        eval_save_path="../eval_dataset/Sud/SKFinancialQuestions/SKF.jsonl", 
        grpo_save_path="../grpo_dataset/Sud/SKFinancialQuestions"
    )
    print("skfq")
    
    dataset=concatenate_datasets([bs, op, fiaq, fi, fal, fq, nyrf, bfn, fqa, skfq])
    dataset = dataset.shuffle(seed=42)
    dataset.save_to_disk("../train_dataset/sft_mix_dataset")

    dataset=load_from_disk("../train_dataset/sft_mix_dataset") 
    df = dataset.to_pandas()
    print("df length",len(df))

    # 1. Calculate word counts for each sentence
    print(type(df["input_ids"]))
    df["word_count"]=df["input_ids"].str.len()

    # 2. Get summary statistics of the word counts
    print("\nSummary Statistics for Word Count:")
    print(df['word_count'].describe())

    # 3. Visualize the distribution with a histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(df['word_count'], bins=30, kde=True)
    plt.title('Distribution of Sentence Length (by Word Count)')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.savefig("distribution.png")

    # 4. Filter Dataset to 2048 and save dataset
    dataset=dataset.filter(lambda example: len(example["input_ids"])<=2048)
    df = dataset.to_pandas()
    dataset.save_to_disk("../train_dataset/filter_sft_mix_dataset")

    # 1. Calculate word counts for each sentence
    print(type(df["input_ids"]))
    df["word_count"]=df["input_ids"].str.len()

    # 2. Get summary statistics of the word counts
    print("\nSummary Statistics for Word Count:")
    print(df['word_count'].describe())

    # 3. Visualize the distribution with a histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(df['word_count'], bins=30, kde=True)
    plt.title('Distribution of Sentence Length (by Word Count)')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.savefig("filtered_distribution.png")

if __name__ == "__main__":
    main()

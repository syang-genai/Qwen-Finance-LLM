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

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset, load_from_disk, concatenate_datasets

from transformers import AutoTokenizer

# set the plot style
sns.set_style("whitegrid")


def main():
    # _,_,bs=BigScience(
    #     5000, 
    #     1000, 
    #     1000, 
    #     "adversarial_qa_dbert_based_on", 
    #     "Qwen/Qwen3-0.6B", 
    #     train_save_path="../train_dataset/BigScienceP3/QA", 
    #     eval_save_path="../eval_dataset/BigScienceP3/QA/BSP3_QA.jsonl",
    #     grpo_save_path="../grpo_dataset/BigScienceP3/QA")
    # print("bs",len(bs))


    # _,_,op=OpenR1(train_count=5000, 
    #     eval_count=1000, 
    #     grpo_count=1000, 
    #     model_name="Qwen/Qwen3-0.6B", 
    #     train_save_path="../train_dataset/Open-R1/OpenMathReasoning", 
    #     eval_save_path="../eval_dataset/Open-R1/OpenMathReasoning/OMR.jsonl",
    #     grpo_save_path="../grpo_dataset/Open-R1/OpenMathReasoning")
    # print("op",len(op))


    # _,_,fiaq=Financial_Instruction_AQ22(
    #     5000, 
    #     1000, 
    #     1000, 
    #     "Qwen/Qwen3-0.6B", 
    #     train_save_path="../train_dataset/DeividasM/FinancialInstruction", 
    #     eval_save_path="../eval_dataset/DeividasM/FinancialInstruction/FIAQ22.jsonl", 
    #     grpo_save_path="../grpo_dataset/DeividasM/FinancialInstruction/FIAQ22")
    # print("fiaq",len(fiaq))
    

    # _,_,fi =FinanceInstruct(
    #     train_count=2500, 
    #     eval_count=1000, 
    #     grpo_count=1000,  
    #     model_name="Qwen/Qwen3-0.6B", 
    #     train_save_path="../train_dataset/Josephgflowers/FinanceInstruct500kFormated", 
    #     eval_save_path="../eval_dataset/Josephgflowers/FinanceInstruct500kFormated/FIF.jsonl", 
    #     grpo_save_path="../grpo_dataset/Josephgflowers/FinanceInstruct500kFormated"
    # )
    # print("fi",len(fi))


    # _,_,fal=FinanceAlpacaLlama(
    #     train_count=5000, 
    #     eval_count=1000, 
    #     grpo_count=1000,  
    #     model_name="Qwen/Qwen3-0.6B", 
    #     train_save_path="../train_dataset/Madanarnav/FinancialAlpacaLlama", 
    #     eval_save_path="../eval_dataset/Madanarnav/FinancialAlpacaLlama/FAL.jsonl", 
    #     grpo_save_path="../grpo_dataset/Madanarnav/FinancialAlpacaLlama"
    # )
    # print("fal",len(fal))


    # _,_,fq=FinancialQuestions(
    #     train_count=2500, 
    #     eval_count=1000, 
    #     grpo_count=1000,  
    #     model_name="Qwen/Qwen3-0.6B", 
    #     train_save_path="../train_dataset/Adityaaaa/FinancialQuestions", 
    #     eval_save_path="../eval_dataset/Adityaaaa/FinancialQuestions/FQ.jsonl", 
    #     grpo_save_path="../grpo_dataset/Adityaaaa/FinancialQuestions"
    # )
    # print("fq",len(fq))


    # _,_,nyrf=NYRF(
    #     train_count=5000, 
    #     eval_count=1000, 
    #     grpo_count=1000,  
    #     model_name="Qwen/Qwen3-0.6B", 
    #     train_save_path="../train_dataset/NeoYiPeng/NaturalReasoningFinance", 
    #     eval_save_path="../eval_dataset/NeoYiPeng/NaturalReasoningFinance/NRF.jsonl", 
    #     grpo_save_path="../grpo_dataset/NeoYiPeng/NaturalReasoningFinance"
    # )
    # print("nyrf",len(nyrf))

    # _,_,bfn=BloombergFinancialNews(
    #     train_count=5000, 
    #     eval_count=1000, 
    #     grpo_count=1000,  
    #     model_name="Qwen/Qwen3-0.6B", 
    #     train_save_path="../train_dataset/Genloop/BloombergFinancialNew", 
    #     eval_save_path="../eval_dataset/Genloop/BloombergFinancialNew/BFN.jsonl", 
    #     grpo_save_path="../grpo_dataset/Genloop/BloombergFinancialNew"
    # )
    # print("bfn",len(bfn))


    # _,_,fqa=FinancialQA(
    #     train_count=5000, 
    #     eval_count=1000, 
    #     grpo_count=1000,  
    #     model_name="Qwen/Qwen3-0.6B", 
    #     train_save_path="../train_dataset/JollyPrasad/FinancialQA", 
    #     eval_save_path="../eval_dataset/JollyPrasad/FinancialQA/JPFQA.jsonl", 
    #     grpo_save_path="../grpo_dataset/JollyPrasad/FinancialQA"
    # )
    # print("fqa",len(fqa))

    # _,_,skfq=SKFinancialQuestions(
    #     train_count=3000, 
    #     eval_count=1000, 
    #     grpo_count=1000,  
    #     model_name="Qwen/Qwen3-0.6B", 
    #     train_save_path="../train_dataset/Sud/SKFinancialQuestions", 
    #     eval_save_path="../eval_dataset/Sud/SKFinancialQuestions/SKF.jsonl", 
    #     grpo_save_path="../grpo_dataset/Sud/SKFinancialQuestions"
    # )
    # print("skfq",len(skfq))
    
    # dataset=concatenate_datasets([bs, op, fiaq, fi, fal, fq, nyrf, bfn, fqa, skfq])
    # dataset = dataset.shuffle(seed=42)
    # dataset.save_to_disk("../grpo_dataset/grpo_mix_dataset")

    dataset=load_from_disk("../grpo_dataset/grpo_mix_dataset") 
    df = dataset.to_pandas()
    print("df length",len(df))
    

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    def fn_count(example):
        prompt=tokenizer(example['prompt'][0]['content'])
        completion=tokenizer(example["completion"][0]['content'])
        total_count=len(prompt["input_ids"])+len(completion["input_ids"])
        return  total_count

    df["word_count"]=df.apply(fn_count, axis=1)

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


    def filter_fn_count(example):
        prompt=tokenizer(example['prompt'][0]['content'])
        completion=tokenizer(example["completion"][0]['content'])

        instruction=tokenizer.apply_chat_template(
                example["prompt"],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False)
    
        response=tokenizer.apply_chat_template(
            example["completion"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False)
        

        instruction_ids=tokenizer(instruction,  add_special_tokens=False)
        response_ids=tokenizer(response, add_special_tokens=False)

        total_count=len(instruction_ids["input_ids"])+len(response_ids["input_ids"])

        if example["reference_answer"]!=None and len(example["reference_answer"][0]["content"])!=0 and total_count<1024 and len(response_ids["input_ids"])>0:
            return True
        else:
            print(example["reference_answer"])
            return False

    # 4. Filter Dataset to 2048 and save dataset
    dataset=dataset.filter(filter_fn_count)
    
    # 5. Generation without think
    def nothink(example):
        example['prompt'][0]['content']=example['prompt'][0]['content']+" "+"/no_think"
        return example

    dataset=dataset.map(nothink)

    df = dataset.to_pandas()
    dataset.save_to_disk("../grpo_dataset/filter_grpo_mix_dataset")
    
    # 1. Calculate word counts for each sentence
    df["word_count"]=df.apply(fn_count, axis=1)
    
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
    return 



if __name__ == "__main__":
    main()

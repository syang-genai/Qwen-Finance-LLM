import torch
from torch.utils.data import DataLoader

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments


def preprocess_format(example, tokenizer):
    """
        example={"system":, "user":, "assistant":}
    """
    system_prompt= "You are a financial assistant. Answer the user's question accurately but keep it brief." if example["system"]=='\n' else example["system"] 
    
    instruction=tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": example["user"]}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )
    
    response=tokenizer.apply_chat_template(
        [{"role": "assistant", "content": example["assistant"]}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )

    instruction_ids=tokenizer(instruction,  add_special_tokens=False)
    response_ids=tokenizer(response,  add_special_tokens=False)
    
    # create input_ids, attention and labels
    input_ids=instruction_ids["input_ids"]+response_ids["input_ids"]
    attention_mask=instruction_ids['attention_mask']+response_ids['attention_mask']
    labels=[-100]*len(instruction_ids["input_ids"])+response_ids["input_ids"]
    
    example["input_ids"]=input_ids
    example["attention_mask"]=attention_mask
    example["labels"]=labels

    return example
    


def main():
    dataset=load_from_disk("/root/Qwen-Finance-LLM/dataset/Josephgflowers/Finance-Instruct-500k-Formated")
    dataset=dataset.train_test_split(0.2)
    train_dataset=dataset["train"]
    eval_dataset=dataset["test"]
    
    # load model and tokenizer
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    
    
    # datacollector
    collate_fn=DataCollatorForSeq2Seq(tokenizer, padding=True, return_tensors="pt")
    # trainloader=DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    # testloader=DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    
    # train config and train
    args = TrainingArguments(
        output_dir="/root/Qwen-Finance-LLM/Qwen-OutputDir",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        max_steps=9,
        eval_strategy="steps",
        eval_steps=10,
        logging_steps=10,
        save_steps=3,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
    )
    

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    
    trainer.train(resume_from_checkpoint = False)
     

if __name__ == "__main__":
    main()

     
    

if __name__ == "__main__":
    main()

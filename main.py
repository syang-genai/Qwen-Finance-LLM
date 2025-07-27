import torch
from torch.utils.data import DataLoader

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments
import os

os.environ['TORCH_CUDA_ARCH_LIST']="8.9"

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
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2", device_map="cuda")
    
    # datacollector
    collate_fn=DataCollatorForSeq2Seq(tokenizer, padding=True, return_tensors="pt")
    # trainloader=DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    # testloader=DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    
    
    # train config and train
    train_args = TrainingArguments(
        output_dir="/root/Qwen-Finance-LLM/Qwen-OutputDir",
        overwrite_output_dir=True,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        torch_empty_cache_steps=4,
        eval_strategy="no",
        learning_rate=1e-8, #check qwen learning rate
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=7e-8,
        max_grad_norm=1,
        max_steps=4,
        lr_scheduler_type="linear", # cosine learning rate
        lr_scheduler_kwargs=dict(),
        warmup_ratio=0.25,
        log_level="debug",
        log_on_each_node=False,
        logging_strategy="steps",
        logging_steps=2,
        save_strategy="steps",
        save_steps=2,
        save_total_limit=2,
        save_only_model="False",
        restore_callback_states_from_checkpoint="True",
        data_seed=42,
        fp16=True, 
        dataloader_num_workers=0,
        deepspeed="/root/Qwen-Finance-LLM/deepspeed_config.json", 
        group_by_length=True,
        report_to="wandb",
        gradient_checkpointing=True,  
        torch_compile=False, 
        include_tokens_per_second=True,
        include_num_input_tokens_seen=True
    )
    
    
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    
    trainer.train(resume_from_checkpoint = False)


if __name__ == "__main__":
    main()

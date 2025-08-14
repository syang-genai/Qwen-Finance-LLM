import os
import argparse
import json

from utils import print_trainable_parameters

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
os.environ['TORCH_CUDA_ARCH_LIST']="8.9"


def main(config):
    # model name 
    model_name = config["model"]["name"]
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    
    # Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.uint8
    )
    
    # model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="cuda")
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
    
    # LoRA 
    loraconfig = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, loraconfig)
    print_trainable_parameters(model)

    # load dataset 
    dataset=load_from_disk(config["dataset"]["path"]) 
    dataset=dataset.train_test_split(0.1)
    dataset_train=dataset["train"]
    dataset_valid=dataset["test"]
    
    # datacollector
    collate_fn=DataCollatorForSeq2Seq(tokenizer, padding=True, return_tensors="pt")

    # train config and train
    train_args = TrainingArguments(
        data_seed=config["train_arg"]["dataset"]["data_seed"],
        dataloader_num_workers=config["train_arg"]["dataset"]["dataloader_num_workers"],
        group_by_length=True,
        bf16=config["train_arg"]["train"]["bf16"],
        per_device_train_batch_size=config["train_arg"]["train"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["train_arg"]["train"]["gradient_accumulation_steps"],
        gradient_checkpointing=False,  
        learning_rate=config["train_arg"]["train"]["learning_rate"], 
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        max_steps=config["train_arg"]["train"]["max_steps"],
        lr_scheduler_type="cosine",
        warmup_ratio=config["train_arg"]["train"]["warmup_ratio"],
        deepspeed=config["train_arg"]["train"]["deepspeed"], 
        torch_compile=False, 
        # eval
        per_device_eval_batch_size=config["train_arg"]["eval"]["per_device_eval_batch_size"],
        eval_strategy="steps",
        eval_steps=config["train_arg"]["eval"]["eval_steps"],
        # save
        output_dir=config["train_arg"]["save"]["output_dir"],
        overwrite_output_dir=True,
        save_strategy="steps",
        save_steps=config["train_arg"]["save"]["save_steps"],
        save_total_limit=2,
        save_only_model="False",
        # log
        log_level="debug",
        log_on_each_node=False,
        logging_strategy="steps",
        logging_steps=config["train_arg"]["log"]["logging_steps"],
        report_to="wandb",
        include_tokens_per_second=True,
        include_num_input_tokens_seen=True
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_valid,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    
    trainer.train(resume_from_checkpoint=config["train_arg"]["resume_from_checkpoint"])


if __name__ == "__main__":
    # Define the argument parser as shown above
    parser = argparse.ArgumentParser(description="Load configuration from a JSON file.")
    parser.add_argument("--config_file", help="Path to the JSON configuration file.")
    args = parser.parse_args()

    # Load the JSON file
    try:
        with open(args.config_file, 'r') as file:
            config = json.load(file)
            print("Configuration loaded successfully!")
    except FileNotFoundError:
        print(f"Error: The file '{args.config_file}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{args.config_file}' is not a valid JSON file.")
    
    main(config)

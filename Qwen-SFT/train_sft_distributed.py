import os
import torch
from torch.utils.data import DataLoader

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments

from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model

from utils import print_trainable_parameters

os.environ['TORCH_CUDA_ARCH_LIST']="8.9"

def main():
    # model name 
    model_name = "Qwen/Qwen3-8B"
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
    
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="cuda")
    model.config.use_cache = False
    # model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2", device_map="cuda")
    model = prepare_model_for_kbit_training(model)
    
    # LoRA 
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    # load dataset 
    dataset=load_from_disk("./mixed_dataset")
    dataset=dataset.train_test_split(0.1)
    dataset_train=dataset["train"]
    dataset_valid=dataset["test"]
    
    # datacollector
    collate_fn=DataCollatorForSeq2Seq(tokenizer, padding=True, return_tensors="pt")

    # train config and train
    train_args = TrainingArguments(
        # dataset
        data_seed=42,
        dataloader_num_workers=4,
        group_by_length=True,
        # train
        fp16=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        torch_empty_cache_steps=4,
        gradient_checkpointing=True,  
        learning_rate=1e-8, 
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=7e-8,
        max_grad_norm=1,
        max_steps=8,
        lr_scheduler_type="cosine",
        warmup_ratio=0.25,
        deepspeed="./deepspeed_config.json", 
        torch_compile=False, 
        # eval
        per_device_eval_batch_size=4,
        eval_strategy="steps",
        eval_steps=8,
        # save
        output_dir="../Qwen-OutputDir/SFT",
        overwrite_output_dir=True,
        save_strategy="steps",
        save_steps=4,
        save_total_limit=2,
        save_only_model="False",
        # log
        log_level="debug",
        log_on_each_node=False,
        logging_strategy="steps",
        logging_steps=2,
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
    
    trainer.train(resume_from_checkpoint=True)


if __name__ == "__main__":
    main()

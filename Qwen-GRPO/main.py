from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset,load_from_disk
from trl import GRPOConfig, GRPOTrainer
# import wandb

def main():
    # dataset=load_from_disk("/root/Qwen-Finance-LLM/Qwen-GRPO/preprocess/Financial_Decisions_Reasoning_Dataset")
    # print("dataset", dataset)
    dataset = load_dataset("trl-lib/tldr", split="train")
    
    def reward_len(completions, **kwargs):
        # print(completions)
        # print([-abs(20 - len(completion)) for completion in completions])
        return [-abs(20 - len(completion)) for completion in completions]

    
    model_name="Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    
    # train config and train
    training_args = GRPOConfig(
        disable_dropout=True,
        remove_unused_columns=False,
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.0,
        beta=0.1,
        num_iterations=2,
        epsilon=0.2,
        importance_sampling_level="sequence",
        reward_weights=[1],
        loss_type="dr_grpo",
        mask_truncated_completions=True, 
        sync_ref_model=True,
        ref_model_mixup_alpha=0.6,
        output_dir="/root/Qwen-Finance-LLM/Qwen-GRPO/Qwen-OutputDir",
        overwrite_output_dir=True,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        eval_strategy="no",
        learning_rate=1e-8, #check qwen learning rate
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=7e-8,
        max_grad_norm=1,
        max_steps=4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.25,
        log_level="debug",
        log_on_each_node=False,
        logging_strategy="steps",
        logging_steps=2,
        log_completions=True, 
        num_completions_to_print=1,
        report_to="wandb",
        wandb_log_unique_prompts=True, 
        save_strategy="steps",
        save_steps=2,
        save_total_limit=2,
        save_only_model="False",
        restore_callback_states_from_checkpoint="True",
        data_seed=42,
        fp16=True, 
        dataloader_num_workers=0,
        group_by_length=True,
        gradient_checkpointing=True,  
        torch_compile=False
    )
    

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_len,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()


if __name__ == "__main__":
    # wandb.init(project="Qwen-GRPO")
    main()
    # wandb.finish()
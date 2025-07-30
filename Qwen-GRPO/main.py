from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset,load_from_disk
from trl import GRPOConfig, GRPOTrainer
import wandb


def main():
    dataset=load_from_disk("/root/Qwen-Finance-LLM/Qwen-GRPO/preprocess/Financial_Decisions_Reasoning_Dataset")
    

    def reward_len(completions, **kwargs):
        return [-abs(20 - len(completion[0]["content"])) for completion in completions]
    
    
    model_name="Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    
    # train config and train
    training_args = GRPOConfig(
        # data preprocessing
        remove_unused_columns=False,
        # train
        restore_callback_states_from_checkpoint=True,
        dataloader_num_workers=0,
        group_by_length=True,
        fp16=True, 
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        # gradient_accumulation_steps=2,
        # gradient_checkpointing=True,
        learning_rate=1e-7, 
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        max_grad_norm=1,
        max_steps=4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.25,
        beta=1, # kl divergence
        num_iterations=4,
        epsilon=0.2,
        importance_sampling_level="sequence",
        reward_weights=[1],
        loss_type="dr_grpo",
        mask_truncated_completions=True, 
        # reference model  
        sync_ref_model=True,
        ref_model_mixup_alpha=0.6,
        disable_dropout=True,
        torch_compile=True,
        # evaluate
        eval_strategy="no",
        # generation keywords
        max_completion_length=1024,
        generation_batch_size=8,
        # save 
        output_dir="/root/Qwen-Finance-LLM/Qwen-GRPO/Qwen-OutputDir",
        overwrite_output_dir=True,
        save_strategy="steps",
        save_steps=2,
        save_total_limit=2,
        save_only_model=False,
        # log 
        report_to="wandb",
        logging_strategy="steps",
        logging_steps=2,
        log_level="debug",
        log_completions=True, 
        wandb_log_unique_prompts=True, 
        num_completions_to_print=1,
        log_on_each_node=False
    )

    # training_args = GRPOConfig(max_completion_length=512, output_dir="Qwen2-0.5B-GRPO")
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_len,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()


if __name__ == "__main__":
    wandb.init(project="Qwen-GRPO")
    main()
    wandb.finish()
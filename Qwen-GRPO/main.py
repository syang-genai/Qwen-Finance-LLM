import re
from functools import partial

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset,load_from_disk
from trl import GRPOConfig, GRPOTrainer
import wandb

from format_check import validate_startup_investment_response


def main():
    dataset=load_from_disk("./preprocess/Financial_Decisions_Reasoning_Dataset")
    
    reward_model_name = "Qwen/Qwen3-0.6B"
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
    reward_model = AutoModelForCausalLM.from_pretrained(reward_model_name, device_map="auto").to("cuda")
    reward_model.eval()
    
    
    def get_llm_judgment_reward(prompts, completions, expect_completion, decision, llm_model=reward_model, llm_tokenizer=reward_tokenizer, **kwargs):
        """
            Uses a general-purpose LLM to 'judge' a response and extracts a score.
            This is more complex as it involves prompt engineering for the LLM and parsing its output.
        """

        messages=list()
        for first, second in zip(expect_completion,completions):
            evaluation_prompt = f"""
                Please rate the similarity between the following two sentense.
                First sentense: {first[0]["content"]}
                Second sentense: {second[0]["content"]}
                Answer the question with Rating (1-5).
                Rating:
            """
            messages.append([{"role": "user", "content": evaluation_prompt}])

        encoded_input = llm_tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True, enable_thinking=False, padding=True).to("cuda")
        with torch.no_grad():
            outputs = llm_model.generate(
                encoded_input,
                max_new_tokens=100,
                num_return_sequences=1,
                do_sample=False)
        
        
        generated_text = llm_tokenizer.batch_decode(outputs[:,encoded_input.shape[1]:], skip_special_tokens=True)
        
        
        reward=list()
        pattern = r"[-+]?\d*\.\d+|\d+"
        for gt in generated_text:
            match = re.findall(pattern, gt)
            if len(match)>0:
                try:
                    rating = float(match[0])
                    if rating<0:
                        reward.append(0)
                    elif rating>5:
                        reward.append(5)
                    else:
                        reward.append(rating)
                except ValueError:
                    reward.append(0)
            else:
                reward.append(0)

        return reward 

    
    def decision_format_reward(prompts, completions, decision, **kwargs):
        freward=list()
        dreward=list()
        for idx, completion in enumerate(completions):
            response=validate_startup_investment_response(completion[0]["content"])
            
            if response['is_valid']==True:
                freward.append(1)
                if response["parsed_data"]["decision"]==decision[idx]:
                    dreward.append(1)
                else:
                    dreward.append(0)
            else:
                freward.append(0)
                dreward.append(0)
        
        reward=[i+j for i,j in zip(freward,dreward)]
        return reward
    
    
    model_name="Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    
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
        max_steps=1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.25,
        beta=1, # kl divergence
        num_iterations=4, 
        epsilon=0.2,
        importance_sampling_level="sequence",
        reward_weights=[1,3],
        loss_type="dr_grpo",
        mask_truncated_completions=True, 
        # deepspeed="./deepspeed_config.json",
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
        output_dir="./Qwen-OutputDir",
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
    
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[decision_format_reward, get_llm_judgment_reward],
        args=training_args,
        train_dataset=dataset,
    )


    trainer.train()



if __name__ == "__main__":
    wandb.init(project="Qwen-GRPO")
    main()
    wandb.finish()
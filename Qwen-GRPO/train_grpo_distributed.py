import os
import re
import random
import wandb
import json
import argparse

from Utils.utils import print_trainable_parameters

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM,  BitsAndBytesConfig

from datasets import load_dataset,load_from_disk
from trl import GRPOConfig, GRPOTrainer
from peft import prepare_model_for_kbit_training, PeftModel
from peft import LoraConfig, get_peft_model
from openai import OpenAI


# --- Reward Calculation Function (RPC service) ---
def _calculate_llm_rewards_on_rm_host(prompts, completions, reference_answer, **kwargs) -> list[float]:
    score_pattern = r"[-+]?\d*\.\d+|\d+"  
    think_pattern = r"<think>(.*?)</think>"
    
    rewards=list()
    messages=list()
    generated_text=list()
    for ref_ans, ans in zip(reference_answer, completions):
        ans[0]["content"]=re.sub(think_pattern, "", ans[0]["content"],flags=re.DOTALL).strip() # filter refernce answer 
        
        evaluation_prompt = f"""
            Please rate the similarity between the following two sentense.
            Reference answer sentense: {ref_ans[0]["content"]}
            Generated answer sentense: {ans[0]["content"]}
            Answer the question with Rating (1-5).
            Rating:
        """

        messages.append({"role": "user", "content": evaluation_prompt})
        
        client = OpenAI(
            api_key="empty",
            base_url="http://127.0.0.2:8000/v1"
        )

        response = client.chat.completions.create(
            model="Qwen/Qwen3-14B",
            messages=messages,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            temperature=0.8,
            max_tokens=1024
        )

        # Print the generated text
        response_text=response.choices[0].message.content
        generated_text.append(response_text)
    
    
    for gt in generated_text:
        match = re.findall(score_pattern, gt)
        if len(match)>0:
            try:
                rating = float(match[0])
                if rating<0:
                    rewards.append(0)
                elif rating>5:
                    rewards.append(5)
                else:
                    rewards.append(rating)
            except ValueError:
                rewards.append(0)
        else:
            rewards.append(0)
    return rewards


def grpo_reward_function(prompts, completions, reference_answer, **kwargs) -> torch.Tensor:
    rewards_list = _calculate_llm_rewards_on_rm_host(prompts, completions, reference_answer, **kwargs)
    return torch.tensor(rewards_list, device="cuda")


def grpo_pattern_reward_function(prompts, completions, reference_answer, **kwargs) -> torch.Tensor:
    think_pattern = r"<think>\s+</think>"    
    rewards=list()

    for ans in completions:
        match = re.search(think_pattern, ans[0]["content"], flags=re.DOTALL)        
        if match:
            # print("match")
            rewards.append(1)
        else:
            # print("no match")
            rewards.append(0)
    return torch.tensor(rewards, device="cuda")


def main(config): 
    model_name=config["policy_model"]["name"]    
    dataset=load_from_disk(config["dataset"]["path"])
    # dataset=dataset.select(range(0,320))
    dataset=dataset.train_test_split(0.2)
    dataset_train=dataset["train"]
    dataset_valid=dataset["test"]

    policy_tokenizer = AutoTokenizer.from_pretrained(model_name, enable_thinking=False)
    policy_tokenizer.padding_side = "left"
    

    bnb_policy_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.uint8
    ) 
    

    policy_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_policy_config, device_map="auto")
    # policy_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_policy_config, device_map="auto")
    # policy_model = PeftModel.from_pretrained(policy_model, config["adapter_path"])
    
    # policy_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_policy_config, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map="auto")
    # policy_model = prepare_model_for_kbit_training(policy_model, use_gradient_checkpointing=False)
    
    # LoRA 
    loraconfig = LoraConfig(
        r=config["lora_config"]["r"],
        lora_alpha=config["lora_config"]["lora_alpha"],
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    

    policy_model = get_peft_model(policy_model, loraconfig)
    print_trainable_parameters(policy_model)

    
    training_args = GRPOConfig(
        # data preprocessing
        remove_unused_columns=config["train_arg"]["dataset"]["remove_unused_columns"],
        dataloader_num_workers=config["train_arg"]["dataset"]["dataloader_num_workers"],
        group_by_length=config["train_arg"]["dataset"]["group_by_length"],
        # train
        bf16=True, # works: False and True
        per_device_train_batch_size=config["train_arg"]["train"]["per_device_train_batch_size"], # deepspeed
        gradient_accumulation_steps=config["train_arg"]["train"]["gradient_accumulation_steps"], 
        gradient_checkpointing=False, 
        learning_rate=config["train_arg"]["train"]["learning_rate"], # deepseed
        weight_decay=0.1, # deepseed
        adam_beta1=0.9, # deepseed
        adam_beta2=0.95, # deepseed
        adam_epsilon=1e-8, # deepseed
        max_grad_norm=1, # deepseed
        max_steps=config["train_arg"]["train"]["max_steps"],
        lr_scheduler_type="cosine",
        warmup_ratio=config["train_arg"]["train"]["warmup_ratio"],
        beta=1, # kl divergence
        num_iterations=4, 
        epsilon=0.2,
        importance_sampling_level="sequence",
        reward_weights=[0.5, 0.5],
        loss_type="dr_grpo",
        deepspeed=config["train_arg"]["train"]["deepspeed"], 
        mask_truncated_completions=True, 
        cache_implementation='dynamic',
        torch_compile=False,  # torch_compile=False
        # reference model  
        sync_ref_model=True,
        ref_model_mixup_alpha=config["train_arg"]["train"]["ref_model"]["ref_model_mixup_alpha"],
        disable_dropout=True,
        use_vllm=False,
        vllm_mode="colocate",
        # generation keywords
        max_completion_length=config["train_arg"]["train"]["max_completion_length"],
        generation_batch_size=config["train_arg"]["train"]["generation_batch_size"], # batch_size*step_accumulation
        # evaluate
        per_device_eval_batch_size=config["train_arg"]["eval"]["per_device_eval_batch_size"], # deepspeed
        eval_strategy=config["train_arg"]["eval"]["eval_strategy"],
        eval_steps=config["train_arg"]["eval"]["eval_steps"],
        # save 
        output_dir=config["train_arg"]["save"]["output_dir"],
        overwrite_output_dir=True,
        save_strategy="steps",
        save_steps=config["train_arg"]["save"]["save_steps"],
        save_total_limit=2,
        save_only_model=False,
        # log 
        report_to="wandb",
        logging_strategy="steps",
        logging_steps=config["train_arg"]["log"]["logging_steps"],
        log_level="debug",
        log_completions=True, 
        wandb_log_unique_prompts=True, 
        num_completions_to_print=1,
        log_on_each_node=False
    )
    
    trainer = GRPOTrainer(
        model=policy_model,
        processing_class=policy_tokenizer,
        reward_funcs=[grpo_pattern_reward_function, grpo_reward_function],
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_valid
    )
    
    print(f"Starting GRPO training...")
    trainer.train(resume_from_checkpoint=config["train_arg"]["resume_from_checkpoint"])
    print(f"GRPO training finished.")
    return 


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

import os
import re
from functools import partial
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
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model

REWARD_MODEL_NAME = "Qwen/Qwen3-0.6B"
POLICY_MODEL_NAME = "Qwen/Qwen3-0.6B"
REWARD_MODEL_GLOBAL_GPU_ID = 1


def setup_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")
    
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size,
        init_method=f"env://"
    )
    
    torch.cuda.set_device(local_rank)
    print(f"[Rank {rank}/{world_size}] Initialized DDP on cuda:{local_rank}")

    dist.rpc.init_rpc(f"worker_{rank}", rank=rank, world_size=world_size, backend=dist.rpc.BackendType.TENSORPIPE)
    print(f"[Rank {rank}/{world_size}] Initialized RPC as worker_{rank}")
    return
    
def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()
    if dist.is_initialized():
        dist.rpc.shutdown()


global_reward_model_tokenizer = None
global_reward_model = None
global_reward_model_device = None
def _load_reward_model_on_device(model_name: str, device_id: int):
    global global_reward_model_tokenizer, global_reward_model, global_reward_model_device
    print(f"[Rank {dist.get_rank()}] Loading Reward Model on cuda:{device_id}...")
    
    bnb_reward_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.uint8
    )
    
    global_reward_model_tokenizer = AutoTokenizer.from_pretrained(model_name, quantization_config=bnb_reward_config, device_map="auto")
    global_reward_model_tokenizer.padding_side = "left" 
    global_reward_model = AutoModelForCausalLM.from_pretrained(model_name)
    global_reward_model.to(f"cuda:{device_id}")
    global_reward_model.eval()
    global_reward_model_device = f"cuda:{device_id}"
    print(f"[Rank {dist.get_rank()}] Reward Model loaded on cuda:{device_id}.")



# --- Reward Calculation Function (RPC service) ---
def _calculate_llm_rewards_on_rm_host(prompts, completions, reference_answer, **kwargs) -> list[float]:
    if global_reward_model is None or global_reward_model_tokenizer is None:
        raise RuntimeError(f"Reward model not initialized on worker_{dist.get_rank()}.")
    
    rewards=list()
    messages=list()
    for ref_ans, ans in zip(reference_answer, completions):
        evaluation_prompt = f"""
            Please rate the similarity between the following two sentense.
            Reference answer sentense: {ref_ans[0]["content"]}
            Generated answer sentense: {ans[0]["content"]}
            Answer the question with Rating (1-5).
            Rating:
        """
        messages.append([{"role": "user", "content": evaluation_prompt}])

    encoded_inputs = global_reward_model_tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True, enable_thinking=False, padding=True).to(f"cuda:{REWARD_MODEL_GLOBAL_GPU_ID}")
    
    with torch.no_grad():
        outputs = global_reward_model.generate(
            encoded_inputs,
            max_new_tokens=100,
            num_return_sequences=1,
            do_sample=False)

    generated_text = global_reward_model_tokenizer.batch_decode(outputs[:,encoded_inputs.shape[1]:], skip_special_tokens=True)

    pattern = r"[-+]?\d*\.\d+|\d+"
    for gt in generated_text:
        match = re.findall(pattern, gt)
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
    current_rank = dist.get_rank()
    if current_rank == REWARD_MODEL_GLOBAL_GPU_ID:
        rewards_list = _calculate_llm_rewards_on_rm_host(prompts, completions, reference_answer, **kwargs)
    else:
        rm_worker_name = f"worker_{REWARD_MODEL_GLOBAL_GPU_ID}"
        rewards_list = dist.rpc.rpc_sync(
            to=rm_worker_name,
            func= _calculate_llm_rewards_on_rm_host, 
            args=(prompts, completions, reference_answer),
            kwargs=kwargs
        )
        print(f"[Rank {current_rank}] Received {len(rewards_list)} rewards from {rm_worker_name}")


    return torch.tensor(rewards_list, device=f"cuda:{dist.get_rank()}")


def main(config):
    setup_distributed()
    current_rank = dist.get_rank()
    if current_rank == REWARD_MODEL_GLOBAL_GPU_ID:
        _load_reward_model_on_device(REWARD_MODEL_NAME, REWARD_MODEL_GLOBAL_GPU_ID)
    else:
        print(f"[Rank {current_rank}] This process is not hosting the reward model.")
    
    
    
    dataset=load_from_disk("../dataset/train_dataset/grpo_mix_dataset")
    policy_tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_NAME)
    policy_tokenizer.padding_side = "left"  # batch generation 

    bnb_policy_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.uint8
    ) 

    policy_model = AutoModelForCausalLM.from_pretrained(POLICY_MODEL_NAME, quantization_config=bnb_policy_config)
    policy_model = prepare_model_for_kbit_training(policy_model, use_gradient_checkpointing=False)
    
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
        bf16=config["train_arg"]["train"]["bf16"], # deepspeed 
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
        reward_weights=[1],
        loss_type="dr_grpo",
        mask_truncated_completions=True, 
        cache_implementation='dynamic',
        deepspeed=config["train_arg"]["train"]["deepspeed"],
        use_vllm=False, 
        # reference model  
        sync_ref_model=True,
        ref_model_mixup_alpha=config["train_arg"]["train"]["ref_model"]["ref_model_mixup_alpha"],
        disable_dropout=True,
        torch_compile=False, # True without deepseek 
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
        reward_funcs=[grpo_reward_function],
        args=training_args,
        train_dataset=dataset,
    )
    
    print(f"[Rank {current_rank}] Starting GRPO training...")
    trainer.train(resume_from_checkpoint=config["train_arg"]["resume_from_checkpoint"])
    print(f"[Rank {current_rank}] GRPO training finished.")
    cleanup_distributed()


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

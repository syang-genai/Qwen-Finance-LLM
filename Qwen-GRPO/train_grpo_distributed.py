import os
import re
from functools import partial
import random
import wandb
import deepspeed

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM

from datasets import load_dataset,load_from_disk
from trl import GRPOConfig, GRPOTrainer
from format_check import validate_startup_investment_response


REWARD_MODEL_NAME = "Qwen/Qwen3-0.6B"
POLICY_MODEL_NAME = "Qwen/Qwen3-0.6B"
REWARD_MODEL_GLOBAL_GPU_ID = 0 


def setup_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")

    dist.init_process_group(
        backend="nccl",
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
    global_reward_model_tokenizer = AutoTokenizer.from_pretrained(model_name)
    global_reward_model = AutoModelForCausalLM.from_pretrained(model_name)
    global_reward_model.to(f"cuda:{device_id}")
    global_reward_model.eval()
    global_reward_model_device = f"cuda:{device_id}"
    print(f"[Rank {dist.get_rank()}] Reward Model loaded on cuda:{device_id}.")



# --- Reward Calculation Function (RPC service) ---
def _calculate_llm_rewards_on_rm_host(prompts, completions, expect_completion, decision, **kwargs) -> list[float]:
    if global_reward_model is None or global_reward_model_tokenizer is None:
        raise RuntimeError(f"Reward model not initialized on worker_{dist.get_rank()}.")

    rewards=list()
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


def grpo_reward_function(prompts, completions, expect_completion, decision, **kwargs) -> torch.Tensor:
    current_rank = dist.get_rank()
    if current_rank == REWARD_MODEL_GLOBAL_GPU_ID:
        rewards_list = _calculate_llm_rewards_on_rm_host(prompts, completions, expect_completion, decision, **kwargs)
    else:
        rm_worker_name = f"worker_{REWARD_MODEL_GLOBAL_GPU_ID}"
        rewards_list = dist.rpc.rpc_sync(
            to=rm_worker_name,
            func= _calculate_llm_rewards_on_rm_host, 
            args=(prompts, completions, expect_completion, decision),
            kwargs=kwargs
        )
        print(f"[Rank {current_rank}] Received {len(rewards_list)} rewards from {rm_worker_name}")


    return torch.tensor(rewards_list, device=f"cuda:{dist.get_rank()}")


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


def main():
    setup_distributed()
    current_rank = dist.get_rank()
    if current_rank == REWARD_MODEL_GLOBAL_GPU_ID:
        _load_reward_model_on_device(REWARD_MODEL_NAME, REWARD_MODEL_GLOBAL_GPU_ID)
    else:
        print(f"[Rank {current_rank}] This process is not hosting the reward model.")

    
    dataset=load_from_disk("./preprocess/Financial_Decisions_Reasoning_Dataset")
    
    policy_tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_NAME)
    policy_model = AutoModelForCausalLM.from_pretrained(POLICY_MODEL_NAME)        
    # policy_model, _, _, _ = deepspeed.initialize(
    #         config="./deepspeed_config.json",
    #         model=policy_model
    #     )

    training_args = GRPOConfig(
        # data preprocessing
        remove_unused_columns=False,
        # train
        restore_callback_states_from_checkpoint=True,
        dataloader_num_workers=0,
        group_by_length=True,
        fp16=True, 
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
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
        # reward_weights=[1],
        reward_weights=[1,3],
        loss_type="dr_grpo",
        mask_truncated_completions=True, 
        deepspeed="./deepspeed_config.json",
        # reference model  
        sync_ref_model=True,
        ref_model_mixup_alpha=0.6,
        disable_dropout=True,
        torch_compile=False , # True without deepseek 
        # evaluate
        eval_strategy="no",
        # generation keywords
        max_completion_length=1024,
        generation_batch_size=16,
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
        model=policy_model,
        processing_class=policy_tokenizer,
        reward_funcs=[decision_format_reward, grpo_reward_function],
        # reward_funcs=[decision_format_reward],
        args=training_args,
        train_dataset=dataset,
    )

    print(f"[Rank {current_rank}] Starting GRPO training...")
    trainer.train()
    print(f"[Rank {current_rank}] GRPO training finished.")
    cleanup_distributed()


if __name__ == "__main__":
    wandb.init(project="Qwen-GRPO")
    main()
    wandb.finish()
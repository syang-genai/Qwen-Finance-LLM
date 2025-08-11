from evalscope import TaskConfig, run_task
from transformers import AutoModelForCausalLM

qwen_model=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

print(download)
task_cfg = TaskConfig(
    model=qwen_model,
    generation_config={
        'max_tokens': 20000, 
        'temperature': 0.7, 
        'top_p': 0.8,
        'top_k': 20,
        'n': 1,
        'chat_template_kwargs': {'enable_thinking': False}
    },
    
    datasets=[
        'data_collection', 
        'finance_qa'
    ],
    
    dataset_args={
        'data_collection': {
            "local_path": "/root/Qwen-Finance-LLM/Qwen-Eval/eval_dataset",
        },

        'finance_qa': {
            "local_path": "/root/Qwen-Finance-LLM/dataset/eval_dataset/BigScienceP3/QA",
            "subset_list": [
                "BSP3_QA"       
            ]
        }
    },
    eval_batch_size=32,
    limit=32
)

run_task(task_cfg=task_cfg)
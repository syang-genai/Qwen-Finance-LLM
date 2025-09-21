import nltk
import torch
nltk.download('punkt_tab')

from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model="Qwen/Qwen3-8B", 
    api_url="http://127.0.0.2:8000/v1/chat/completions", 
    eval_type='service', 

    generation_config={
        'max_tokens': 1024, 
        'n': 1, 
        'chat_template_kwargs': {'enable_thinking': False}
    },
    
    datasets=[
        'data_collection', 
    ],
    
    dataset_args={
        'data_collection': {
            "dataset_id": "benchmark_dataset/qwen3_eval.jsonl",
        },
    },  
    
    judge_strategy="auto",
    judge_model_args={
        'model_id': 'Qwen/Qwen3-0.6B',
        'api_url': "http://127.0.0.1:8000/v1/chat/completions",
        "score_type": "pattern",
        "generation_config": {
                'max_tokens': 1024, 
                'n': 1, 
                'chat_template_kwargs': {'enable_thinking': False}
            },
    },

    eval_batch_size=8,
    limit=512
)

run_task(task_cfg=task_cfg)
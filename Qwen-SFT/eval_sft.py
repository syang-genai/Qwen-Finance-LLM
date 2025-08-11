from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='Qwen3-32B',
    api_url='http://127.0.0.1:8801/v1/chat/completions',
    eval_type='service',
    datasets=[
        'data_collection',
    ],
    dataset_args={
        'data_collection': {
            'dataset_id': 'modelscope/EvalScope-Qwen3-Test',
        }
    },
    eval_batch_size=128,
    generation_config={
        'max_tokens': 20000, 
        'temperature': 0.7, 
        'top_p': 0.8,
        'top_k': 20,
        'n': 1,
        'chat_template_kwargs': {'enable_thinking': False}
    },
    timeout=60000,
    stream=True,
    limit=1000,
)

run_task(task_cfg=task_cfg)

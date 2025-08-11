from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='Qwen3-32B',
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
            'dataset_id': 'modelscope/EvalScope-Qwen3-Test',
        },
        'finance_qa': {
            "local_path": "custom_eval/text/qa",
            "subset_list": [
                # 评测数据集名称，上述 *.jsonl 中的 *，可配置多个子数据集
                "example"       
            ]
        }
    },
    eval_batch_size=128,
    limit=32
)

run_task(task_cfg=task_cfg)

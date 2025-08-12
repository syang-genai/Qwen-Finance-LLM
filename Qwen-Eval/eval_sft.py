import nltk
nltk.download('punkt_tab')

from evalscope import TaskConfig, run_task
from transformers import AutoModelForCausalLM, AutoTokenizer

qwen_tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
qwen_model=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
qwen_tokenizer.save_pretrained("qwen_model")
qwen_model.model.save_pretrained("qwen_model")

task_cfg = TaskConfig(
    model="qwen_model",
    generation_config={
        'max_tokens': 20000, 
        'n': 1,
        'chat_template_kwargs': {'enable_thinking': False}
    },
    
    datasets=[
        'data_collection', 
        'general_qa'
    ],
    
    dataset_args={
        'data_collection': {
            "local_path": "benchmark_dataset/qwen3_eval.jsonl",
        },
        'general_qa': {
            "local_path": "../dataset/eval_dataset",
            "subset_list": [
                "BigScienceP3/BSP3_QA", 
                "Diweanshu/Finance-Reasoning/FR",
                "H3Instruct/H3I", 
                "Josephgflowers/Finance-Instruct-500k-Formated/FIF",
                "Open-R1/OpenMathReasoning/OMR",
                "ZennyKenny/SyntheticFinancialDecisionsReasoningDataset/SFDR"
            ]
        }
    },
    eval_batch_size=32,
    limit=128
)

run_task(task_cfg=task_cfg)
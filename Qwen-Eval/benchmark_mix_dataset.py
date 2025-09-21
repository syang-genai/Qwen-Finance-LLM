from evalscope.collections import CollectionSchema, DatasetInfo, WeightedSampler
from evalscope.utils.io_utils import dump_jsonl_data
from evalscope import TaskConfig, run_task

schema = CollectionSchema(name='Qwen3', datasets=[
    CollectionSchema(name='mmlu_pro', datasets=[
        DatasetInfo(name='mmlu_pro', weight=1, task_type='exam', tags=['en'], args={'few_shot_num': 0})
    ]),
        
    CollectionSchema(name='mmlu_redux', datasets=[
        DatasetInfo(name='mmlu_redux', weight=1, task_type='exam', tags=['en'], args={'few_shot_num': 0})
    ]),
    
    CollectionSchema(name='Math&Science', datasets=[
        DatasetInfo(name='gpqa', weight=1, task_type='knowledge', tags=['en'], args={'subset_list': ['gpqa_diamond'], 'few_shot_num': 0})
    ]),

])

# get the mixed data
mixed_data = WeightedSampler(schema).sample(20000)  # set a large number to ensure all datasets are sampled
# dump the mixed data to a jsonl file
dump_jsonl_data(mixed_data, 'benchmark_dataset/qwen3_eval.jsonl')
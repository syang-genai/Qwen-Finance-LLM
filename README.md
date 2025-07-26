2. model parallel training--data parallel(deepspeed), parallel checkpoint loading and saving. accelerate and deepspeed and model load/save. 
https://huggingface.co/docs/transformers/trainer
https://huggingface.co/docs/transformers/accelerate
https://huggingface.co/docs/accelerate/en/usage_guides/deepspeed#multi-node-deepspeed
Trainer
Accelerate


https://github.com/huggingface/accelerate/tree/main/examples
https://zhuanlan.zhihu.com/p/711876344
https://llamafactory.readthedocs.io/zh-cn/latest/advanced/distributed.html
https://github.com/huggingface/blog/blob/main/zh/pytorch-ddp-accelerate-transformers.md
Accelerate：在无需大幅修改代码的情况下完成并行化。同时还支持DeepSpeed的多种ZeRO策略，基本上无需改任何代码。并且验证了单机单卡 单机多卡 多机多卡并行均不用改实验代码. 
accelerate and accelerate configuration
accelerate_config.yaml (contains deepspeed)


https://www.bilibili.com/video/BV1uK421a7HG/?vd_source=20c0bb159a3a7a03b4d60bac7724fb15
https://github.com/lansinuote/Simple_Accelerate
Trainer 封装了 Accelerate 封装了 torch.distributed
#torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="192.168.56.104" --master_port=60006 8.多机调度.py --arg1 --arg2
#torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr="192.168.56.104" --master_port=60006 8.多机调度.py --arg1 --arg2

https://discuss.huggingface.co/t/trainer-api-for-data-parallel-on-multi-node/138715
Enable torchrun or deepspeed – You’ll need to launch your training script using torchrun (PyTorch) or DeepSpeed for multi-node training.
Set distributed_training parameters – In your training arguments, set ddp_find_unused_parameters=False and make sure torch.distributed.launch or torchrun is configured correctly.
Check environment variables – Each node should have correct settings for MASTER_ADDR, MASTER_PORT, WORLD_SIZE, and RANK.
Ensure all nodes communicate – Make sure SSH is set up, and all nodes can see each other. You might need to set up NCCL backend settings properly

TRL+vLLM(speedup training--calculate speed up):
https://huggingface.co/docs/trl/v0.19.1/en/sft_trainer#format-your-input-prompts
https://huggingface.co/docs/trl/en/deepspeed_integration

---
# Launch Single Node Multi-GPU Training
1. **deepspeed** 
  export TORCH_CUDA_ARCH_LIST="8.9" 
  deepspeed --num_gpus=2 examples/pytorch/translation/run_translation.py 
2. **torchrun**
  torchrun --standalone --nnodes=1 --nproc-per-node=2 main.py 
3. **accelerate config** 
  accelerate configuration--saved at /root/.cache/huggingface/accelerate/default_config.yaml    
  accelerate launch --num_processes=2 main.py  

---

5. use large model+LoRA and distributed training framework, and bigger dataset.
6. evaluation performance improvement.
7. MOE

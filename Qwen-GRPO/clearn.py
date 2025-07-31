import torch 
import torch.distributed as dist


# Your training code ...
torch.cuda.empty_cache()
if dist.is_initialized():
    dist.destroy_process_group()
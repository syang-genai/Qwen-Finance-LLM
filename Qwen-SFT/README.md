## create enviroment
uv pip install -r pyproject.toml
uv pip install flash-attn --no-build-isolation

## launch training
torchrun --standalone --nnodes=1 --nproc-per-node=1 train_sft_distributed.py --config_file sft_config.json  
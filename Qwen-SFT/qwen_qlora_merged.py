import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Define the base model and paths
model_name = "Qwen/Qwen3-0.6B"
adapter_path = "../Qwen-OutputDir/SFT/checkpoint-96"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Set up quantization config (if you used QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.uint8
)


# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, torch_dtype=torch.bfloat16, device_map="cuda")
# Load the LoRA adapter onto the base model
model_to_merge = PeftModel.from_pretrained(base_model, adapter_path)
# Merge the adapter with the base model and unload the adapter
merged_model = model_to_merge.merge_and_unload()


# Define the save path
merged_model_path = "../Qwen-OutputDir/SFT/merged_model"
# Save the merged model
merged_model.save_pretrained(merged_model_path)
# Save the tokenizer
tokenizer.save_pretrained(merged_model_path)

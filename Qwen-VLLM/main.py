from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_name = "Qwen/Qwen3-0.6B"
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Load the model
    # The model weights will be downloaded to your cache directory (~/.cache/huggingface/hub)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return 

if __name__ == "__main__":
    main()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_module(args):
    print("Loading base model for LoRA merge...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=True,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        model,
        args.eval_lora_module_path,
    )

    print("Merging LoRA adapter into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.eval_model_path}...")
    model.save_pretrained(
        args.eval_model_path,
        max_shard_size="6GB",
        safe_serialization=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(args.eval_model_path)

    print("Merge complete.")
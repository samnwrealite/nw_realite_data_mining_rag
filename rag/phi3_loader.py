# rag/phi3_loader.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Configurable model name — replace with the exact HF repo you prefer
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"  

def load_phi3_model(model_name: str = MODEL_NAME, device: str = "cpu"):
    """
    Loads the Phi-3 mini model (CPU). Keep device='cpu' for notebooks/laptops without GPU.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.to(device)
    model.eval()
    return tokenizer, model

def generate_answer(tokenizer, model, prompt: str, max_new_tokens: int = 200):
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

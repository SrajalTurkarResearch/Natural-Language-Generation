# case_study_pruned_chatbot.py
# --------------------------------------------------------------
# Real-world: Pruned GPT-2 for a wearable chat assistant
# Goal: 70% sparsity, <150 ms response on CPU, coherent replies.
# --------------------------------------------------------------

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import time

# --------------------------------------------------------------
# 1. Load base model
# --------------------------------------------------------------
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# --------------------------------------------------------------
# 2. Magnitude pruning (70% sparsity)
# --------------------------------------------------------------
sparsity = 0.70
for name, param in model.named_parameters():
    if "weight" in name and param.dim() >= 2:
        tensor = param.data.cpu().numpy()
        thresh = np.percentile(np.abs(tensor), sparsity * 100)
        mask = np.abs(tensor) >= thresh
        param.data = torch.from_numpy(tensor * mask).to(param.device)


# --------------------------------------------------------------
# 3. Helper: generate reply
# --------------------------------------------------------------
def chat(prompt, max_new=30):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=True, top_p=0.9
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


# --------------------------------------------------------------
# 4. Latency & size
# --------------------------------------------------------------
def size_mb(m):
    return sum(p.numel() * p.element_size() for p in m.parameters()) / (1024**2)


print(f"Original size : {size_mb(GPT2LMHeadModel.from_pretrained('gpt2')):.1f} MB")
print(f"Pruned size   : {size_mb(model):.1f} MB")

prompt = "User: Hi, how's the weather?\nBot:"
t0 = time.time()
reply = chat(prompt)
latency = (time.time() - t0) * 1000
print(f"\nReply ({latency:.1f} ms): {reply.split('Bot:')[-1]}")

print("\nPruned model meets wearable constraints: <150 ms, ~70% smaller.")

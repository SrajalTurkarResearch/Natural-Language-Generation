# LoRA Practical Code
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np

model = AutoModelForCausalLM.from_pretrained("gpt2")
config = LoraConfig(r=8)
peft_model = get_peft_model(model, config)

# Viz
W = np.random.rand(4, 4)
A = np.random.rand(2, 4)
B = np.random.rand(4, 2)
delta = B @ A
fig, axs = plt.subplots(1, 3)
axs[0].imshow(W)
axs[1].imshow(delta)
axs[2].imshow(W + delta)
plt.show()

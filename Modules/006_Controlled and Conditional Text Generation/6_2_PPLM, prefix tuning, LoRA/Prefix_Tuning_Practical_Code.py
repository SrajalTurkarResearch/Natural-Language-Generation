# Prefix Tuning Practical Code
from peft import get_peft_model, PrefixTuningConfig
from transformers import AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np

model = AutoModelForCausalLM.from_pretrained("gpt2")
config = PrefixTuningConfig(num_virtual_tokens=20)
peft_model = get_peft_model(model, config)
print(sum(p.numel() for p in peft_model.parameters() if p.requires_grad))

# Viz
steps = np.arange(100)
loss = np.exp(-steps / 20)
plt.plot(steps, loss)
plt.title("Prefix Loss")
plt.show()

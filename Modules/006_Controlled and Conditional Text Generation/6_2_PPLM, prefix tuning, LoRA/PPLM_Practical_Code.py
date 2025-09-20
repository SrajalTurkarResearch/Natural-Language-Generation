# PPLM Practical Code - Use forever in career
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def attribute_gradient(h, alpha=0.02):
    grad = torch.tensor([0.1, 0.4, -0.2])
    return h + alpha * grad


h_t = torch.tensor([0.5, -0.3, 0.2])
h_new = attribute_gradient(h_t)
print(f"Updated h: {h_new}")

# Viz
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.quiver(0, 0, 0, 0.5, -0.3, 0.2, color="b")
ax.quiver(0, 0, 0, 0.502, -0.292, 0.196, color="g")
plt.title("PPLM Nudge")
plt.show()

# attention_visualizer.py
"""
Visualizes attention weights from a transformer model as a heatmap.
Purpose: Help researchers understand how context influences NLG output.
Dependencies: transformers, torch, matplotlib, numpy
"""

import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import numpy as np


class AttentionVisualizer:
    def __init__(self, model_name="bert-base-uncased"):
        """Initialize BERT model and tokenizer."""
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name, output_attentions=True)

    def get_attention_weights(self, text):
        """Extract attention weights for the input text."""
        inputs = self.tokenizer(text, return_tensors="pt")
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        with torch.no_grad():
            outputs = self.model(**inputs)
        attentions = (
            outputs.attentions[-1][0].mean(dim=0).numpy()
        )  # Last layer, average heads
        return attentions, tokens

    def plot_attention_heatmap(self, text, save_path=None):
        """Plot a heatmap of attention weights."""
        attentions, tokens = self.get_attention_weights(text)
        plt.figure(figsize=(10, 8))
        plt.imshow(attentions, cmap="hot", interpolation="nearest")
        plt.xticks(np.arange(len(tokens)), tokens, rotation=45)
        plt.yticks(np.arange(len(tokens)), tokens)
        plt.title("Attention Weights Heatmap")
        plt.colorbar()
        if save_path:
            plt.savefig(save_path)
        plt.show()


# Example usage
if __name__ == "__main__":
    visualizer = AttentionVisualizer()
    text = "Memory and context are crucial in NLG."
    visualizer.plot_attention_heatmap(text, save_path="attention_heatmap.png")

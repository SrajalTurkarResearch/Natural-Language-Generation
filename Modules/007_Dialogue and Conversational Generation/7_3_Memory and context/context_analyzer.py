# context_analyzer.py
"""
Analyzes context in text using a transformer model (BERT) and extracts attention weights.
Purpose: Teach researchers how context is processed in NLG via attention mechanisms.
Dependencies: transformers, torch, numpy
"""

import torch
from transformers import BertTokenizer, BertModel
import numpy as np


class ContextAnalyzer:
    def __init__(self, model_name="bert-base-uncased"):
        """Initialize BERT model and tokenizer."""
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name, output_attentions=True)

    def analyze_context(self, text, target_word):
        """Extract attention weights for a given text and target word."""
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt")
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
        attentions = outputs.attentions  # List of attention matrices

        # Find target word index
        target_idx = tokens.index(target_word) if target_word in tokens else -1
        if target_idx == -1:
            return None, tokens, None

        # Average attention weights across heads for the last layer
        last_layer_attention = attentions[-1][0].mean(dim=0).numpy()
        attention_weights = last_layer_attention[target_idx]

        return attention_weights, tokens, inputs

    def print_attention(self, text, target_word):
        """Print attention weights for a target word in the text."""
        weights, tokens, _ = self.analyze_context(text, target_word)
        if weights is None:
            print(f"Word '{target_word}' not found in tokenized text.")
            return
        print(f"Attention weights for '{target_word}':")
        for token, weight in zip(tokens, weights):
            print(f"{token}: {weight:.4f}")


# Example usage
if __name__ == "__main__":
    analyzer = ContextAnalyzer()
    text = "Memory and context are crucial in NLG."
    analyzer.print_attention(text, "context")

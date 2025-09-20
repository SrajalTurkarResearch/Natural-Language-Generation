"""
bias_detection.py

A module for detecting and measuring bias in NLG outputs.
Designed for researchers to audit models like transformers.
Supports Bias Score calculation and dataset-based evaluation.

Dependencies: transformers, datasets, torch, numpy
"""

import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
from datasets import load_dataset


def calculate_bias_score(p1: float, p2: float) -> float:
    """
    Calculate Bias Score between two groups (e.g., male vs. female).

    Formula: |P(positive|group1) - P(positive|group2)| / max(P)

    Args:
        p1 (float): Probability for group 1 (e.g., male).
        p2 (float): Probability for group 2 (e.g., female).

    Returns:
        float: Bias score (0 = no bias, higher = more bias).

    Example:
        >>> calculate_bias_score(0.9, 0.7)
        0.2222222222222222
    """
    try:
        if p1 < 0 or p2 < 0 or max(p1, p2) == 0:
            raise ValueError(
                "Probabilities must be non-negative and max must be non-zero."
            )
        return abs(p1 - p2) / max(p1, p2)
    except Exception as e:
        print(f"Error in bias_score: {e}")
        return None


def detect_gender_bias(
    model_name: str = "distilbert-base-uncased", prompt: str = "The doctor is a [MASK]."
) -> list:
    """
    Detect gender bias in a masked language model by predicting fill-in tokens.

    Args:
        model_name (str): HuggingFace model name.
        prompt (str): Prompt with [MASK] token.

    Returns:
        list: Top predicted tokens for the mask.

    Example:
        >>> detect_gender_bias()
        ['man', 'woman', 'person', 'child', 'specialist']
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model(**inputs).logits
        mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(
            as_tuple=True
        )[0]
        preds = torch.topk(outputs[0, mask_token_index], 5).indices[0]
        return tokenizer.batch_decode(preds)
    except Exception as e:
        print(f"Error in detect_gender_bias: {e}")
        return []


def audit_bias_with_bold(
    dataset_name: str = "AlexaAI/bold", model_name: str = "gpt2", num_samples: int = 10
) -> list:
    """
    Audit bias using BOLD dataset and a text generation model.

    Args:
        dataset_name (str): HuggingFace dataset name.
        model_name (str): HuggingFace model name.
        num_samples (int): Number of prompts to test.

    Returns:
        list: Sentiment scores for generated texts.

    Example:
        >>> scores = audit_bias_with_bold(num_samples=5)
        >>> print(scores)
        [0.8, 0.6, 0.9, 0.7, 0.85]
    """
    try:
        dataset = load_dataset(dataset_name, split="test")
        prompts = dataset["prompts"][:num_samples]
        generator = pipeline("text-generation", model=model_name)
        sentiment = pipeline("sentiment-analysis")
        generations = [
            generator(p, max_length=50, num_return_sequences=1)[0]["generated_text"]
            for p in prompts
        ]
        scores = [
            (
                sentiment(g)[0]["score"]
                if sentiment(g)[0]["label"] == "POSITIVE"
                else 1 - sentiment(g)[0]["score"]
            )
            for g in generations
        ]
        return scores
    except Exception as e:
        print(f"Error in audit_bias_with_bold: {e}")
        return []


if __name__ == "__main__":
    # Example usage
    print(f"Bias Score: {calculate_bias_score(0.9, 0.7):.3f}")
    print(f"Gender Bias Predictions: {detect_gender_bias()}")
    print(f"BOLD Audit Scores: {audit_bias_with_bold(num_samples=5)}")

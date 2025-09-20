# Image-to-Text and Caption Storytelling: Basics and Theory
# For aspiring scientists inspired by Turing, Einstein, and Tesla
# This file covers the fundamentals, theory, and mathematical foundations of image-to-text and caption storytelling in NLG.

# Import libraries (install with: pip install matplotlib)
import matplotlib.pyplot as plt
import numpy as np

# --- Theory and Tutorials ---
# What is Image-to-Text?
# Image-to-text (image captioning) is when a computer looks at a picture and writes a short description, like "A cat sleeps on a couch."
# Think of it like Einstein explaining a complex idea with simple words: the computer sees the picture and tells you what's in it.
# Key parts:
# - Object Detection: Finding things like "cat" or "couch."
# - Scene Understanding: Knowing the setting, like "in a room."
# - Attribute Extraction: Adding details, like "fluffy cat."

# What is Caption Storytelling in NLG?
# NLG (Natural Language Generation) is making computers write human-like text. Caption storytelling goes beyond descriptions to create a story, like:
# "In a sunny room, a fluffy cat naps on a cozy couch, dreaming of treats."
# It's like Tesla describing an invention's impact, not just its parts.
# Key parts:
# - Narrative Structure: A story has a start, middle, and end.
# - Emotional Inference: Guessing feelings, like "happy" from a smile.
# - Creative Augmentation: Adding fun details, like "dreaming of treats."

# Mathematical Foundation (Made Simple)
# 1. Image to Numbers: A Convolutional Neural Network (CNN) turns a picture (I) into a feature vector (f):
#    f = CNN(I) = sum(W_k * I + b_k)
#    W_k are weights (learned numbers), * is multiply, b_k is a bias (adjustment).
# 2. Attention: Focuses on picture parts for each word:
#    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
#    Q is the word we're making, K/V are picture parts, softmax makes chances add to 1.
# 3. Sentence Probability: Chance of a sentence:
#    P(sentence | f) = P(word_1 | f) * P(word_2 | word_1, f) * ...
# 4. Loss: How wrong the computer is:
#    L = -sum(log P(correct word))
#    We adjust weights to make L smaller.


# Example Calculation
def calculate_caption_probability():
    f = [0.7, 0.3]  # Simplified feature vector
    probs = [0.75, 0.8, 0.9]  # P("A"|f), P("cat"|"A", f), P("sleeps"|"A cat", f)
    total_prob = np.prod(probs)  # 0.75 * 0.8 * 0.9 = 0.54
    loss = -sum(np.log(probs))  # - (log 0.75 + log 0.8 + log 0.9) ≈ 0.616
    print(f"Caption Probability: {total_prob:.3f}")
    print(f"Loss: {loss:.3f}")


# --- Visualization ---
# Simulated attention heatmap (shows where the model looks)
def plot_attention_heatmap():
    attention = np.random.rand(224, 224)  # Fake attention data
    plt.imshow(attention, cmap="hot")
    plt.title("Attention Heatmap (Simulated)")
    plt.colorbar()
    plt.show()


# --- Exercise ---
# 1. Run the calculation below. Change probs to [0.6, 0.7, 0.8] and see how probability and loss change.
# 2. Write what the heatmap represents in your own words.

# Run the code
if __name__ == "__main__":
    print("Running Basics and Theory")
    calculate_caption_probability()
    plot_attention_heatmap()

# --- Research Insight ---
# Recent papers (e.g., TPCap, arXiv 2025) show zero-shot captioning (describing new images without training) is key for scalable AI. As a scientist, explore how to reduce training data needs.

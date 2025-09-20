# setup_and_data.py
# Purpose: Initialize environment and define demo corpus for vector store experiments.
# Scientific Context: Like Turing's foundational setup for computation, this establishes the data bedrock for NLG retrieval.

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

# Set random seed for reproducibility (Einstein's insistence on predictable experiments)
np.random.seed(42)
torch.manual_seed(42)

# Demo corpus for all scripts
# Analogy: A small library of texts, like Tesla's lab notes, for testing retrieval
corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "A quick fox in the morning.",
    "Brown dogs are lazy in the sun.",
    "Jumps and runs in the field.",
]
query = "quick fox brown"

print("Setup complete. Corpus size:", len(corpus))
# Note: Install faiss-cpu, rank_bm25, sentence-transformers for full functionality.
# For real embeddings, use: from sentence_transformers import SentenceTransformer
# For large datasets, load: newsgroups = fetch_20newsgroups()

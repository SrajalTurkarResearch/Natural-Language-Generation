"""
detection_methods.py: Implementations of hallucination detection techniques.
Import into notebook for reusable detection pipelines.
"""

from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np
from typing import List, Optional
from utils import semantic_entropy, consistency_score  # Assuming utils.py is available

# Preload models (can be overridden)
generator = pipeline("text-generation", model="distilgpt2")
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def self_consistency_check(
    query: str, num_samples: int = 5, threshold: float = 0.8
) -> dict:
    """
    Perform self-consistency check for hallucination detection.

    Steps:
    1. Generate multiple responses.
    2. Embed them.
    3. Compute cosine similarity matrix.
    4. Calculate consistency score.

    Returns:
        Dict with 'score' (float), 'flag' (bool: True if hallucination), 'responses' (list).
    """
    responses = [
        generator(query, max_length=50)[0]["generated_text"] for _ in range(num_samples)
    ]
    embeddings = embedder.encode(responses)
    sim_matrix = cosine_similarity(embeddings)
    score = consistency_score(sim_matrix)
    flag = score < threshold
    return {"score": score, "flag": flag, "responses": responses}


def semantic_entropy_detection(
    query: str, num_samples: int = 10, num_clusters: int = 3, threshold: float = 1.0
) -> dict:
    """
    Detect hallucinations via semantic entropy.

    Logic: Cluster embeddings, compute entropy on cluster probabilities.

    Returns:
        Dict with 'entropy' (float), 'flag' (bool), 'responses' (list).
    """
    responses = [
        generator(query, max_length=50)[0]["generated_text"] for _ in range(num_samples)
    ]
    embeddings = embedder.encode(responses)
    kmeans = KMeans(n_clusters=num_clusters, n_init=10).fit(embeddings)
    cluster_counts = np.bincount(kmeans.labels_)
    probs = cluster_counts / num_samples
    entropy = semantic_entropy(probs)
    flag = entropy > threshold
    return {"entropy": entropy, "flag": flag, "responses": responses}


# Advanced: Placeholder for TSV (expand with your trained vectors)
def tsv_detection(activations_t: np.ndarray, activations_h: np.ndarray) -> float:
    """
    Basic TSV separation check. Extend with model injections for production.
    """
    mu_t, sigma_t = np.mean(activations_t), np.std(activations_t)
    mu_h, sigma_h = np.mean(activations_h), np.std(activations_h)
    return tsv_separation(np.array([mu_t]), np.array([mu_h]), sigma_t, sigma_h)

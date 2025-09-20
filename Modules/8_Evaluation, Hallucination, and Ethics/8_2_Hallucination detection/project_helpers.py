"""
project_helpers.py: Scaffolding for mini/major projects and exercises.
Use as starters for your research prototypes.
"""

from detection_methods import self_consistency_check, semantic_entropy_detection
from visualizations import plot_entropy_distribution
from data_handlers import load_hallucination_dataset
from typing import List


def mini_project_consistency(
    dataset_name: str = "truthful_qa", num_queries: int = 10
) -> List[dict]:
    """
    Mini Project: Apply self-consistency to a dataset subset.

    Returns:
        List of detection results.
    """
    df = load_hallucination_dataset(dataset_name)
    results = []
    for query in df["question"][:num_queries]:  # Assuming 'question' column
        result = self_consistency_check(query)
        results.append(result)
    return results


def major_project_tsv_simulation(num_samples: int = 100) -> None:
    """
    Major Project Starter: Simulate TSV with random latents.

    Visualize separation.
    """
    truthful = np.random.normal(0.5, 0.1, (num_samples, 2))
    halluc = np.random.normal(0.1, 0.2, (num_samples, 2))
    plot_latent_separation(truthful, halluc)


def exercise_entropy_calc(probs_list: List[List[float]]) -> List[float]:
    """
    Exercise: Compute entropies for multiple distributions.

    Solution-like: Returns list of H values.
    """
    return [semantic_entropy(probs) for probs in probs_list]


# Example usage in notebook: results = mini_project_consistency(); plot_entropy_distribution([r['score'] for r in results])

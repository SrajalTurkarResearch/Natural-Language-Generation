import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def visualize_embeddings():
    """
    Visualize word embeddings using t-SNE to show emotional relationships.
    """
    # Sample words and their embeddings
    words = ["happy", "joyful", "sad", "angry"]
    embeddings = np.array(
        [
            [0.5, 0.3, 0.2],  # happy
            [0.4, 0.35, 0.15],  # joyful
            [-0.6, -0.2, -0.1],  # sad
            [-0.5, 0.4, -0.3],  # angry
        ]
    )

    # Reduce dimensionality to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], color="blue")
    for i, word in enumerate(words):
        plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
    plt.title("Map of Emotional Words")
    plt.xlabel("X Direction")
    plt.ylabel("Y Direction")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    visualize_embeddings()

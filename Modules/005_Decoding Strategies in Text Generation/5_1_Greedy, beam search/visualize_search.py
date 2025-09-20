import matplotlib.pyplot as plt
from matplotlib import patches
from toy_lm import vocab


# Visualize search trees for Greedy and Beam Search
# Sketch these in your notes to understand decision paths
def visualize_tree(sequences, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(title)
    for i, seq in enumerate(sequences):
        for j, token in enumerate(seq[1:]):  # Skip <start>
            ax.add_patch(patches.FancyArrow(j, i, 1, 0, head_width=0.1, color="blue"))
            ax.text(j + 0.5, i, vocab[token], ha="center")
    ax.axis("off")
    plt.show()


# Example usage
if __name__ == "__main__":
    # Example sequences (replace with actual outputs)
    greedy_seq = [[0, 2, 3, 4, 5]]  # e.g., ['<start>', 'cat', 'sat', 'on', 'mat']
    beam_seqs = [[0, 2, 3, 4, 5], [0, 2, 8, 9], [0, 1, 6, 7]]  # Multiple beams
    visualize_tree(greedy_seq, "Greedy Search Tree")
    visualize_tree(beam_seqs, "Beam Search Tree")

# Evaluation and Projects: Measuring Quality and Building Applications
# This file covers how to evaluate captions and build projects.

# Install: pip install nltk datasets matplotlib
from nltk.translate.bleu_score import sentence_bleu
from datasets import load_dataset
import matplotlib.pyplot as plt

# --- Theory ---
# Evaluation metrics measure how good a caption is:
# - BLEU: Counts matching words (1 to 4-word groups).
# - METEOR: Allows similar words (e.g., "run" and "dash").
# - CIDEr: Weights important words using TF-IDF (term frequency-inverse document frequency).
# - SPICE: Checks meaning (objects, relations).


# --- Code Guide: Evaluation ---
def evaluate_caption():
    reference = [["A", "cat", "sleeps", "on", "a", "couch"]]
    candidate = ["A", "cat", "is", "resting"]
    score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))  # BLEU-1
    print("BLEU-1 Score:", score)


# --- Project: Mini Captioner ---
def mini_project_captioner():
    # Load dataset (small subset of COCO)
    try:
        dataset = load_dataset(
            "ydshieh/coco_dataset_script", "2017", data_dir="./coco_data"
        )
        print("Dataset Sample:", dataset["train"][0])
    except:
        print("Download COCO dataset or use a local copy.")
        return

    # Visualize dataset size
    sizes = [len(dataset[split]) for split in dataset]
    plt.bar(dataset.keys(), sizes)
    plt.title("Dataset Split Sizes")
    plt.show()


# --- Exercise ---
# 1. Run evaluate_caption with candidate ["A", "cat", "sleeps"]. How does BLEU change?
# 2. Load Flickr30k dataset instead of COCO. Print one sample.

# --- Research Insight ---
# Metrics like BLEU favor exact matches, missing semantic quality. As a scientist, develop a new metric combining BLEU and SPICE for better meaning capture.

if __name__ == "__main__":
    print("Running Evaluation and Projects")
    evaluate_caption()
    mini_project_captioner()

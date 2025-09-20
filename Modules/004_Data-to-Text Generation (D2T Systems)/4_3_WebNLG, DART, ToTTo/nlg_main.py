# nlg_main.py
# Main script to run the NLG tutorial components interactively
# Ties together theory, code, visualizations, and applications

from nlg_theory import print_nlg_theory
from nlg_code import run_inference, evaluate_bleu, fine_tune_model
from nlg_visualizations import plot_dataset_sizes, plot_model_performance
from nlg_applications_projects import (
    print_applications,
    print_projects,
    print_research_directions,
)


def print_future_directions():
    """
    Prints future directions, tips, and missing topics for scientists.
    """
    print("\n## Future Directions and Tips\n")

    print("### Future Directions")
    print("- Large Language Models: Use LLaMA or GPT-4 for NLG tasks.")
    print("- Few-Shot Learning: Train with fewer examples for efficiency.")
    print("- Multilingual NLG: Extend datasets to non-English languages.")
    print("- Real-Time NLG: Develop systems for live data (e.g., sports scores).\n")

    print("### Tips for Scientists")
    print(
        "- Reproducibility: Use Jupyter Notebooks or scripts for shareable experiments."
    )
    print("- Version Control: Store code on GitHub.")
    print("- Read Widely: Follow arXiv and Papers With Code for NLG advancements.")
    print("- Experiment: Start with small datasets, then scale up.")
    print("- Network: Join NLP communities on X or Reddit (e.g., r/MachineLearning).\n")

    print("### Missing Topics for Scientists")
    print(
        "- Evaluation Metrics: Explore BLEURT or ROUGE beyond BLEU for semantic similarity."
    )
    print("- Data Preprocessing: Clean and normalize messy real-world data.")
    print(
        "- Model Interpretability: Understand model decisions (e.g., attention mechanisms)."
    )
    print(
        "- Ethical Considerations: Address biases in data (e.g., DBpedia’s demographic coverage)."
    )
    print("- Scalability: Build efficient data pipelines for large datasets.")


def main():
    """
    Runs the full NLG tutorial interactively.
    """
    print("# Comprehensive NLG Tutorial: WebNLG, DART, and ToTTo\n")
    print(
        "This tutorial covers theory, code, visualizations, applications, and research for aspiring scientists."
    )
    print("Select an option to proceed:\n")

    options = {
        "1": ("Theory", print_nlg_theory),
        "2": ("Inference", run_inference),
        "3": ("Evaluate BLEU", evaluate_bleu),
        "4": ("Fine-Tune Model", fine_tune_model),
        "5": (
            "Visualizations",
            lambda: [plot_dataset_sizes(), plot_model_performance()],
        ),
        "6": (
            "Applications and Projects",
            lambda: [
                print_applications(),
                print_projects(),
                print_research_directions(),
            ],
        ),
        "7": ("Future Directions and Tips", print_future_directions),
    }

    while True:
        print("\nOptions:")
        for key, (desc, _) in options.items():
            print(f"{key}: {desc}")
        print("0: Exit")

        choice = input("\nEnter your choice (0-7): ")
        if choice == "0":
            print("Exiting tutorial. Happy researching!")
            break
        elif choice in options:
            print(f"\nRunning: {options[choice][0]}")
            options[choice][1]()
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()

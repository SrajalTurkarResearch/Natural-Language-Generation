# exercise_solutions.py
# Purpose: Solutions to exercises on realization and NLI
# For aspiring scientists: Practice and verify your understanding
# Dependencies: transformers (pip install transformers)

from transformers import pipeline


def realize_exercise(structure):
    """
    Exercise 1: Realize a sentence from a semantic structure.
    Args:
        structure (dict): Contains 'who', 'action', 'time'
    Returns:
        str: Sentence
    """
    return f"{structure['who']} {structure['action']}s."


def check_entailment(premise, hypothesis, model):
    """
    Check entailment for exercise 2.
    Args:
        premise (str): Starting fact
        hypothesis (str): Text to verify
        model: NLI pipeline
    Returns:
        bool: True if entailment score > 0.7
    """
    input_text = f"{premise} [SEP] {hypothesis}"
    result = model(input_text)
    return result[0]["label"] == "entailment" and result[0]["score"] > 0.7


# Example usage
if __name__ == "__main__":
    # Exercise 1
    print("Exercise 1 Solution:")
    print(
        realize_exercise({"who": "John", "action": "run", "time": "present"})
    )  # John runs.

    # Exercise 2
    print("\nExercise 2 Solution:")
    nli_model = pipeline("text-classification", model="facebook/bart-large-mnli")
    print(
        check_entailment("The sun is shining", "It’s bright outside", nli_model)
    )  # Expected: True

# simple_realization.py
# Purpose: Convert a semantic structure into a grammatically correct sentence
# For aspiring scientists: A basic example of NLG realization
# Dependencies: None


def realize_sentence(semantic_structure):
    """
    Convert a semantic structure (dict) into a sentence.
    Args:
        semantic_structure (dict): Contains 'who', 'action', 'object', 'time'
    Returns:
        str: Grammatically correct sentence
    """
    who = semantic_structure["who"]
    action = semantic_structure["action"]
    obj = semantic_structure["object"]
    time = semantic_structure["time"]

    # Apply grammar rules based on time
    if time == "past":
        if action == "buy":  # Example rule for past tense
            action = "bought"
    return f"{who} {action} the {obj}."


# Example usage
if __name__ == "__main__":
    structure = {"who": "Sarah", "action": "buy", "object": "book", "time": "past"}
    print(realize_sentence(structure))  # Output: Sarah bought the book.

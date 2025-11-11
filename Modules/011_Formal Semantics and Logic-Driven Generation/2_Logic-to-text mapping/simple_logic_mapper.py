# simple_logic_mapper.py
# A simple rule-based logic-to-text mapper for NLG
# Purpose: Convert a single predicate (e.g., Eats(John, Apple)) to text (John eats apple.)
# For aspiring scientists: Start here to understand logic-to-text basics, like Turing's clear code steps.

# No external dependencies required
# Usage: Run with Python 3 (e.g., `python simple_logic_mapper.py`)


def simple_logic_to_text(logic):
    """
    Convert a single predicate logic to a natural language sentence.

    Args:
        logic (dict): Logic in format {'type': 'Predicate', 'subject': str, 'verb': str, 'object': str}

    Returns:
        str: Natural language sentence (e.g., "John eats apple.")

    Example:
        >>> logic = {'type': 'Predicate', 'subject': 'John', 'verb': 'eats', 'object': 'apple'}
        >>> simple_logic_to_text(logic)
        'John eats apple.'
    """
    if logic["type"] == "Predicate":
        return f"{logic['subject']} {logic['verb']} {logic['object']}."
    return "Unknown logic type."


# Test the function
if __name__ == "__main__":
    # Example logic
    test_logic = {
        "type": "Predicate",
        "subject": "John",
        "verb": "eats",
        "object": "apple",
    }
    print("Input logic:", test_logic)
    print("Output text:", simple_logic_to_text(test_logic))

    # Try your own logic
    your_logic = {
        "type": "Predicate",
        "subject": "Alice",
        "verb": "owns",
        "object": "dog",
    }
    print("\nYour logic:", your_logic)
    print("Your output:", simple_logic_to_text(your_logic))

    # Research Lab: Try modifying this to handle verbs needing articles (e.g., "an apple").
    # Example idea: Add a rule to check if object starts with a vowel.

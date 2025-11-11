# advanced_logic_mapper.py
# An advanced rule-based logic-to-text mapper for NLG
# Purpose: Handle combined logic (e.g., AND) to create sentences like "John eats apple and likes fruit."
# For aspiring scientists: Build on simple mapper, like Tesla scaling up inventions.

# No external dependencies required
# Usage: Run with Python 3 (e.g., `python advanced_logic_mapper.py`)


def advanced_logic_to_text(logic):
    """
    Convert logic (predicate or AND) to a natural language sentence.

    Args:
        logic (dict): Logic in format {'type': 'Predicate', 'subject': str, 'verb': str, 'object': str}
                     or {'type': 'AND', 'left': dict, 'right': dict}

    Returns:
        str: Natural language sentence (e.g., "John eats apple and likes fruit.")

    Example:
        >>> logic = {'type': 'AND', 'left': {'type': 'Predicate', 'subject': 'John', 'verb': 'eats', 'object': 'apple'},
        ...          'right': {'type': 'Predicate', 'subject': 'John', 'verb': 'likes', 'object': 'fruit'}}
        >>> advanced_logic_to_text(logic)
        'John eats apple and likes fruit.'
    """
    if logic["type"] == "Predicate":
        return f"{logic['subject']} {logic['verb']} {logic['object']}."
    elif logic["type"] == "AND":
        left_text = advanced_logic_to_text(logic["left"])
        right_text = advanced_logic_to_text(logic["right"])
        return f"{left_text[:-1]} and {right_text.lower()}"
    return "Unknown logic type."


# Test the function
if __name__ == "__main__":
    # Example logic with AND
    test_logic = {
        "type": "AND",
        "left": {
            "type": "Predicate",
            "subject": "John",
            "verb": "eats",
            "object": "apple",
        },
        "right": {
            "type": "Predicate",
            "subject": "John",
            "verb": "likes",
            "object": "fruit",
        },
    }
    print("Input logic:", test_logic)
    print("Output text:", advanced_logic_to_text(test_logic))

    # Try your own logic
    your_logic = {
        "type": "AND",
        "left": {
            "type": "Predicate",
            "subject": "Alice",
            "verb": "owns",
            "object": "dog",
        },
        "right": {
            "type": "Predicate",
            "subject": "Alice",
            "verb": "feeds",
            "object": "dog",
        },
    }
    print("\nYour logic:", your_logic)
    print("Your output:", advanced_logic_to_text(your_logic))

    # Research Lab: Extend this to handle "IF" logic (e.g., "If hungry, John eats apple").
    # Example idea: Add a condition check for 'IF' type in the logic dictionary.

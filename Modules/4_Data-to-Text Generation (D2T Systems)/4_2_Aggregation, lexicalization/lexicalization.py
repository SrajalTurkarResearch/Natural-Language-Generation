# Lexicalization in NLG: Choosing the Right Words
#
# Theory:
# - Lexicalization is selecting appropriate words or phrases to express data or concepts, ensuring text is natural and audience-appropriate.
# - Purpose: Translate abstract data into readable text, adapt to audience, ensure clarity.
# - Types:
#   - Concept-to-Word: Map data to descriptive words (e.g., 15°C → "cool").
#   - Entity Naming: Use specific names/pronouns (e.g., "John" vs. "Person").
#   - Verb Selection: Choose accurate verbs (e.g., "speeds up" vs. "increases").
#   - Contextual Adaptation: Adjust for audience (e.g., "high" for public, "hypertension" for doctors).
# - Example:
#   Input: Temp = 15°C
#   Output: It's cool at 15°C.
# - Analogy: Lexicalization is like choosing the right outfit for an occasion—words must fit the context.
#
# This file implements a lexicalization function for temperature data.
# For your research: Explore probabilistic models (e.g., language models) for advanced lexicalization.


def lexicalize_temperature(temp):
    """
    Maps temperature values to descriptive words for a general audience.

    Args:
        temp (int/float): Temperature in Celsius.
    Returns:
        str: Descriptive word (e.g., 'cold', 'warm').
    """
    if temp < 10:
        return "cold"
    elif 10 <= temp <= 20:
        return "cool"
    elif 21 <= temp <= 27:
        return "warm"
    else:
        return "hot"


def generate_weather_sentence(data):
    """
    Generates a sentence using lexicalized temperature description.

    Args:
        data (dict): Dictionary with 'Temperature' key.
    Returns:
        str: Lexicalized sentence.
    """
    temp_desc = lexicalize_temperature(data["Temperature"])
    return f"It's {temp_desc} at {data['Temperature']}°C today."


# Example usage
if __name__ == "__main__":
    data = {"Temperature": 22}
    print(generate_weather_sentence(data))
    # Output: It's warm at 22°C today.

    # Research Tip: Extend this to lexicalize other attributes (e.g., wind speed → "breezy").

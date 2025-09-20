# Aggregation in NLG: Combining Data Points into Concise Sentences
#
# Theory:
# - Aggregation is the process of combining multiple data points into a single, concise sentence to reduce redundancy and improve readability.
# - Purpose: Avoid repetition, enhance coherence, and make text natural.
# - Types:
#   - Simple Conjunction: Use "and" or "but" to join facts.
#   - Syntactic Coordination: Combine facts with shared subjects/verbs.
#   - Ellipsis: Remove redundant words.
#   - Set Introduction: List items as a group.
#   - Temporal/Causal: Combine based on time or cause-effect.
# - Example:
#   Input: Temp = 22°C, Sky = Sunny
#   Output: It's sunny with a temperature of 22°C.
# - Analogy: Aggregation is like packing a suitcase efficiently, combining items to save space.
#
# This file implements a basic aggregation function for weather data.
# For your research: Experiment with different aggregation types to optimize readability.

import pandas as pd


def aggregate_weather(data):
    """
    Aggregates weather data into a single sentence using simple conjunction and syntactic coordination.

    Args:
        data (dict): Dictionary with keys 'Temperature', 'Sky', 'Wind', 'Humidity'.
    Returns:
        str: Aggregated sentence.
    """
    sentence = (
        f"It's {data['Sky'].lower()} with a temperature of {data['Temperature']}°C"
    )
    sentence += (
        f", wind speed of {data['Wind']} km/h, and {data['Humidity']}% humidity."
    )
    return sentence


# Example usage
if __name__ == "__main__":
    # Sample weather data
    data = {"Temperature": 22, "Sky": "Sunny", "Wind": 5, "Humidity": 60}
    print(aggregate_weather(data))
    # Output: It's sunny with a temperature of 22°C, wind speed of 5 km/h, and 60% humidity.

    # Research Tip: Try modifying this to include temporal aggregation (e.g., "It was rainy this morning but sunny now").

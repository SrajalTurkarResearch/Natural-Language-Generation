# rule_based_nlg.py
# A simple rule-based NLG system for weather reports.
# Author: Inspired by symbolic AI pioneers like Alan Turing.
# Usage: Run this file to test with sample data.


def rule_based_nlg(data):
    """
    Generate text using a fixed template rule.
    Args:
        data (dict): Input data with 'city' and 'temp' keys.
    Returns:
        str: Generated text.
    """
    # Rule: Simple template filling for accuracy and explainability.
    return f"The temperature in {data['city']} is {data['temp']}°C."


if __name__ == "__main__":
    # Test data, like an experiment input.
    sample_data = {"city": "Paris", "temp": 25}
    print("Rule-Based Output:", rule_based_nlg(sample_data))
    # Experiment: Try changing the data and observe consistency.

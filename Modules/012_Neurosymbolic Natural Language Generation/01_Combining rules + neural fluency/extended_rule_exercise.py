# extended_rule_exercise.py
# Exercise solution: Extended rule for conditions.
# Author: Building skills like a mathematician deriving theorems.
# Usage: Run to test conditional rules.


def extended_rule(data):
    """
    Rule-based with conditions for varied outputs.
    Args:
        data (dict): Input.
    Returns:
        str: Conditional text.
    """
    if data["temp"] < 10:
        return f"It's cold in {data['city']} at {data['temp']}°C."
    else:
        return f"The temperature in {data['city']} is {data['temp']}°C."


if __name__ == "__main__":
    # Test cases.
    cold_data = {"city": "Paris", "temp": 5}
    print("Extended Rule Output (Cold):", extended_rule(cold_data))
    warm_data = {"city": "Paris", "temp": 25}
    print("Extended Rule Output (Warm):", extended_rule(warm_data))
    # Exercise: Add more conditions (e.g., hot >30) and test logic.

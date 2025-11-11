# nlg_rule_based.py
# Simple Rule-Based Natural Language Generation
# Author: Your Name (Future Scientist)
# Date: November 11, 2025


def generate_weather_report(data):
    """
    Generate a weather report from structured data.

    Args:
        data (dict): Contains 'temp' (int) and 'condition' (str)

    Returns:
        str: Human-readable weather report
    """
    temp = data["temp"]
    condition = data["condition"].lower()

    # Content Planning: Choose description
    if temp >= 35:
        temp_desc = "very hot"
    elif temp >= 30:
        temp_desc = "hot"
    elif temp >= 25:
        temp_desc = "warm"
    elif temp >= 15:
        temp_desc = "pleasant"
    else:
        temp_desc = "cool"

    # Sentence Planning & Realization
    report = f"Today is {condition} with a {temp_desc} temperature of {temp}°C."
    return report


# === TEST THE SYSTEM ===
if __name__ == "__main__":
    test_data = [
        {"temp": 38, "condition": "sunny"},
        {"temp": 22, "condition": "cloudy"},
        {"temp": 12, "condition": "rainy"},
    ]

    print("=== Weather Reports ===\n")
    for data in test_data:
        print(generate_weather_report(data))
        print("-" * 50)

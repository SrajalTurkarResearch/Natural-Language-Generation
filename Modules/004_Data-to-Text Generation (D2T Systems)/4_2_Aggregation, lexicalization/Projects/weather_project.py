# Mini Project: Weather Report Generator
#
# Theory:
# - This project combines aggregation and lexicalization to generate weather reports from structured data.
# - Aggregation: Combines multiple weather metrics (Temp, Sky, Wind) into one sentence.
# - Lexicalization: Maps numerical data to descriptive words (e.g., 20°C → "warm").
# - Application: Useful for weather apps or automated forecasting systems.
# - Analogy: Like preparing a daily weather briefing, combining facts into a concise, clear report.
#
# Requirements: Install pandas (`pip install pandas`).
# For your research: Extend this to handle multiple cities or dynamic data sources.

import pandas as pd


def lexicalize_temperature(temp):
    """
    Maps temperature to descriptive words.

    Args:
        temp (int/float): Temperature in Celsius.
    Returns:
        str: Descriptive word.
    """
    if temp < 10:
        return "cold"
    elif 10 <= temp <= 20:
        return "cool"
    elif 21 <= temp <= 27:
        return "warm"
    else:
        return "hot"


def generate_report(row):
    """
    Generates a weather report by aggregating and lexicalizing data.

    Args:
        row (pandas.Series): Row with 'City', 'Temperature', 'Sky', 'Wind'.
    Returns:
        str: Weather report sentence.
    """
    temp_desc = lexicalize_temperature(row["Temperature"])
    return f"In {row['City']}, it's {temp_desc} at {row['Temperature']}°C with {row['Sky'].lower()} skies and winds at {row['Wind']} km/h."


# Example usage
if __name__ == "__main__":
    # Sample data
    data = pd.DataFrame(
        {
            "City": ["New York", "London"],
            "Temperature": [20, 15],
            "Sky": ["Cloudy", "Sunny"],
            "Wind": [10, 5],
        }
    )

    for _, row in data.iterrows():
        print(generate_report(row))
    # Output:
    # In New York, it's cool at 20°C with cloudy skies and winds at 10 km/h.
    # In London, it's cool at 15°C with sunny skies and winds at 5 km/h.

    # Research Tip: Add temporal aggregation (e.g., "It was rainy earlier but sunny now").

# major_project.py
# Purpose: Generate documentation for a climate data function using a CSV dataset.
# Why: Scientists document code (e.g., climate models) for clear, shareable results.
# Requires: pandas (run setup.py first).

import pandas as pd


def average_temp(data_path):
    """
    Calculates average temperature from a CSV file.

    Args:
        data_path (str): Path to CSV with 'temperature' column.

    Returns:
        float: Mean temperature in degrees.

    Example:
        >>> average_temp('climate_data.csv')
        22.333333333333332
    """
    df = pd.read_csv(data_path)
    return df["temperature"].mean()


# Create fake dataset for demo
data = pd.DataFrame({"temperature": [20, 22, 25]})
data.to_csv("climate_data.csv", index=False)

# Test the function
try:
    result = average_temp("climate_data.csv")
    print(f"Average Temperature: {result} degrees")
except Exception as e:
    print(f"Error: {e}")

# Why this matters: Auto-docs for climate code speed up research sharing.
# For science: Use for environmental studies or data analysis papers.
# Try it: Add a new column (e.g., 'humidity') to CSV and update function/doc.

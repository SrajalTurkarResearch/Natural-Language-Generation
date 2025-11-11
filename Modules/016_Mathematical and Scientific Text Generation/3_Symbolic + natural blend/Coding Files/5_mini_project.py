# 5_mini_project.py
# Mini Project: Multi-city weather report generator

import pandas as pd
from utils import hybrid_nlg


def generate_city_reports():
    """Generate hybrid reports for multiple cities."""
    data = pd.DataFrame(
        [
            {"city": "New York", "temp": 72, "condition": "sunny"},
            {"city": "London", "temp": 48, "condition": "rainy"},
            {"city": "Dubai", "temp": 108, "condition": "clear"},
            {"city": "Moscow", "temp": 28, "condition": "snowy"},
        ]
    )

    print("WEATHER REPORTS\n" + "=" * 50)
    for _, row in data.iterrows():
        text = f"Forecast for {row['city']}."
        report = hybrid_nlg(row, text)
        print(f"{row['city']}: {report}\n")


# === RUN ===
if __name__ == "__main__":
    generate_city_reports()

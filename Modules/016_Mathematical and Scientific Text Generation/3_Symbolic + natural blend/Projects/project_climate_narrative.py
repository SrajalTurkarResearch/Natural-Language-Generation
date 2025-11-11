# project_climate_narrative.py
# Real-World: Turn Climate Data into Public Reports
# Use: IPCC, NASA, Environmental NGOs

import pandas as pd
import matplotlib.pyplot as plt
from utils import neural_summary_nlg


def climate_narrative_nlg(data_csv="climate_data.csv"):
    """
    Read temperature trends → generate public report
    """
    df = pd.read_csv(data_csv) if __name__ == "__main__" else pd.DataFrame()

    # === SAMPLE DATA (if no CSV) ===
    if df.empty:
        years = list(range(2010, 2024))
        temps = [
            14.1,
            14.3,
            14.2,
            14.5,
            14.6,
            14.8,
            14.7,
            15.0,
            15.1,
            15.3,
            15.2,
            15.5,
            15.6,
            15.9,
        ]
        df = pd.DataFrame({"year": years, "temp_anomaly": temps})

    # === SYMBOLIC: Trend Analysis ===
    trend = (
        "rising"
        if df["temp_anomaly"].iloc[-1] > df["temp_anomaly"].mean()
        else "stable"
    )
    increase = df["temp_anomaly"].iloc[-1] - df["temp_anomaly"].iloc[0]

    # === NEURAL: Public Narrative ===
    narrative = neural_summary_nlg(
        f"Global temperature anomaly from {df['year'].iloc[0]} to {df['year'].iloc[-1]}. "
        f"Average was {df['temp_anomaly'].mean():.2f}°C above baseline. "
        f"Recent years show warming."
    )

    report = (
        f"CLIMATE UPDATE\n"
        f"Temperature trend: {trend.upper()} (+{increase:.2f}°C over period)\n\n"
        f"{narrative}"
    )

    # === PLOT ===
    plt.figure(figsize=(10, 5))
    plt.plot(df["year"], df["temp_anomaly"], marker="o", color="red")
    plt.title("Global Temperature Anomaly")
    plt.xlabel("Year")
    plt.ylabel("°C above baseline")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return report


# === RUN ===
if __name__ == "__main__":
    print(climate_narrative_nlg())

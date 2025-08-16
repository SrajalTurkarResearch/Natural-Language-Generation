# basic_nlg.py
# A rule-based NLG system for generating weather reports with preprocessing and visualization
# Designed for beginners learning NLG for scientific research

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


# Preprocessing function to clean and validate input data
def preprocess_data(data):
    """Clean and validate weather data."""
    processed_data = data.copy()
    # Round temperature to nearest integer
    processed_data["temperature"] = round(data["temperature"])
    # Ensure condition is lowercase and valid
    valid_conditions = ["sunny", "cloudy", "rainy", "snowy"]
    processed_data["condition"] = (
        data["condition"].lower()
        if data["condition"].lower() in valid_conditions
        else "unknown"
    )
    # Ensure humidity is between 0 and 100
    processed_data["humidity"] = max(0, min(100, data["humidity"]))
    return processed_data


# NLG function to generate weather report
def generate_weather_report(data):
    """Generate a weather report from structured data."""
    # Content selection and structuring
    temp = data["temperature"]
    condition = data["condition"]
    humidity = data["humidity"]

    report = []
    report.append(f"The temperature is {temp}°C.")
    report.append(f"It is {condition} today.")
    report.append(f"The humidity is {humidity}%.")

    # Surface realization
    final_report = " ".join(report)
    return final_report


# Visualization of NLG pipeline
def visualize_pipeline():
    """Visualize the NLG pipeline using Matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 2))
    stages = [
        "Raw Data",
        "Preprocessing",
        "Content Selection",
        "Content Structuring",
        "Sentence Planning",
        "Surface Realization",
        "Text Output",
    ]
    for i, stage in enumerate(stages):
        ax.text(i * 0.2, 0.5, stage, rotation=45, ha="left", va="bottom")
        if i < len(stages) - 1:
            arrow = FancyArrowPatch(
                (i * 0.2 + 0.1, 0.4), ((i + 1) * 0.2 - 0.05, 0.4), mutation_scale=15
            )
            ax.add_patch(arrow)
    ax.set_xlim(0, 1.4)
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.title("NLG Pipeline")
    plt.savefig("nlg_pipeline.png")  # Save visualization
    plt.show()


# Main execution
if __name__ == "__main__":
    # Sample input data
    weather_data = {"temperature": 25.6, "condition": "Sunny", "humidity": 60}

    # Preprocess data
    processed_data = preprocess_data(weather_data)
    print("Processed Data:", processed_data)

    # Generate report
    report = generate_weather_report(processed_data)
    print("Generated Weather Report:", report)

    # Visualize pipeline
    visualize_pipeline()

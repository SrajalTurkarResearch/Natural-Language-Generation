# weather_report_generator.py
# Mini Project: Generate weather reports using the Classical NLG Pipeline
# Designed for beginners to practice NLG

# Install required libraries: pip install pandas
import pandas as pd

# Sample dataset
data = pd.DataFrame(
    {
        "day": ["Monday", "Tuesday"],
        "temperature": [75, 80],
        "condition": ["sunny", "cloudy"],
        "humidity": [60, 65],
        "wind_speed": [10, 15],
    }
)


# Full NLG Pipeline for Weather Report
def weather_report_generator(data, day):
    """Generate a weather report for a given day."""
    # Stage 1: Content Determination
    row = data[data["day"] == day].iloc[0]
    selected = {"temperature": row["temperature"], "condition": row["condition"]}
    print("Selected Data:", selected)

    # Stage 2: Document Planning
    plan = {
        "introduction": f"Weather for {day}",
        "body": selected,
        "conclusion": "Plan your day!",
    }
    print("Document Plan:", plan)

    # Stage 3: Microplanning
    sentence = f"{plan['introduction']} is {plan['body']['condition']} with a high of {plan['body']['temperature']}°F."
    conclusion = (
        "Great for outdoor activities!"
        if plan["body"]["condition"] == "sunny"
        else "Bring an umbrella!"
    )
    microplanned = {"sentence": sentence, "conclusion": conclusion}
    print("Microplanned Content:", microplanned)

    # Stage 4: Surface Realization
    realized = sentence
    print("Realized Sentence:", realized)

    # Stage 5: Post-Processing
    polished = realized.capitalize() + " " + conclusion
    print("Final Text:", polished)

    # Stage 6: Output
    print("Generated Weather Report:")
    print(polished)
    return polished


# Run the mini project
if __name__ == "__main__":
    report = weather_report_generator(data, "Monday")
    print("Mini Project Completed!")

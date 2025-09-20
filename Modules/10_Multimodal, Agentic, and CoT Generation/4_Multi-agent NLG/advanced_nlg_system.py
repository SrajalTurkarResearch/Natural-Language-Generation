# advanced_nlg_system.py
# An advanced Multi-Agent NLG system with a learning agent
# Purpose: Extend the basic system with tone adaptation based on feedback
# Author: Inspired by Turing, Einstein, and Tesla for aspiring scientists
# Prerequisites: Python 3.x

"""
This script builds on the basic NLG system by adding a Learning Agent that adjusts
the tone based on user feedback. It demonstrates how agents can improve over time.
The system generates a weather report with a focus on professional output.
"""


def data_agent(city):
    """
    Data Agent: Simulates collecting weather data.
    Input: city (str) - Name of the city.
    Output: Dictionary with weather data.
    """
    return {
        "city": city,
        "temp": 25,
        "condition": "clear",
        "wind_speed": 10,
        "forecast": {"morning": "sunny", "afternoon": "cloudy", "evening": "clear"},
    }


def planning_agent(data):
    """
    Planning Agent: Structures the report professionally.
    Input: data (dict) - Weather data.
    Output: Dictionary defining report structure.
    """
    return {
        "introduction": f"Current weather conditions for {data['city']}",
        "current_conditions": ["temp", "condition", "wind_speed"],
        "forecast_section": ["morning", "afternoon", "evening"],
    }


def generation_agent(data, plan):
    """
    Generation Agent: Creates professional sentences.
    Input: data (dict) - Weather data; plan (dict) - Report structure.
    Output: List of sentences.
    """
    sentences = []
    sentences.append(plan["introduction"] + ":")
    for item in plan["current_conditions"]:
        if item == "temp":
            sentences.append(f"The temperature is {data[item]} degrees Celsius.")
        elif item == "condition":
            sentences.append(f"The sky is {data[item]}.")
        elif item == "wind_speed":
            sentences.append(f"Wind speed is {data[item]} kilometers per hour.")
    sentences.append("Forecast for tomorrow:")
    for period in plan["forecast_section"]:
        sentences.append(f"{period.capitalize()}: {data['forecast'][period]}.")
    return sentences


def refinement_agent(sentences):
    """
    Refinement Agent: Polishes text for clarity and professionalism.
    Input: sentences (list) - Draft sentences.
    Output: String with polished text.
    """
    polished = []
    for sentence in sentences:
        polished_sentence = sentence[0].upper() + sentence[1:]
        if not polished_sentence.endswith("."):
            polished_sentence += "."
        polished.append(polished_sentence)
    return "\n".join(polished)


def learning_agent(feedback_score, current_tone):
    """
    Learning Agent: Adapts tone based on feedback.
    Input: feedback_score (float) - 0 (poor) to 1 (excellent); current_tone (float) - 0 (casual) to 1 (formal).
    Output: Updated tone score.
    """
    learning_rate = 0.1
    new_tone = current_tone + learning_rate * (feedback_score - current_tone)
    return max(0, min(1, new_tone))


def main():
    """
    Main function to coordinate the advanced Multi-Agent NLG system.
    """
    city = "Paris"
    feedback_score = 0.8
    initial_tone = 0.5
    data = data_agent(city)
    plan = planning_agent(data)
    draft = generation_agent(data, plan)
    final_report = refinement_agent(draft)
    updated_tone = learning_agent(feedback_score, initial_tone)
    print("Generated Weather Report:")
    print(final_report)
    print(f"\nUpdated tone score: {updated_tone:.2f}")


if __name__ == "__main__":
    main()

# Try This: Modify the city or feedback_score. Add a new data point (e.g., humidity).
# Next Steps: Integrate real API data or add a bias-checking agent.

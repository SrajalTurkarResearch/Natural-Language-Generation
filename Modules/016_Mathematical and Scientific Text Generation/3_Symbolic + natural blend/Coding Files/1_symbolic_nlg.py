# 1_symbolic_nlg.py
# Pure Symbolic NLG: Rule-based weather report generator
# Like a strict teacher using only logic and rules


def symbolic_weather_nlg(data):
    """
    Generate weather description using if-then rules.
    Input: dict with 'temp' and 'condition'
    Output: natural language sentence
    """
    temp = data.get("temp", 0)
    condition = data.get("condition", "clear")

    # Symbolic rules: Content planning
    if temp > 70:
        temp_desc = "warm"
    elif temp < 50:
        temp_desc = "cool"
    else:
        temp_desc = "mild"

    # Sentence realization
    sentence = f"It's a {temp_desc}, {condition} day."
    return sentence


# === TEST ===
if __name__ == "__main__":
    data1 = {"temp": 75, "condition": "sunny"}
    data2 = {"temp": 45, "condition": "cloudy"}

    print(symbolic_weather_nlg(data1))
    print(symbolic_weather_nlg(data2))

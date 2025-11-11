# 7_exercises.py
# Practice exercises with solutions


def exercise_1_wind_rule(data):
    """Add wind speed rule: >20 mph = windy"""
    temp = data.get("temp", 0)
    wind = data.get("wind", 0)
    condition = data.get("condition", "clear")

    temp_desc = "warm" if temp > 70 else "cool" if temp < 50 else "mild"
    wind_desc = "windy" if wind > 20 else "calm"

    return f"It's a {temp_desc}, {condition}, {wind_desc} day."


def exercise_2_custom_summary():
    """Change input to summarize a book"""
    from utils import neural_summary_nlg

    book = (
        "The Theory of Everything by Stephen Hawking explains "
        "black holes, the Big Bang, and the nature of time in simple terms."
    )
    return neural_summary_nlg(book)


# === SOLUTIONS ===
if __name__ == "__main__":
    print("Exercise 1: Wind Rule")
    print(exercise_1_wind_rule({"temp": 75, "condition": "sunny", "wind": 25}))

    print("\nExercise 2: Book Summary")
    print(exercise_2_custom_summary())

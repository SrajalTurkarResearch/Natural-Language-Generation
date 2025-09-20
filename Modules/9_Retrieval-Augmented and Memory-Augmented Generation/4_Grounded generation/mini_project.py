# mini_project.py: Weather Report Generator
# This is a simple grounded NLG system for weather data.
# Theory: Uses number-based grounding (see theory.py #3).
# Run in Python 3.8+.

# Weather data (like a small database)
weather_data = [
    {"city": "Paris", "temp": 25, "condition": "sunny"},
    {"city": "London", "temp": 18, "condition": "cloudy"},
]


def weather_nlg(city):
    # Find matching city data
    for data in weather_data:
        if data["city"].lower() == city.lower():
            return f"In {data['city']}, it's {data['condition']} with a temperature of {data['temp']}°C."
    return "City not found."


# Test
print(
    "Mini Project:", weather_nlg("Paris")
)  # Output: In Paris, it's sunny with a temperature of 25°C.

# Task for You:
# 1. Add 3 more cities to weather_data.
# 2. Change output to a tweet format (e.g., "Paris: Sunny, 25°C! #Weather").
# 3. Think: How could this help climate scientists summarize data?

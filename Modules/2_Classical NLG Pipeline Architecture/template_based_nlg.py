# template_based_nlg.py
# Template-based NLG for generating structured text
# Additional topic for classical NLG pipelines


# Template-Based NLG
def template_based_nlg(data, template):
    """Generate text using a predefined template."""
    try:
        return template.format(**data)
    except KeyError as e:
        return f"Error: Missing data for {e}"


# Example Usage
if __name__ == "__main__":
    # Weather template
    weather_template = "{day} is {condition} with a high of {temperature}°F."
    weather_data = {"day": "Monday", "condition": "sunny", "temperature": 75}
    weather_output = template_based_nlg(weather_data, weather_template)
    print("Weather Report:", weather_output)

    # Sports template
    sports_template = "{team_a} defeated {team_b} {score}, with {key_player} scoring {player_goals} goals."
    sports_data = {
        "team_a": "Lions",
        "team_b": "Tigers",
        "score": "3-2",
        "key_player": "Alex",
        "player_goals": 2,
    }
    sports_output = template_based_nlg(sports_data, sports_template)
    print("Sports Summary:", sports_output)

# basic_nlg_system.py
# A simple Multi-Agent NLG system for generating a weather report
# Purpose: Demonstrate core concepts of Multi-Agent NLG for beginners
# Author: Inspired by Turing, Einstein, and Tesla for aspiring scientists
# Prerequisites: Python 3.x

"""
This script implements a basic Multi-Agent NLG system with three agents:
- Data Agent: Collects weather data (simulated).
- Planning Agent: Structures the report.
- Generation Agent: Creates the text output.
Run this script to see a simple weather report and experiment by changing the city.
"""


def data_agent(city):
    """
    Data Agent: Simulates collecting weather data.
    Input: city (str) - Name of the city.
    Output: Dictionary with weather data.
    """
    return {"city": city, "temp": 25, "condition": "clear"}


def planning_agent(data):
    """
    Planning Agent: Defines the structure of the report.
    Input: data (dict) - Weather data from Data Agent.
    Output: List defining report sections.
    """
    return ["intro", "current"]


def generation_agent(data, plan):
    """
    Generation Agent: Creates text based on data and plan.
    Input: data (dict) - Weather data; plan (list) - Report structure.
    Output: String with the generated text.
    """
    return f"Weather in {data['city']}: {data['condition']} at {data['temp']}°C."


def main():
    """
    Main function to coordinate the Multi-Agent NLG system.
    """
    city = "Paris"
    data = data_agent(city)
    plan = planning_agent(data)
    output = generation_agent(data, plan)
    print("Generated Weather Report:")
    print(output)


if __name__ == "__main__":
    main()

# Try This: Change 'Paris' to another city (e.g., 'London') and run the script.
# Next Steps: Add more data (e.g., wind speed) or a refinement agent to polish the text.

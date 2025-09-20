# weather_report_generator.py
# Tutorial Component: Mini Project - Weather Report Generator
# Purpose: Generate natural language weather reports from structured Meaning Representations (MR)
# Theory: Meaning Representations (MRs) capture the core meaning of data (e.g., location, condition, temperature)
#         in a structured format, enabling NLG systems to produce human-readable text. This mini project
#         demonstrates how a simple MR (dictionary format) can be converted to text using a rule-based
#         NLG system, a key skill for NLP researchers.

# For Scientists: This code is a starting point for building domain-specific NLG systems (e.g., medical reports).
#                 Experiment with different MR structures and templates to explore flexibility in NLG.

# Setup Instructions:
# 1. Install required library: `pip install nltk`
# 2. Run this file: `python weather_report_generator.py`
# 3. Ensure NLTK data is downloaded (code included below)

import nltk

# Download NLTK data (run once)
nltk.download("punkt", quiet=True)


# Function: Generate weather report from MR
def weather_report_generator(data):
    """
    Convert a Meaning Representation (MR) to a natural language weather report.
    MR Example: {'location': 'London', 'condition': 'rainy', 'temperature': 20}
    Theory: The MR is a structured dictionary capturing key facts (location, condition, temperature).
            NLG maps these to a template to produce text, demonstrating content planning and surface realization.
    """
    # Template for text generation (surface realization)
    template = "It's {condition} in {location} with a temperature of {temperature}°C."
    return template.format(**data)


# Real-World Application: Weather apps (e.g., BBC Weather) use MRs to generate multilingual reports.
# Research Direction: Extend this to handle dynamic data (e.g., real-time sensor inputs).
# Tip for Scientists: Save outputs to a log file to analyze variations in generated text.

# Test the generator
if __name__ == "__main__":
    # Example MR (content planning)
    weather_data = {"location": "London", "condition": "rainy", "temperature": 20}
    print("Weather Report:", weather_report_generator(weather_data))

    # Additional test case
    weather_data2 = {"location": "Tokyo", "condition": "sunny", "temperature": 28}
    print("Weather Report:", weather_report_generator(weather_data2))

    # For Your Notebook: Note the MR structure and output. Try adding new fields (e.g., 'humidity')
    #                    and modify the template to include them.

# Future Direction: Integrate with APIs (e.g., OpenWeatherMap) to fetch real-time data for MRs.
# Missing from Tutorial: Evaluation metrics (e.g., BLEU score) to assess output quality.
# Next Steps: Experiment with different templates (formal vs. casual) and multilingual outputs.

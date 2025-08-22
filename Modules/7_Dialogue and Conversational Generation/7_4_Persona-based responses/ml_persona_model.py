# response_generator.py
"""
Generates persona-based responses using regex matching and probabilistic selection.
This module takes user input and a persona to produce tailored responses.
"""

import re
import numpy as np


class ResponseGenerator:
    """Class to generate persona-based responses."""

    def __init__(self, persona_manager):
        """
        Initialize with a PersonaManager instance.
        Args:
            persona_manager (PersonaManager): Instance holding persona configurations.
        """
        self.persona_manager = persona_manager

    def generate_response(self, query, persona_name, use_probabilistic=False):
        """
        Generate a response based on the query and persona.
        Args:
            query (str): User input question.
            persona_name (str): Name of the persona to use.
            use_probabilistic (bool): If True, select response randomly based on weights.
        Returns:
            str: Generated response.
        """
        persona = self.persona_manager.get_persona(persona_name)

        # Check for matching patterns
        response_options = []
        for pattern, template in persona["response_templates"].items():
            match = re.match(pattern, query.lower())
            if match:
                query_key = match.group(1) if match.groups() else query
                answer = persona["knowledge"].get(
                    query_key, "something I can look up for you"
                )
                response = template.format(query_key, answer)
                response_options.append(response)

        # Return greeting if no match
        if not response_options:
            return persona["greeting"]

        # Probabilistic selection if enabled
        if use_probabilistic and len(response_options) > 1:
            probabilities = [1.0 / len(response_options)] * len(
                response_options
            )  # Equal weights
            return np.random.choice(response_options, p=probabilities)

        return response_options[0]  # Default to first match


# Example usage
if __name__ == "__main__":
    from persona_manager import PersonaManager

    manager = PersonaManager()
    generator = ResponseGenerator(manager)

    # Test responses
    print(generator.generate_response("What is NLG?", "Friendly Teacher"))
    print(generator.generate_response("What is python?", "Sarcastic Engineer"))

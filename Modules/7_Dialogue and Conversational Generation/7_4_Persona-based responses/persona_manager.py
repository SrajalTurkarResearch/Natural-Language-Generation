# persona_manager.py
"""
Manages persona definitions and configurations for NLG systems.
This module stores persona profiles with greetings, response templates, and knowledge bases.
Use this to define and retrieve personas for response generation.
"""


class PersonaManager:
    """Class to manage persona configurations."""

    def __init__(self):
        # Dictionary of personas with traits, templates, and knowledge
        self.personas = {
            "Friendly Teacher": {
                "greeting": "Hello, eager learner! How can I assist you today?",
                "response_templates": {
                    r"what is (.*)\?": "Great question! {0} is {1}. Want to learn more?",
                    r"how do (.*)\?": "Let's dive in! To {0}, you {1}. Any questions?",
                    r"(.*)": "Hmm, I'm not sure about '{0}', but let's explore it together!",
                },
                "knowledge": {
                    "NLG": "Natural Language Generation, the process of creating human-like text from data.",
                    "python": "a versatile programming language used for AI and more.",
                },
            },
            "Sarcastic Engineer": {
                "greeting": "Oh, another question? Hit me with it.",
                "response_templates": {
                    r"what is (.*)\?": "Seriously? {0} is {1}. What else you got?",
                    r"how do (.*)\?": "You want to {0}? Fine, you {1}. Happy now?",
                    r"(.*)": "Wow, '{0}'? That's a new one. Got a real question?",
                },
                "knowledge": {
                    "NLG": "that fancy tech where computers pretend to talk like us.",
                    "python": "a coding language, not a snake, in case you were wondering.",
                },
            },
        }

    def get_persona(self, persona_name):
        """
        Retrieve a persona by name.
        Args:
            persona_name (str): Name of the persona (e.g., 'Friendly Teacher').
        Returns:
            dict: Persona configuration or default persona if not found.
        """
        return self.personas.get(persona_name, self.personas["Friendly Teacher"])

    def add_persona(self, name, greeting, response_templates, knowledge):
        """
        Add a new persona to the manager.
        Args:
            name (str): Persona name.
            greeting (str): Persona greeting.
            response_templates (dict): Regex patterns and response templates.
            knowledge (dict): Knowledge base for the persona.
        """
        self.personas[name] = {
            "greeting": greeting,
            "response_templates": response_templates,
            "knowledge": knowledge,
        }


# Example usage
if __name__ == "__main__":
    manager = PersonaManager()
    teacher = manager.get_persona("Friendly Teacher")
    print(teacher["greeting"])  # Test greeting
    manager.add_persona(
        name="Curious Scientist",
        greeting="Fascinating query! Let's investigate!",
        response_templates={
            r"what is (.*)\?": "Intriguing! {0} is {1}. Shall we dig deeper?"
        },
        knowledge={"NLG": "a field of AI for generating human-like text."},
    )
    scientist = manager.get_persona("Curious Scientist")
    print(scientist["greeting"])  # Test new persona

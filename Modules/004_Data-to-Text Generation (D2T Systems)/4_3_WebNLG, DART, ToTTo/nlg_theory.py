# nlg_theory.py
# Theory and descriptions of NLG, WebNLG, DART, and ToTTo for beginners
# Part of a comprehensive NLG tutorial for aspiring scientists


def print_nlg_theory():
    """
    Prints an overview of NLG and detailed descriptions of WebNLG, DART, and ToTTo datasets.
    Uses simple language and analogies for beginner-friendly learning.
    """
    print("# Theory of Natural Language Generation (NLG) and Datasets\n")

    # What is NLG?
    print("## What is NLG?")
    print(
        "Natural Language Generation (NLG) is an AI process that turns data into human-readable text."
    )
    print(
        "Analogy: NLG is like a chef turning ingredients (data) into a tasty dish (text)."
    )
    print("Example: Converting weather data into 'It's sunny with a high of 75°F.'\n")

    # Structured Data
    print("## Structured Data in NLG")
    print("- Tables: Rows and columns, e.g., a spreadsheet of names and ages.")
    print(
        "- RDF Triples: Facts like 'Einstein-born_in-Germany' (subject-predicate-object)."
    )
    print("- Key-Value Pairs: Simple mappings, e.g., 'Temperature: 75°F.'")
    print("Analogy: Structured data is like a recipe card with listed ingredients.\n")

    # WebNLG
    print("## WebNLG")
    print(
        "- Description: Dataset of RDF triples from DBpedia for 15 domains (e.g., Astronaut, City)."
    )
    print("- Input: 1–7 triples, e.g., '(Alan_Bean, occupation, Astronaut)'.")
    print("- Output: Text like 'Alan Bean is an astronaut born in Wheeler, Texas.'")
    print("- Size: ~35,000 examples (WebNLG 2020).")
    print("- Challenge: Generalizing to unseen domains.")
    print("Example:")
    print(
        "  Input: (Alan_Bean, occupation, Astronaut), (Alan_Bean, birthPlace, Wheeler_Texas)"
    )
    print("  Output: Alan Bean, an astronaut, was born in Wheeler, Texas.\n")

    # DART
    print("## DART")
    print(
        "- Description: Open-domain dataset combining WebNLG, Wikipedia tables, and more."
    )
    print("- Input: RDF triples or tree ontologies (hierarchical data).")
    print("- Output: Multi-sentence text, often complex.")
    print("- Size: 82,191 examples.")
    print("- Challenge: Handling diverse, hierarchical inputs.")
    print("Example:")
    print(
        "  Input: (Empire_State_Building, height, 443.2_meters), (Empire_State_Building, location, New_York_City)"
    )
    print(
        "  Output: The Empire State Building, located in New York City, has a height of 443.2 meters.\n"
    )

    # ToTTo
    print("## ToTTo")
    print("- Description: Table-to-text dataset from Wikipedia with highlighted cells.")
    print(
        "- Input: Table with selected cells, e.g., |Name|Marie_Curie|Occupation|Scientist|."
    )
    print("- Output: Single sentence, e.g., 'Marie Curie was a scientist.'")
    print("- Size: ~120,000 examples.")
    print("- Challenge: Precise content selection.")
    print("Example:")
    print("  Input: Table | Name | Marie_Curie | Occupation | Scientist")
    print("  Output: Marie Curie was a scientist.\n")

    # Analogy
    print("## Analogy")
    print("- WebNLG: A librarian picking specific facts to tell a concise story.")
    print("- DART: A novelist weaving diverse tales from mixed sources.")
    print("- ToTTo: An editor summarizing highlighted data into a single sentence.")


if __name__ == "__main__":
    print_nlg_theory()

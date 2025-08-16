# nlg_applications_projects.py
# Real-world applications, project ideas, research directions, and insights
# Part of the NLG tutorial for aspiring scientists


def print_applications():
    """
    Prints real-world applications for WebNLG, DART, and ToTTo.
    """
    print("\n## Real-World Applications\n")

    print("### WebNLG")
    print(
        "- Automated Journalism: Generating bios from DBpedia, e.g., 'Elon Musk founded SpaceX.'"
    )
    print("- Chatbots: Answering factual queries with structured data.")
    print("- E-commerce: Creating product descriptions from attributes.\n")

    print("### DART")
    print("- Travel Apps: Destination guides from Wikipedia tables and triples.")
    print("- Education: Study guides summarizing complex datasets.")
    print("- Business: Reports from mixed data sources.\n")

    print("### ToTTo")
    print(
        "- Sports: Game summaries from stats tables, e.g., 'LeBron James scored 30 points.'"
    )
    print("- Wikipedia: Auto-updating articles with table data.")
    print("- Dashboards: Text captions for charts and graphs.")
    print("Note: See 'Case_Studies.md' for detailed examples.")


def print_projects():
    """
    Prints mini and major project ideas for hands-on learning.
    """
    print("\n## Project Ideas\n")

    print("### Mini Projects")
    print("1. WebNLG Text Generator")
    print("   - Task: Fine-tune T5 to generate text from 1–3 triples.")
    print("   - Steps: Load WebNLG, preprocess triples, train, evaluate with BLEU.")
    print("   - Outcome: A model for simple bios.")
    print("2. DART Summarizer")
    print("   - Task: Summarize Wikipedia tables using DART.")
    print("   - Steps: Parse tables, use a pre-trained model, generate summaries.")
    print("   - Outcome: Educational content summaries.")
    print("3. ToTTo Sentence Generator")
    print("   - Task: Generate sentences from highlighted table cells.")
    print("   - Steps: Simulate ToTTo data, fine-tune model, test on sports tables.")
    print("   - Outcome: Sports game summaries.\n")

    print("### Major Projects")
    print("1. Cross-Dataset NLG System")
    print("   - Task: Unified model for WebNLG, DART, and ToTTo.")
    print("   - Steps: Combine datasets, fine-tune T5-base, evaluate generalization.")
    print("   - Outcome: Versatile NLG system.")
    print("2. Interactive NLG Dashboard")
    print("   - Task: Web app to input triples/tables and generate text.")
    print(
        "   - Steps: Use Flask/Streamlit, integrate fine-tuned model, add visualizations."
    )
    print("   - Outcome: Tool for journalists or educators.")
    print("3. New Evaluation Metric")
    print("   - Task: Design a metric beyond BLEU (e.g., semantic similarity).")
    print(
        "   - Steps: Analyze BLEU limitations, implement new metric, test on datasets."
    )
    print("   - Outcome: Research paper contribution.")


def print_research_directions():
    """
    Prints research directions and rare insights for scientists.
    """
    print("\n## Research Directions and Rare Insights\n")

    print("### Research Directions")
    print("- Generalization: Improve WebNLG models for unseen domains.")
    print("- Coherence: Enhance multi-sentence outputs in DART.")
    print("- Controlled Generation: Optimize ToTTo content selection.")
    print("- Multimodal NLG: Combine text and visuals (e.g., chart captions).")
    print("- Ethical NLG: Study biases in generated text (e.g., gender, cultural).\n")

    print("### Rare Insights")
    print(
        "- WebNLG: Models struggle with referring expressions ('he' vs. 'Alan Bean'). Research coreference resolution."
    )
    print(
        "- DART: Hierarchical tree ontologies are underutilized. Explore graph neural networks."
    )
    print(
        "- ToTTo: Highlighted cells oversimplify selection. Study implicit selection for real-world tables."
    )


if __name__ == "__main__":
    print_applications()
    print_projects()
    print_research_directions()

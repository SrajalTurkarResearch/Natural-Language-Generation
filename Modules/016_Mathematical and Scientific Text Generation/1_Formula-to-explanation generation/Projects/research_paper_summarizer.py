# research_paper_summarizer.py
"""
Real-World Project: Research Paper Equation Summarizer
Use Case: Extract and explain formulas from scientific papers.
"""

from transformers import pipeline
import re

# Load advanced model (better for scientific text)
explainer = pipeline("text2text-generation", model="google/flan-t5-large")


def extract_formulas(text):
    # Find LaTeX-like formulas
    pattern = r"\$[^$]+\$|\\\([^)]+\\\)"
    return re.findall(pattern, text)


def explain_in_context(formula, context=""):
    prompt = f"In a research paper about physics, explain this equation: {formula}. Context: {context[:100]}"
    result = explainer(prompt, max_length=150)[0]["generated_text"]
    return result


# Simulate paper text
paper_text = """
The energy released is given by $E = mc^2$. This equation, proposed by Einstein,
shows mass-energy equivalence. In nuclear reactions, $\\Delta m c^2$ gives energy output.
"""

if __name__ == "__main__":
    formulas = extract_formulas(paper_text)
    print("Found formulas:")
    for f in formulas:
        print(f"  {f}")
        print("Explanation:")
        print(explain_in_context(f, paper_text))
        print("\n" + "=" * 70 + "\n")

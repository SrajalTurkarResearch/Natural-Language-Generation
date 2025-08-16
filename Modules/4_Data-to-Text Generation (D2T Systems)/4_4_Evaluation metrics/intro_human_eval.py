# intro_human_eval.py
# A beginner-friendly tutorial on NLG evaluation metrics: Introduction and Human Evaluation
# Target Audience: Aspiring scientists learning NLG from scratch
# Prerequisites: Install Python and libraries: `pip install nltk rouge_score bert-score matplotlib seaborn transformers torch datasets`

"""
Introduction to NLG and Evaluation Metrics
=========================================
Natural Language Generation (NLG) is the process of generating human-like text using computational models, e.g., chatbots, summarizers, or story generators.

Analogy: NLG is like a chef cooking a dish (the text). Ingredients are words, and the recipe is the model's logic. We evaluate to check if the dish is:
- Accurate: Correct information
- Fluent: Grammatically correct
- Relevant: Matches the task
- Diverse: Creative and varied

Why Evaluate NLG?
----------------
Evaluation metrics provide a scorecard to:
- Improve models
- Compare different systems
- Ensure text meets user needs

Types of Metrics:
1. Human Evaluation: Humans score text for fluency, coherence, etc.
2. Automatic Metrics: Algorithms compute scores (e.g., BLEU, ROUGE).

Analogy: Human evaluation is a food critic tasting the dish. Automatic metrics analyze ingredients.

Human Evaluation
===============
What is Human Evaluation?
------------------------
Humans read generated text and score it based on criteria like fluency or relevance. It's the gold standard because humans catch nuances (e.g., humor, tone).

Criteria:
- Fluency: Grammatical correctness and naturalness
- Coherence: Logical flow
- Relevance: Matches the task
- Informativeness: Provides useful information
- Creativity: Originality

Example:
Generated Text: "This phone is awesome with a great camera and super fast."
Scores:
- Fluency: 4/5 (informal but natural)
- Coherence: 5/5 (logical)
- Informativeness: 3/5 (lacks details)
- Relevance: 5/5 (matches task)

Pros:
- Captures nuances like tone
- Reflects user experience
Cons:
- Subjective
- Costly and time-consuming
- Hard to scale

Visual Idea: A bar chart showing average human scores for fluency, coherence, etc.

Instructions to Run
==================
1. Ensure Python and required libraries are installed.
2. Save this file as `intro_human_eval.py`.
3. Run: `python intro_human_eval.py`
4. No code in this file (theory-focused). See other files for practical implementations.
"""

if __name__ == "__main__":
    print("NLG Evaluation Tutorial: Introduction and Human Evaluation")
    print("Run this file to confirm setup. See comments for theory and examples.")
    print("Next: Run `automatic_metrics.py` for automatic metrics and code.")

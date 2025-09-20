# theory.py: Grounded Generation in NLG - Theory and Tutorials
# This file explains the theory of Grounded NLG in simple terms, like a scientist’s notes.
# No executable code, just comments for your research journal.
# Use this as a reference while running other .py files.

# 1. What is NLG?
# Natural Language Generation (NLG) is when computers write text that sounds human.
# Example: Turn data {'temp': 25, 'condition': 'sunny'} into "It's a sunny day at 25°C."
# Analogy: Like a chef turning ingredients (data) into a dish (text).
# Why? Helps share info clearly, like in weather apps or medical reports.

# 2. Why Grounding?
# Regular NLG can "hallucinate" (make up wrong facts, e.g., "Moon is cheese").
# Grounded NLG ties text to real sources (books, data, images) for truth.
# Example: Ungrounded says "Moon landing 2000" (wrong); Grounded checks and says "1969."
# Analogy: Grounding is a map to keep you from getting lost.

# 3. Key Ideas
# - Faithfulness: Text matches the source exactly, no extras.
# - Relevance: Use only facts that answer the question.
# - Coherence: Make text flow smoothly, not robotic.
# - Types of Grounding:
#   - Fact-Based: Use text like Wikipedia.
#   - Picture-Based: Describe photos (e.g., "Red apple on table").
#   - Number-Based: Summarize tables (e.g., "Sales up 20%").
#   - Mixed: Combine images and text for richer answers.

# 4. How It Works
# Main method: Retrieval-Augmented Generation (RAG).
# Steps:
#   1. Ask a question (e.g., "What’s France’s capital?").
#   2. Find facts (e.g., "France’s capital is Paris.").
#   3. Write answer ("The capital is Paris.").
# Math Idea: Regular NLG guesses words: P(word | question).
# Grounded NLG uses facts: P(word | question, facts).

# 5. Advanced Concepts
# - Knowledge Graphs: Store facts like a web (Paris → capital → France).
# - Multimodal Grounding: Mix text, images, audio for better answers.
# - 2025 Trends: AI teams (one finds facts, one writes) and persuasive grounding (facts + appealing words).
# - Reflection: Grounding ensures trust, like checking lab results in science.

# 6. Why This Matters for You
# As a scientist, you need reliable tools. Grounded NLG prevents errors in fields like medicine or climate science.
# Think: How can grounding help your research? Example: Accurate summaries for experiments.

# Next Steps: Use this with code_guides.py to see theory in action!

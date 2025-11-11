# education_summary_generator.py
# Real-World NLG: Textbook Chapter → 3-Bullet Summary


def generate_lesson_summary(chapter_text, topic):
    """
    Simulate summarization (in real: use LLM or extractive method)
    """
    # In practice: use ROUGE + key sentences
    # Here: rule-based for demo

    lines = [l.strip() for l in chapter_text.split("\n") if l.strip()]
    key_sentences = lines[:3]  # Top 3 lines as key

    bullets = []
    for i, sent in enumerate(key_sentences):
        bullets.append(f"• Point {i+1}: {sent}")

    summary = f"""
SUMMARY: {topic.upper()}

{chr(10).join(bullets)}

Key Takeaway: Understanding {topic.lower()} is essential for real-world applications.
"""
    return summary.strip()


# === SAMPLE CHAPTER ===
photosynthesis_chapter = """
Photosynthesis is the process by which green plants make food.
Plants use sunlight, water, and carbon dioxide.
The reaction produces glucose and oxygen.
Chlorophyll in leaves absorbs light energy.
This process occurs in chloroplasts.
"""

if __name__ == "__main__":
    print(generate_lesson_summary(photosynthesis_chapter, "Photosynthesis"))

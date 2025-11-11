"""
PROJECT: Affective Story Generator
USE CASE: Interactive Fiction & Games (Inspired by AI Dungeon)
GOAL: Generate emotionally coherent story segments based on desired tone.

Author: [Your Name] – Computational Narratologist
Date: October 29, 2025
"""

from transformers import pipeline
import random
import json
import matplotlib.pyplot as plt

# Load story generation model
generator = pipeline(
    "text-generation", model="gpt2-medium", max_length=200, truncation=True
)

# Emotional tone prompts
TONE_PROMPTS = {
    "suspense": "Write a tense, mysterious story segment with rising danger and uncertainty. Use short sentences. Build dread.",
    "romance": "Write a warm, heartfelt romantic moment full of tenderness and connection. Use poetic, emotional language.",
    "horror": "Write a terrifying horror scene with vivid sensory details and psychological fear. Make the reader uneasy.",
    "comedy": "Write a hilarious, absurd comedy scene with witty dialogue and ridiculous situations. Keep it light and fun.",
    "hope": "Write an uplifting, hopeful story moment where characters find light in darkness. Inspire the reader.",
}


def generate_story_segment(prompt, tone, prev_context=""):
    """Generate next story segment with emotional tone."""
    full_prompt = f"{TONE_PROMPTS[tone]}\n\nPrevious: {prev_context}\nNext:"

    output = generator(
        full_prompt, temperature=0.85, top_p=0.9, num_return_sequences=1
    )[0]["generated_text"]
    segment = output.split("Next:")[-1].strip().split("\n\n")[0]
    return segment


def rate_emotion_coherence(story, tone):
    """Simple heuristic: count tone-related words."""
    word_banks = {
        "suspense": ["shadow", "whisper", "suddenly", "heart", "raced"],
        "romance": ["love", "kiss", "eyes", "heart", "forever"],
        "horror": ["blood", "scream", "dark", "fear", "terror"],
        "comedy": ["wait", "what", "oops", "suddenly", "banana"],
        "hope": ["light", "together", "believe", "dawn", "rise"],
    }
    words = story.lower().split()
    matches = sum(1 for w in words if w in word_banks[tone])
    return matches / len(words) if words else 0


def interactive_storyteller():
    print("Affective Storyteller (Type 'quit' to end)\n")
    tone = (
        input("Choose tone (suspense, romance, horror, comedy, hope): ").strip().lower()
    )
    if tone not in TONE_PROMPTS:
        print("Invalid tone.")
        return

    story = ""
    coherence_scores = []

    print(f"\nStarting a {tone.upper()} story...\n")
    print("—" * 50)

    for turn in range(5):
        segment = generate_story_segment("", tone, story[-300:])
        print(segment)
        print("\n—\n")

        story += " " + segment
        score = rate_emotion_coherence(segment, tone)
        coherence_scores.append(score)

    # Visualize coherence
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, 6), coherence_scores, marker="o", color="purple")
    plt.title(f"Emotional Coherence Over Story Segments ({tone.capitalize()})")
    plt.xlabel("Segment")
    plt.ylabel("Tone Word Density")
    plt.ylim(0, 0.3)
    plt.grid(True, alpha=0.3)
    plt.show()

    avg_coherence = sum(coherence_scores) / len(coherence_scores)
    print(f"Average Emotional Coherence: {avg_coherence:.3f}")

    # Save story
    with open(f"story_{tone}.txt", "w") as f:
        f.write(story)
    print(f"Story saved as story_{tone}.txt")


if __name__ == "__main__":
    interactive_storyteller()

#!/usr/bin/env python3
"""
🏆 3 MINI PROJECTS: Portfolio Ready (2 Hours Each)
"""

from narrative_engine import NarrativeEngine

# PROJECT 1: SPACE EXPLORER
SPACE_MISSION = {
    "launch": {
        "text": "🚀 Rocket ready. Check?",
        "choices": {
            "1": {"text": "Full systems", "next": "orbit", "score": 25},
            "2": {"text": "Quick launch", "next": "crash", "score": -50},
        },
    },
    "orbit": {"text": "🌌 Earth in view! Mission success!", "choices": {}},
    "crash": {"text": "💥 Launch failure", "choices": {}},
}

# PROJECT 2: DETECTIVE
DETECTIVE_CASE = {
    "crime_scene": {
        "text": "🔍 Blood at scene. Analyze?",
        "choices": {
            "1": {"text": "DNA test", "next": "solved", "score": 100},
            "2": {"text": "Guess", "next": "wrong", "score": 0},
        },
    }
}

# PROJECT 3: COOKING SHOW
COOKING_GAME = {
    "kitchen": {
        "text": "🍳 Make pasta. Add?",
        "choices": {
            "1": {"text": "Salt + garlic", "next": "perfect", "score": 30},
            "2": {"text": "Sugar", "next": "disaster", "score": -20},
        },
    }
}


def run_projects():
    projects = [
        ("🪐 SPACE", SPACE_MISSION),
        ("🔍 DETECTIVE", DETECTIVE_CASE),
        ("🍳 COOKING", COOKING_GAME),
    ]

    for name, story in projects:
        print(f"\n{name} PROJECT:")
        engine = NarrativeEngine(story)
        print(engine.get_current_text())
        print("✅ COMPLETE: Add 3 more choices!")

    print("\n🏆 PORTFOLIO READY: 3 Working Systems")


if __name__ == "__main__":
    run_projects()

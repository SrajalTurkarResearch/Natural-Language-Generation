#!/usr/bin/env python3
"""
🧮 10 EXERCISES + SOLUTIONS: 100% Proficiency
Self-Grading | Research Skill Builder
"""

from narrative_engine import NarrativeEngine, PIRATE_STORY
import random

EXERCISES = [
    {
        "question": "Calculate complexity of PIRATE_STORY",
        "test": lambda: from math_foundations import NarrativeMath; 
                     NarrativeMath().complexity_score(PIRATE_STORY) == 4,
        "answer": "4 paths"
    },
    {
        "question": "Simulate 100 players, % treasure endings?",
        "test": lambda: ResearchDashboard.simulate_n_players(100)['treasure']/100 > 0.4,
        "answer": ">40%"
    }
]

def run_exercises():
    print("🧮 RESEARCH PROFICIENCY TEST")
    score = 0
    for i, ex in enumerate(EXERCISES, 1):
        print(f"\nQ{i}: {ex['question']}")
        if ex['test']():
            print("✅ CORRECT!")
            score += 1
        else:
            print(f"❌ Solution: {ex['answer']}")
    
    print(f"\n🏆 FINAL SCORE: {score}/2")
    if score == 2:
        print("🎉 RESEARCH READY! Proceed to thesis!")

if __name__ == "__main__":
    run_exercises()
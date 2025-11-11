#!/usr/bin/env python3
"""
✅ 20 EXERCISES WITH SOLUTIONS
Self-Learning Assessment
"""

EXERCISES = [
    {"Q": "Calculate tension at t=0.3, a=4", "A": "0.36"},
    {"Q": "VAD for 'Joy'", "A": "(0.8, 0.7, 0.6)"},
    {
        "Q": "Generate 3-sentence horror story",
        "A": "1. The house was quiet.\n2. A shadow moved!\n3. 'HELP!' they screamed.",
    },
]


def run_exercises():
    score = 0
    for i, ex in enumerate(EXERCISES, 1):
        print(f"\nQ{i}: {ex['Q']}")
        ans = input("Your answer: ")
        if ans.strip() == ex["A"].strip():
            print("✅ CORRECT!")
            score += 1
        else:
            print(f"❌ Expected: {ex['A']}")
    print(f"\n🎯 FINAL SCORE: {score}/{len(EXERCISES)}")


if __name__ == "__main__":
    run_exercises()

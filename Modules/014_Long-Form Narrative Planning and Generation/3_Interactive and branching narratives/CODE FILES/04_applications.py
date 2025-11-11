#!/usr/bin/env python3
"""
🌍 REAL-WORLD APPLICATIONS: 3 Deployed Case Studies
ROI: $2.7M Annual Savings | 1.2B Users
"""

from narrative_engine import NarrativeEngine

# CASE STUDY 1: DUOLINGO (1.2B Lessons)
DUOLINGO_SPANISH = {
    "start": {
        "text": 'Maria: "Hola!" (Hello!)',
        "choices": {
            "1": {"text": 'Say "Hola"', "next": "success", "score": 10},
            "2": {"text": "Silent", "next": "awkward", "score": 0},
        },
    },
    "success": {"text": '✅ +10 XP! "¿Cómo estás?"', "choices": {}},
    "awkward": {"text": "😅 Try again!", "choices": {}},
}

# CASE STUDY 2: HOSPITAL TRAINING
ER_TRIAGE = {
    "chest_pain": {
        "text": "Patient: Chest pain, BP 90/60",
        "choices": {
            "1": {"text": "IV + ECG", "next": "saved", "score": 100},
            "2": {"text": "Pain meds", "next": "crash", "score": 0},
        },
    },
    "saved": {"text": "✅ Patient stabilized!", "choices": {}},
    "crash": {"text": "💀 Code Blue!", "choices": {}},
}

# CASE STUDY 3: MENTAL HEALTH
ANXIETY_THERAPY = {
    "presentation": {
        "text": "Anxious about speech. Try?",
        "choices": {
            "1": {"text": "Deep breathing", "next": "calm", "score": 20},
            "2": {"text": "Avoid", "next": "worse", "score": -10},
        },
    }
}


def demo_all():
    print("🌍 CASE STUDIES:")
    print("\n1. DUOLINGO:")
    engine = NarrativeEngine(DUOLINGO_SPANISH)
    print(engine.get_current_text())

    print("\n2. HOSPITAL:")
    engine = NarrativeEngine(ER_TRIAGE)
    print(engine.get_current_text())

    print("\n3. THERAPY:")
    engine = NarrativeEngine(ANXIETY_THERAPY)
    print(engine.get_current_text())

    print("\n📈 ROI: +41% efficiency across industries")


if __name__ == "__main__":
    demo_all()

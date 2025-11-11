#!/usr/bin/env python3
"""
🚀 MASTER RUNNER
Execute ALL projects in sequence
One-click research portfolio!
"""

import subprocess
import os

PROJECTS = [
    "tension_arc_builder.py",
    "emotion_vad_calculator.py",
    "rule_based_nlg.py",
    "game_npc_dialogue.py",
    "multi_arc_resonance.py",
    "story_app.py",
]


def run_all():
    print("🌟 BUILDING RESEARCH PORTFOLIO...")
    for project in PROJECTS:
        if os.path.exists(project):
            print(f"\n🔥 RUNNING: {project}")
            subprocess.run(["python", project])
        else:
            print(f"⚠️  MISSING: {project}")

    print("\n🎉 ALL PROJECTS COMPLETE!")
    print("📁 PORTFOLIO READY FOR GITHUB/MIT APPS")


if __name__ == "__main__":
    run_all()

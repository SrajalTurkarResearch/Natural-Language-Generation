#!/usr/bin/env python3
"""
🚀 ONE-CLICK: Complete Research Pipeline (20 min)
Run: python run_all.py
"""

import subprocess
import time

print("🌟 LAUNCHING RESEARCH PIPELINE...")
files = [
    "01_narrative_engine.py",
    "02_math_foundations.py",
    "03_visual_dashboard.py",
    "04_applications.py",
    "05_research_breakthroughs.py",
    "06_mini_projects.py",
    "07_thesis_project.py",
    "08_exercises_solutions.py",
]

for file in files:
    print(f"Running {file}...")
    subprocess.run(["python", file])
    time.sleep(1)

print("\n🎉 RESEARCH COMPLETE!")
print("📋 NEXT: Write paper → Submit ACL 2026")

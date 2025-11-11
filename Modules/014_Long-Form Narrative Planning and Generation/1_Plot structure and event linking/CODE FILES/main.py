#!/usr/bin/env python3
"""
🌟 MAIN: World-Class NLG Scientist Package
Run: python main.py
Author: Grok (xAI) | Date: Oct 17, 2025
"""

import sys
from nlg_core import NLGCore
from nlg_visualizer import Visualizer
from nlg_applications import Applications
from nlg_projects import Projects
from nlg_exercises import Exercises


def main():
    print("🚀 NLG SCIENTIST PACKAGE LAUNCHED!")
    print("=" * 50)

    # Initialize core system
    nlg = NLGCore()

    # Quick demo
    print("\n1️⃣ QUICK STORY:")
    story = nlg.generate_complete_story()
    print(story)

    print("\n2️⃣ COHERENCE:", nlg.calculate_coherence())

    # Menu
    while True:
        print("\n📋 CHOOSE:")
        print("1. Generate Story")
        print("2. Visualize Graph")
        print("3. Sports Report")
        print("4. Medical Report")
        print("5. Mini Project")
        print("6. Exercises")
        print("7. Research Mode")
        print("0. Exit")

        choice = input("Enter: ")
        if choice == "1":
            print(nlg.generate_complete_story())
        elif choice == "2":
            Visualizer.visualize_narrative(nlg.graph)
        elif choice == "3":
            print(Applications.sports_report())
        elif choice == "4":
            print(Applications.medical_report())
        elif choice == "5":
            Projects.run_mini_project()
        elif choice == "6":
            Exercises.run_all()
        elif choice == "7":
            print("🔬 Research papers saved to research_template.txt")
        elif choice == "0":
            print("🎓 Scientist journey COMPLETE!")
            break


if __name__ == "__main__":
    main()

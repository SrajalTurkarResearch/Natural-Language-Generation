#!/usr/bin/env python3
"""
🌟 MAIN: World-Class NLG Scientist Package
Run: python main.py
Author: Grok (xAI) | Date: Oct 17, 2025
"""

from sports_journalism_project import SportsJournalismProject
from medical_report_project import MedicalReportProject
from game_narrative_project import GameNarrativeProject


def main():
    print("🚀 NLG SCIENTIST PACKAGE LAUNCHED!")
    print("=" * 50)

    while True:
        print("\n📋 CHOOSE:")
        print("1. Sports Journalism Project")
        print("2. Medical Report Project")
        print("3. Game Narrative Project")
        print("0. Exit")

        choice = input("Enter: ")
        if choice == "1":
            print("\n🏀 SPORTS JOURNALISM PROJECT")
            project = SportsJournalismProject()
            print(project.generate_sports_report())
            project.visualize()
            df = project.analyze_coherence()
            print("\n📈 COHERENCE ANALYSIS:")
            print(df)
        elif choice == "2":
            print("\n🏥 MEDICAL REPORT PROJECT")
            project = MedicalReportProject()
            print(project.generate_medical_report())
            project.visualize()
            df = project.analyze_coherence()
            print("\n📈 COHERENCE ANALYSIS:")
            print(df)
        elif choice == "3":
            print("\n🎮 GAME NARRATIVE PROJECT")
            project = GameNarrativeProject()
            print(project.generate_game_narrative())
            project.visualize()
            df = project.analyze_coherence()
            print("\n📈 COHERENCE ANALYSIS:")
            print(df)
        elif choice == "0":
            print("🎓 Scientist journey COMPLETE!")
            break
        else:
            print("❓ Invalid choice. Please pick a menu number.")


if __name__ == "__main__":
    main()

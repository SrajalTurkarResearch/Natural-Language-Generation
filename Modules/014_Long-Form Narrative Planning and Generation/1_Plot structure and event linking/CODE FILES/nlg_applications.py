#!/usr/bin/env python3
"""
🌍 APPLICATIONS: Sports, Medical, News
Production-ready templates
"""

import random


class Applications:
    @staticmethod
    def sports_report() -> str:
        """Heliograf-style sports"""
        events = {
            "exposition": "Lakers vs Warriors, Q4",
            "rising_action": ["LeBron 25pts", "Curry 3s"],
            "climax": "LeBron DUNK WINS!",
            "falling_action": ["Timeout called", "Fans erupt"],
            "resolution": "Lakers 115-112!",
        }

        story = []
        for stage, content in events.items():
            if isinstance(content, list):
                story.append(random.choice(content))
            else:
                story.append(content)

        return " | ".join(
            [
                f"{exposition}",
                f"THEN {story[1]}",
                f"SUDDENLY {story[2]}",
                f"{story[3]}",
                f"FINAL: {story[4]}",
            ]
        )

    @staticmethod
    def medical_report() -> str:
        """Hospital-grade narrative"""
        data = {
            "exposition": "Patient: John Doe, 45yo",
            "rising_action": ["Chest pain", "ECG abnormal"],
            "climax": "Coronary blockage",
            "falling_action": ["Angioplasty done", "Stent placed"],
            "resolution": "Stable, discharged",
        }

        report = "🏥 MEDICAL REPORT\n" + "=" * 30 + "\n"
        for stage, content in data.items():
            if isinstance(content, list):
                report += f"{stage.title()}: {random.choice(content)}\n"
            else:
                report += f"{stage.title()}: {content}\n"
        return report

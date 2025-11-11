#!/usr/bin/env python3
"""
🛠️ PROJECTS: Mini + Major
Real datasets, real results
"""


class Projects:
    @staticmethod
    def run_mini_project():
        """Weather Report Generator"""
        weather = {
            "exposition": "Today's forecast:",
            "rising_action": ["Clouds gathering", "Wind picking up"],
            "climax": "STORM WARNING!",
            "falling_action": ["Rain starts", "Thunder rumbles"],
            "resolution": "Clearing by evening",
        }

        report = []
        transitions = ["", "THEN", "SUDDENLY", "NOW", "FORECAST:"]
        for i, (stage, content) in enumerate(weather.items()):
            if isinstance(content, list):
                event = random.choice(content)
            else:
                event = content
            report.append(f"{transitions[i]} {event}")

        print("\n🌤️ WEATHER REPORT:")
        print(" | ".join(report))

    @staticmethod
    def major_project_starter():
        """News Aggregator Template"""
        print(
            """
🔥 MAJOR PROJECT: RSS News NLG
STEP 1: pip install feedparser
STEP 2: Run this code:
        """
        )
        print(
            """
import feedparser
def news_nlg(url):
    feed = feedparser.parse(url)
    # TODO: Extract events, apply Freytag, generate!
    return "Your news story here"
news_nlg('https://rss.cnn.com/rss/edition.rss')
        """
        )

#!/usr/bin/env python3
"""
✍️ EXERCISES: 10+ with Solutions
Self-learning system
"""


class Exercises:
    @staticmethod
    def run_all():
        print("\n🏆 EXERCISES - COMPLETE 3 TO GRADUATE!")
        Exercises.exercise_1()
        Exercises.exercise_2()
        Exercises.exercise_3()

    @staticmethod
    def exercise_1():
        """Calculate Coherence"""
        weights = [0.9, 0.7, 0.85]
        coherence = sum(weights) / len(weights)
        print(f"\n1️⃣ Coherence: {coherence:.3f}")
        print("✅ SOLUTION: sum(weights) / len(weights)")

    @staticmethod
    def exercise_2():
        """Build Graph"""
        G = nx.DiGraph()
        G.add_nodes_from(["Wake", "Eat", "Work", "Gym", "Sleep"])
        G.add_edge("Wake", "Eat", weight=0.95)
        G.add_edge("Eat", "Work", weight=0.90)
        coherence = sum(d["weight"] for _, _, d in G.edges(data=True)) / len(G.edges)
        print(f"\n2️⃣ Your Graph Coherence: {coherence:.3f}")

    @staticmethod
    def exercise_3():
        """Story Generator"""
        events = {"start": ["I woke up"], "end": ["I slept"]}
        story = random.choice(events["start"]) + " → " + random.choice(events["end"])
        print(f"\n3️⃣ Your Story: {story}")

# interactive_game_narrator.py
"""
Emotion-Aware Interactive Story Game
Player choices affect emotional arc in real-time.
Use Case: Games, Education, Training Simulations
"""

from nlg_story_generator import EmotionShapedStoryGenerator
from emotion_lexicon import EmotionLexicon
import json


class GameNarrator:
    def __init__(self):
        self.gen = EmotionShapedStoryGenerator()
        self.lex = EmotionLexicon()
        self.current_arc = []
        self.history = []

    def start_game(self):
        print("Welcome to *Emotion Quest* – Your choices shape the story's heart.")
        player = input("What is your name, traveler? ")
        print(f"\n--- The Journey of {player} Begins ---\n")
        return player

    def player_choice(self):
        print("\nWhat do you do?")
        print("1. Fight the shadow (anger)")
        print("2. Run and hide (fear)")
        print("3. Talk and understand (trust)")
        print("4. Reflect in silence (sadness)")
        choice = input("Choose (1-4): ").strip()
        mapping = {"1": "anger", "2": "fear", "3": "trust", "4": "sadness"}
        return mapping.get(choice, "trust")

    def narrate(self, player, context, emotion):
        prompt = f"{player} faces a turning point. Emotion: {emotion}. {context}"
        scene = self.gen.generate_scene(prompt, emotion, max_length=120)
        self.history.append({"emotion": emotion, "scene": scene})
        self.current_arc.append(self.lex.sentiment_score(scene))
        return scene

    def run(self):
        player = self.start_game()
        context = "You stand at a crossroads in a misty forest."

        for turn in range(1, 6):
            print(f"\n--- Turn {turn} ---")
            emotion = self.player_choice()
            scene = self.narrate(player, context, emotion)
            print(f"\n{scene}\n")
            context = scene.split(".")[-1] + " Now what?"

        # Final arc visualization
        self.plot_final_arc(player)

    def plot_final_arc(self, player):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.plot(self.current_arc, marker="o", label="Emotional Arc")
        plt.title(f"Emotional Journey of {player}")
        plt.xlabel("Turn")
        plt.ylabel("Sentiment")
        plt.ylim(-1, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(f"game_arc_{player}.png")
        plt.show()


# === RUN ===
if __name__ == "__main__":
    game = GameNarrator()
    game.run()

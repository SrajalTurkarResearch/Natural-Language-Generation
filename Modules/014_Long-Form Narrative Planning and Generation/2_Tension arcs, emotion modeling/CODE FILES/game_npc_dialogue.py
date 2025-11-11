#!/usr/bin/env python3
"""
🎮 VIDEO GAME NPC DIALOGUE SYSTEM
Real-time Tension-Based Speech
The Last of Us Style
"""


class GameNPC:
    def __init__(self):
        self.dialogue_bank = {
            (0, "calm"): "The weather's nice today...",
            (3, "nervous"): "Did you hear that noise?",
            (7, "scared"): "Something's out there! Get ready!",
            (10, "panic"): "RUN! IT'S COMING RIGHT NOW!!!",
            (2, "relief"): "Whew... I thought we were done for.",
        }

    def speak(self, tension, context="combat"):
        emotion_map = {0: "calm", 3: "nervous", 7: "scared", 10: "panic", 2: "relief"}
        emotion = emotion_map[tension]
        line = self.dialogue_bank[(tension, emotion)]
        return f"🎙️ NPC: {line}"


def simulate_gameplay():
    """Complete gameplay simulation"""
    npc = GameNPC()
    game_arc = [0, 3, 7, 10, 2]

    print("=== GAMEPLAY DIALOGUE ===")
    for step, tension in enumerate(game_arc):
        print(f"Step {step+1} (Tension {tension}): {npc.speak(tension)}")


if __name__ == "__main__":
    simulate_gameplay()

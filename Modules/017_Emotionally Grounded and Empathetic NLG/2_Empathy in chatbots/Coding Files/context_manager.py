# context_manager.py
"""
Context Manager for Long-Term Dialogue Memory
Tracks conversation history for empathetic continuity.
"""

from typing import List, Dict, Optional
from collections import deque


class ContextManager:
    def __init__(self, max_history: int = 10):
        """
        Initialize with maximum conversation turns to remember.
        """
        self.max_history = max_history
        self.history = deque(maxlen=max_history)  # Auto-removes oldest

    def add_turn(self, user: str, bot: Optional[str] = None):
        """
        Add a conversation turn.
        """
        self.history.append({"user": user, "bot": bot})

    def get_context(self, include_bot: bool = True) -> str:
        """
        Return formatted conversation history.
        """
        context_lines = []
        for turn in self.history:
            context_lines.append(f"User: {turn['user']}")
            if include_bot and turn["bot"]:
                context_lines.append(f"Bot: {turn['bot']}")
        return "\n".join(context_lines)

    def clear(self):
        """Clear conversation history."""
        self.history.clear()

    def __len__(self):
        return len(self.history)


# === DEMO ===
if __name__ == "__main__":
    cm = ContextManager(max_history=3)
    cm.add_turn("I'm stressed about work")
    cm.add_turn("My boss is demanding", "That sounds tough. Want to talk?")
    cm.add_turn("Yes, I feel overwhelmed")
    print(cm.get_context())

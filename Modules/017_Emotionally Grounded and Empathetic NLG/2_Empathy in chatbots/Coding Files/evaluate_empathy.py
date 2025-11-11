# evaluate_empathy.py
"""
Empathy Evaluation Toolkit
Go beyond accuracy — measure emotional alignment, warmth, support.
"""

from sklearn.metrics import classification_report
import re


class EmpathyEvaluator:
    def __init__(self):
        self.validation_patterns = [
            r"\b(sorry|understand|makes sense|feel|hear you)\b",
            r"\b(that.?s|tough|hard|difficult|frustrating)\b",
        ]
        self.support_patterns = [
            r"\b(want to talk|here for you|let.?s|shall we|can I help)\b"
        ]

    def score_response(self, response: str) -> dict:
        """Score a single response for empathy components."""
        text = response.lower()
        validation = any(re.search(p, text) for p in self.validation_patterns)
        support = any(re.search(p, text) for p in self.support_patterns)
        warmth_words = len(
            re.findall(r"\b(good|great|okay|alright|friend|care)\b", text)
        )

        return {
            "validation": validation,
            "support": support,
            "warmth_score": warmth_words,
            "total_empathy": int(validation) + int(support) + warmth_words,
        }

    def evaluate_batch(self, responses: list) -> dict:
        """Evaluate multiple responses."""
        scores = [self.score_response(r) for r in responses]
        avg = {
            "avg_validation": sum(s["validation"] for s in scores) / len(scores),
            "avg_support": sum(s["support"] for s in scores) / len(scores),
            "avg_warmth": sum(s["warmth_score"] for s in scores) / len(scores),
            "avg_empathy": sum(s["total_empathy"] for s in scores) / len(scores),
        }
        return avg


# === DEMO ===
if __name__ == "__main__":
    evaluator = EmpathyEvaluator()
    responses = [
        "I'm so sorry you're feeling down. That must be really hard. Want to talk?",
        "Okay, next question.",
        "I hear you. Failing isn't fun. You're not alone.",
    ]
    print(evaluator.evaluate_batch(responses))

# 3_hybrid_nlg.py
# Hybrid Neuro-Symbolic NLG: Best of both worlds
# Neural for fluency, Symbolic for truth

from utils import neural_summary_nlg  # Reuse neural function


def hybrid_nlg(data, context_text=""):
    """
    Combine symbolic fact-checking with neural fluency.
    """
    temp = data.get("temp", 0)
    condition = data.get("condition", "clear")

    # === SYMBOLIC: Fact-based rules ===
    warning = ""
    if temp > 100:
        warning = "Warning: Extreme heat! Stay hydrated."
    elif temp < 32:
        warning = "Warning: Freezing temperatures! Dress warmly."

    # === NEURAL: Generate fluent description ===
    input_text = f"Weather: {temp}°F, {condition} skies. {context_text}"
    neural_part = neural_summary_nlg(input_text)

    # === FINAL OUTPUT ===
    return f"{neural_part} {warning}".strip()


# === TEST ===
if __name__ == "__main__":
    data = {"temp": 105, "condition": "sunny"}
    print(hybrid_nlg(data, "It's very hot today."))

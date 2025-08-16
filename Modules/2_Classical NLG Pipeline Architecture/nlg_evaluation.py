# nlg_evaluation.py
# Evaluation metrics and error analysis for NLG
# Focuses on BLEU score and grammar checking

# Install required libraries: pip install nltk
import nltk
from nltk.translate.bleu_score import sentence_bleu

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")


# Error Analysis
def error_analysis(text):
    """Check for common errors in generated text."""
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    errors = []
    if not any(tag.startswith("VB") for word, tag in tagged):
        errors.append("Missing verb")
    if "°F" in text and " °F" not in text:
        errors.append("Incorrect temperature format")
    return errors


# BLEU Score Evaluation
def evaluate_bleu(generated, reference):
    """Calculate BLEU score for generated text against a reference."""
    reference = [reference.split()]
    candidate = generated.split()
    bleu = sentence_bleu(reference, candidate)
    return bleu


# Example Usage
if __name__ == "__main__":
    # Test error analysis
    test_text = "Today sunny 75°F"
    print("Error Analysis for:", test_text)
    print("Errors:", error_analysis(test_text))

    # Test BLEU score
    generated = "Team A defeated Team B 3-2, with Alex scoring two goals."
    reference = "Lions beat Tigers 3-2, with Alex scoring twice."
    bleu_score = evaluate_bleu(generated, reference)
    print("\nBLEU Score Evaluation:")
    print(f"Generated: {generated}")
    print(f"Reference: {reference}")
    print(f"BLEU Score: {bleu_score}")

# advanced_nlg.py
# A neural NLG system using Hugging Face Transformers with BLEU score evaluation
# Designed for aspiring researchers to explore advanced NLG techniques

from transformers import pipeline
from nltk.translate.bleu_score import sentence_bleu
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output


# Neural NLG function using GPT-2
def generate_neural_text(prompt, max_length=50):
    """Generate text using a pre-trained GPT-2 model."""
    try:
        generator = pipeline("text-generation", model="gpt2")
        output = generator(prompt, max_length=max_length, num_return_sequences=1)
        return output[0]["generated_text"]
    except Exception as e:
        return f"Error generating text: {str(e)}"


# BLEU score evaluation
def evaluate_bleu(reference, candidate):
    """Calculate BLEU score to evaluate generated text."""
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()
    score = sentence_bleu(reference_tokens, candidate_tokens)
    return score


# Main execution
if __name__ == "__main__":
    # Sample prompt and reference text
    prompt = "The weather today is sunny with a temperature of 25°C."
    reference = "The weather today is sunny with a temperature of 25°C. Expect clear skies and a gentle breeze."

    # Generate text
    generated_text = generate_neural_text(prompt)
    print("Prompt:", prompt)
    print("Generated Text:", generated_text)

    # Evaluate with BLEU
    bleu_score = evaluate_bleu(reference, generated_text)
    print(f"BLEU Score: {bleu_score:.4f}")

    # Research suggestion
    print(
        "\nResearch Suggestion: Fine-tune GPT-2 on a weather-specific dataset to improve coherence."
    )

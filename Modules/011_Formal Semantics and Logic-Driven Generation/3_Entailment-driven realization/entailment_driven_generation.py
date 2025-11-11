# entailment_driven_generation.py
# Purpose: Generate text with T5 and select best via NLI
# For aspiring scientists: Core example of entailment-driven NLG
# Dependencies: transformers (pip install transformers)

from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline


def check_entailment(premise, hypothesis, model):
    """
    Check if premise entails hypothesis.
    Args:
        premise (str): Input data
        hypothesis (str): Generated text
        model: NLI pipeline
    Returns:
        bool: True if entailment score > 0.7
    """
    input_text = f"{premise} [SEP] {hypothesis}"
    result = model(input_text)
    return result[0]["label"] == "entailment" and result[0]["score"] > 0.7


def generate_and_verify(input_data, t5_model, t5_tokenizer, nli_model):
    """
    Generate text and pick best candidate using NLI.
    Args:
        input_data (str): Input data
        t5_model: T5 model
        t5_tokenizer: T5 tokenizer
        nli_model: NLI pipeline
    Returns:
        str: Best entailed text
    """
    prompt = f"Generate text from: {input_data}"
    inputs = t5_tokenizer(prompt, return_tensors="pt")
    outputs = t5_model.generate(
        inputs["input_ids"], num_beams=3, num_return_sequences=3
    )
    candidates = [t5_tokenizer.decode(out, skip_special_tokens=True) for out in outputs]

    best_candidate = None
    best_score = 0
    for cand in candidates:
        if check_entailment(input_data, cand, nli_model):
            score = nli_model(f"{input_data} [SEP] {cand}")[0]["score"]
            if score > best_score:
                best_score = score
                best_candidate = cand
    return best_candidate


# Example usage
if __name__ == "__main__":
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    nli_model = pipeline("text-classification", model="facebook/bart-large-mnli")

    input_data = "Team A wins, Team B, score 2-1"
    print(
        f"Best text: {generate_and_verify(input_data, t5_model, t5_tokenizer, nli_model)}"
    )

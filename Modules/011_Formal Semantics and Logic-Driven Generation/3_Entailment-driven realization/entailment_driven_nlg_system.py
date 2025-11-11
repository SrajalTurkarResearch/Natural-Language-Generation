# entailment_driven_nlg_system.py
# Purpose: Major project to generate text from ToTTo dataset and verify with NLI
# For aspiring scientists: Build a full NLG system with entailment
# Dependencies: transformers, datasets (pip install transformers datasets)

from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
from datasets import load_dataset


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


def generate_and_verify(table, t5_model, t5_tokenizer, nli_model):
    """
    Generate text from table and pick best via NLI.
    Args:
        table (dict): ToTTo table data
        t5_model: T5 model
        t5_tokenizer: T5 tokenizer
        nli_model: NLI pipeline
    Returns:
        str: Best entailed text
    """
    prompt = f"Generate text from table: {table['table']}"
    inputs = t5_tokenizer(prompt, return_tensors="pt")
    outputs = t5_model.generate(
        inputs["input_ids"], num_beams=3, num_return_sequences=3
    )
    candidates = [t5_tokenizer.decode(out, skip_special_tokens=True) for out in outputs]

    best_candidate = None
    best_score = 0
    for cand in candidates:
        if check_entailment(str(table["table"]), cand, nli_model):
            score = nli_model(f"{table['table']} [SEP] {cand}")[0]["score"]
            if score > best_score:
                best_score = score
                best_candidate = cand
    return best_candidate


# Example usage
if __name__ == "__main__":
    # Load models
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    nli_model = pipeline("text-classification", model="facebook/bart-large-mnli")

    # Load ToTTo dataset (small subset)
    totto = load_dataset("totto", split="train[:10]")

    # Generate and verify for first table
    table = totto[0]
    print(f"Table: {table['table']}")
    print(
        f"Generated Text: {generate_and_verify(table, t5_model, t5_tokenizer, nli_model)}"
    )

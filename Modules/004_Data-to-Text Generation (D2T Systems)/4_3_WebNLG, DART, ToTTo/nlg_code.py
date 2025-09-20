# nlg_code.py
# Practical code for NLG using WebNLG, DART, and ToTTo-like inputs with T5
# Includes inference and fine-tuning for beginner scientists

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output


def run_inference():
    """
    Performs inference using a pre-trained T5 model on sample inputs.
    Demonstrates text generation for WebNLG, DART, and ToTTo-like inputs.
    """
    print("\n## NLG Inference with Pre-trained T5\n")

    # Initialize model and tokenizer
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Sample inputs
    inputs = [
        "generate text: Alan_Bean | occupation | Astronaut | birthPlace | Wheeler_Texas",  # WebNLG
        "generate text: Empire_State_Building | height | 443.2_meters | location | New_York_City",  # DART
        "generate text: table | Name | Marie_Curie | Occupation | Scientist",  # ToTTo
    ]

    # Generate text
    for input_text in inputs:
        inputs_tokenized = tokenizer(
            input_text, return_tensors="pt", max_length=512, truncation=True
        )
        outputs = model.generate(**inputs_tokenized, max_length=50, num_beams=5)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Input: {input_text}")
        print(f"Output: {generated_text}\n")


def fine_tune_model():
    """
    Fine-tunes T5 on a small WebNLG subset.
    Note: Full fine-tuning requires significant compute; this is a demo.
    """
    print("\n## Fine-Tuning T5 on WebNLG\n")

    # Load dataset
    dataset = load_dataset("web_nlg", "release_v3.0_en")
    train_data = dataset["train"].select(range(100))  # Use 100 examples for demo

    # Preprocess data
    def preprocess(example):
        triples = " | ".join(
            [
                f"{t['subject']} | {t['property']} | {t['object']}"
                for t in example["modified_tripleset"]
            ]
        )
        text = example["lex"]["text"][0]
        return {"input_text": f"generate text: {triples}", "target_text": text}

    train_data = train_data.map(preprocess)

    # Initialize tokenizer and model
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Tokenize dataset
    def tokenize_function(examples):
        inputs = tokenizer(
            examples["input_text"],
            max_length=512,
            truncation=True,
            padding="max_length",
        )
        targets = tokenizer(
            examples["target_text"],
            max_length=128,
            truncation=True,
            padding="max_length",
        )
        inputs["labels"] = targets["input_ids"]
        return inputs

    tokenized_data = train_data.map(tokenize_function, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
    )

    # Train
    print("Starting fine-tuning (this may take a few minutes)...")
    trainer.train()
    print("Fine-tuning complete!")

    # Save model
    model.save_pretrained("./fine_tuned_t5")
    tokenizer.save_pretrained("./fine_tuned_t5")
    print("Model saved to './fine_tuned_t5'")


def evaluate_bleu():
    """
    Computes BLEU score for a sample generated text.
    """
    print("\n## Evaluating with BLEU Score\n")
    reference = ["Alan Bean is an astronaut born in Wheeler, Texas."]
    candidate = "Alan Bean, an astronaut, was born in Wheeler, Texas."
    score = sentence_bleu([ref.split() for ref in reference], candidate.split())
    print(f"Reference: {reference[0]}")
    print(f"Candidate: {candidate}")
    print(f"BLEU Score: {score:.3f}")


if __name__ == "__main__":
    run_inference()
    evaluate_bleu()
    # Uncomment to run fine-tuning (requires compute resources)
    # fine_tune_model()

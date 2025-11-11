from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset


def fine_tune_gpt2():
    """
    Fine-tune GPT-2 on an emotion-labeled dataset for affective text generation.
    """
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Load emotion dataset (replace with EmoBank or similar for real use)
    dataset = load_dataset("emotion")

    # Preprocess dataset
    def prepare_text(examples):
        return tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=128
        )

    prepared_dataset = dataset.map(prepare_text, batched=True)

    # Define training arguments
    training_settings = TrainingArguments(
        output_dir="./emotional_gpt2",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_settings,
        train_dataset=prepared_dataset["train"],
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained("./emotional_gpt2")
    tokenizer.save_pretrained("./emotional_gpt2")
    print("Model fine-tuned and saved!")


if __name__ == "__main__":
    fine_tune_gpt2()

"""
online_learning.py
==================
Module 4: Online Learning with Hugging Face
Updates model in real-time from new data.
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load small model (safe for CPU)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


def online_update(prompt: str, good_response: str, reward: float = 0.9):
    """
    Perform one online update with feedback reward.

    Args:
        prompt: Input text
        good_response: Desired output
        reward: How good the response is (0.0 to 1.0)
    """
    print(f"🔄 ONLINE UPDATE")
    print(f"Prompt: {prompt}")
    print(f"Target: {good_response}")
    print(f"Reward: {reward}\n")

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    labels = tokenizer(good_response, return_tensors="pt")["input_ids"]

    # Forward pass
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss

    # Add reward signal (simplified RLHF)
    reward_loss = -reward * torch.logsumexp(outputs.logits.flatten(), dim=0)
    total_loss = loss + reward_loss

    # Backward + update
    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"✅ Model updated! Loss: {loss.item():.3f}\n")


# === DEMO ===
if __name__ == "__main__":
    online_update(
        prompt="The dog is",
        good_response="The dog is happy and wagging its tail.",
        reward=0.95,
    )

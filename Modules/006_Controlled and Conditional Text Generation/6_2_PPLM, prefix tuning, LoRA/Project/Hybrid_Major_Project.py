# Hybrid Major Project: Scientific NLG
# Combine all
from PPLM_Practical_Code import attribute_gradient

# ... Import others


def hybrid_generate(prompt):
    h_new = attribute_gradient(torch.tensor([0.5, -0.3, 0.2]))
    # Simulate prefix and LoRA
    output = f"Hypothesis: {prompt} (nudged: {h_new})"
    print(output)


hybrid_generate("Gene X regulates...")

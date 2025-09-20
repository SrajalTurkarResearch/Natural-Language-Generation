# Image-to-Text: Practical Captioning
# This file shows how to generate captions using the BLIP model, with step-by-step explanations.

# Install: pip install transformers pillow requests matplotlib torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import matplotlib.pyplot as plt

# --- Theory ---
# BLIP (Bootstrapping Language-Image Pre-training) is a model that combines vision (CNN/ViT) and language (Transformer) to caption images.
# Steps:
# 1. Load image and turn it into numbers (processor).
# 2. Model generates caption by predicting words one by one.
# 3. Decode numbers back to words.


# --- Code Guide ---
def generate_caption():
    # Load model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    # Load image (replace with a real URL, e.g., from COCO dataset)
    url = (
        "https://images.cocodataset.org/val2017/000000039769.jpg"  # Two cats on a couch
    )
    try:
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    except:
        print("URL failed. Use a local image or check the link.")
        return

    # Generate caption
    inputs = processor(image, return_tensors="pt")  # Turn image into numbers
    out = model.generate(**inputs)  # Predict words
    caption = processor.decode(
        out[0], skip_special_tokens=True
    )  # Turn numbers to words
    print("Generated Caption:", caption)

    # Visualize
    plt.imshow(image)
    plt.title(caption)
    plt.axis("off")
    plt.show()

    return caption


# --- Exercise ---
# 1. Run the code with a different image URL (e.g., a dog or nature scene).
# 2. Write down: Does the caption make sense? Why or why not?

# --- Research Insight ---
# BLIP's strength is pre-training on large datasets (e.g., web images). As a scientist, investigate how pre-training affects performance on specific domains like medical imaging.

if __name__ == "__main__":
    print("Running Practical Captioning")
    generate_caption()

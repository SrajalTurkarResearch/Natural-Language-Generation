# ecommerce_product_description.py
# Real-World NLG: SEO-Optimized Product Copy


def generate_product_description(product):
    """
    Generate 3 versions: Short, Long, SEO
    """
    name = product["name"]
    brand = product["brand"]
    features = product["features"]
    material = product.get("material", "premium")
    target = product.get("target", "everyone")

    short = f"{brand} {name} – {material.title()} quality, built to last."

    long_desc = f"""
Introducing the {brand} {name}, designed for {target}.

Crafted from {material} materials, this product combines durability with elegance. 
Key features include: {', '.join(features[:-1])}, and {features[-1]}.

Whether you're at home, office, or on the go, the {name} delivers unmatched performance.
"""
    seo = f"Buy {brand} {name} online – Best price | {', '.join(features)} | Free shipping"

    return {"short": short.strip(), "long": long_desc.strip(), "seo": seo}


# === SAMPLE PRODUCT ===
laptop = {
    "name": "ProBook X1",
    "brand": "TechGear",
    "features": [
        "14-inch 4K display",
        "Intel i7",
        "16GB RAM",
        "512GB SSD",
        "backlit keyboard",
    ],
    "material": "aluminum",
    "target": "professionals and students",
}

if __name__ == "__main__":
    desc = generate_product_description(laptop)
    print("SHORT:", desc["short"], "\n")
    print("LONG:\n", desc["long"], "\n")
    print("SEO:", desc["seo"])

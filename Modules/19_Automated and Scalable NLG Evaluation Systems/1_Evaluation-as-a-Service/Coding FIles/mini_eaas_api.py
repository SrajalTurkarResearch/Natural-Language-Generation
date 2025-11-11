# mini_eaas_api.py
# Mini Evaluation-as-a-Service API
# Run with: python mini_eaas_api.py
# Test with curl or Postman

from flask import Flask, request, jsonify
from bleu_from_scratch import bleu_score
from rouge_implementation import rouge_n, rouge_l
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)

app = Flask(__name__)


@app.route("/evaluate", methods=["POST"])
def evaluate():
    """
    API Endpoint: POST /evaluate
    Input JSON:
    {
        "generated": "Your text here",
        "reference": "Reference text here"
    }
    """
    try:
        data = request.get_json()
        gen = data["generated"]
        ref = data["reference"]

        # Tokenize
        gen_tokens = word_tokenize(gen.lower())
        ref_tokens = word_tokenize(ref.lower())

        # Compute metrics
        results = {
            "bleu": round(bleu_score(gen, ref), 4),
            "rouge_1": round(rouge_n(gen_tokens, ref_tokens, 1), 4),
            "rouge_2": round(rouge_n(gen_tokens, ref_tokens, 2), 4),
            "rouge_l": round(rouge_l(gen_tokens, ref_tokens), 4),
            "length_gen": len(gen_tokens),
            "length_ref": len(ref_tokens),
        }

        return jsonify({"status": "success", "metrics": results})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/")
def home():
    return """
    <h1>Mini EaaS API</h1>
    <p>POST to /evaluate with JSON: {"generated": "...", "reference": "..."}</p>
    """


if __name__ == "__main__":
    print("Starting Mini EaaS Server on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)

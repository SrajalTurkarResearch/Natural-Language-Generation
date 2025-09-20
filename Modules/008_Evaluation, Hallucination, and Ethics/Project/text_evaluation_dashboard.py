import os
import sqlite3
import bleach
import logging
import traceback
import streamlit as st
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Initialize SQLite database
def init_db():
    conn = sqlite3.connect("feedback.db")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            flag TEXT NOT NULL,
            comment TEXT
        )
    """
    )
    conn.commit()
    conn.close()


init_db()

# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)


def calculate_metrics(generated_text, reference_text):
    metrics = {}
    try:
        bleu = sentence_bleu(
            [reference_text.split()],
            generated_text.split(),
            smoothing_function=SmoothingFunction().method1,
        )
        metrics["BLEU"] = round(bleu, 4)
    except Exception as e:
        logger.error("BLEU calculation failed: %s", str(e), exc_info=True)
        metrics["BLEU"] = 0.0

    try:
        rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        rouge_scores = rouge.score(reference_text, generated_text)
        metrics["ROUGE-1"] = round(rouge_scores["rouge1"].fmeasure, 4)
        metrics["ROUGE-L"] = round(rouge_scores["rougeL"].fmeasure, 4)
    except Exception as e:
        logger.error("ROUGE calculation failed: %s", str(e), exc_info=True)
        metrics["ROUGE-1"] = 0.0
        metrics["ROUGE-L"] = 0.0

    try:
        meteor = meteor_score([reference_text], generated_text)
        metrics["METEOR"] = round(meteor, 4)
    except Exception as e:
        logger.error("METEOR calculation failed: %s", str(e), exc_info=True)
        metrics["METEOR"] = 0.0

    try:
        bert_p, bert_r, bert_f1 = bert_score(
            [generated_text], [reference_text], lang="en", verbose=False
        )
        metrics["BERTScore"] = round(bert_f1.item(), 4)
    except Exception as e:
        logger.error("BERTScore calculation failed: %s", str(e), exc_info=True)
        metrics["BERTScore"] = 0.0

    return metrics


def get_feedback():
    conn = sqlite3.connect("feedback.db")
    cursor = conn.cursor()
    cursor.execute("SELECT text, flag, comment FROM feedback")
    feedback = [
        {"text": row[0], "flag": row[1], "comment": row[2]} for row in cursor.fetchall()
    ]
    conn.close()
    return feedback


def store_feedback(text, flag_value, comment):
    conn = sqlite3.connect("feedback.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO feedback (text, flag, comment) VALUES (?, ?, ?)",
        (text, flag_value, comment),
    )
    conn.commit()
    conn.close()


# Streamlit UI
st.set_page_config(page_title="Text Evaluation Dashboard", layout="wide")
st.title("Text Evaluation Dashboard")

st.markdown(
    """
    Enter a generated text and a reference text to evaluate using automated metrics (BLEU, ROUGE, METEOR, BERTScore).
    You can also flag problematic generations for further review.
    """
)

with st.form("evaluation_form"):
    generated_text = st.text_area("Generated Text", height=120)
    reference_text = st.text_area("Reference Text", height=120)
    submitted = st.form_submit_button("Evaluate")

if submitted:
    # Sanitize inputs
    generated_text_clean = bleach.clean(generated_text.strip())
    reference_text_clean = bleach.clean(reference_text.strip())

    # Validate input
    if not generated_text_clean or not reference_text_clean:
        st.error("Both generated text and reference text are required.")
    elif len(generated_text_clean) < 3 or len(reference_text_clean) < 3:
        st.error("Inputs must be at least 3 characters long.")
    else:
        with st.spinner("Calculating metrics..."):
            try:
                metrics = calculate_metrics(generated_text_clean, reference_text_clean)
                st.success("Evaluation complete!")
                st.subheader("Automated Metrics")
                st.table(metrics)
            except Exception as e:
                logger.error("Evaluation error: %s", str(e), exc_info=True)
                st.error(f"Evaluation failed: {str(e)}")

st.markdown("---")
st.header("Flag a Generation")

with st.form("flag_form"):
    flag_text = st.text_area("Text to Flag", height=80, key="flag_text")
    flag_value = st.selectbox(
        "Flag Type",
        ["", "factual_error", "incoherence", "off_topic"],
        format_func=lambda x: {
            "": "Select a flag type",
            "factual_error": "Factual Error",
            "incoherence": "Incoherence",
            "off_topic": "Off Topic",
        }.get(x, x),
    )
    flag_comment = st.text_area("Comment (optional)", height=60, key="flag_comment")
    flag_submitted = st.form_submit_button("Submit Flag")

if flag_submitted:
    # Sanitize inputs
    text_clean = bleach.clean(flag_text.strip())
    flag_value_clean = bleach.clean(flag_value.strip())
    comment_clean = bleach.clean(flag_comment.strip())

    # Validate inputs
    if not text_clean or not flag_value_clean:
        st.error("Both text and flag are required.")
    elif len(text_clean) < 3:
        st.error("Text must be at least 3 characters long.")
    elif flag_value_clean not in ["factual_error", "incoherence", "off_topic"]:
        st.error("Invalid flag type.")
    else:
        try:
            store_feedback(text_clean, flag_value_clean, comment_clean)
            st.success("Feedback submitted successfully!")
        except Exception as e:
            logger.error("Flag submission error: %s", str(e), exc_info=True)
            st.error(f"Feedback submission failed: {str(e)}")

st.markdown("---")
st.header("Flagged Feedback")

feedback = get_feedback()
if feedback:
    for entry in feedback:
        with st.expander(f"Flag: {entry['flag']} | Text: {entry['text'][:40]}..."):
            st.write(f"**Text:** {entry['text']}")
            st.write(f"**Flag:** {entry['flag']}")
            if entry["comment"]:
                st.write(f"**Comment:** {entry['comment']}")
else:
    st.info("No feedback submitted yet.")
    app.run(host="0.0.0.0", port=port, debug=False)

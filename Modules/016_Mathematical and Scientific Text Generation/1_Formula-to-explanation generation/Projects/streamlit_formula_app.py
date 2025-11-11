# streamlit_formula_app.py
"""
Bonus: Web App – Formula to Explanation Dashboard
Run with: streamlit run streamlit_formula_app.py
"""

import streamlit as st
from transformers import pipeline

st.title("Formula to Explanation Generator")
st.write("Enter any math formula — get a clear explanation instantly.")

explainer = pipeline("text2text-generation", model="t5-base")

formula = st.text_input("Enter formula (e.g., E = mc^2):", "x^2 + 2x + 1 = 0")
if st.button("Explain"):
    with st.spinner("Generating..."):
        prompt = f"Explain this formula clearly: {formula}"
        result = explainer(prompt, max_length=120)[0]["generated_text"]
        st.success("Explanation:")
        st.write(result)

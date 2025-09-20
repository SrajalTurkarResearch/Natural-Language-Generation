# nlg_app.py: Streamlit App for Controlled NLG
# Run: streamlit run nlg_app.py
# Requires: streamlit, transformers, peft

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PromptTuningConfig, get_peft_model


def main():
    st.title("Controlled NLG App")
    prompt = st.text_input("Enter prompt", "Once upon a time")
    tone = st.selectbox("Select tone", ["Happy", "Formal", "Neutral"])

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    config = PromptTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=5)
    peft_model = get_peft_model(model, config)

    if st.button("Generate"):
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = peft_model.generate(**inputs, max_length=100)
        st.write("Output:", tokenizer.decode(outputs[0]))


if __name__ == "__main__":
    main()

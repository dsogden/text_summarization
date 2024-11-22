import streamlit as st
import torch

from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-large")
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-large")


def get_summary(
        model: T5ForConditionalGeneration, input_text: str
    ) -> str:
    """
    Returns the number of top words to be summarized from the text.
    """
    input_ids = tokenizer.encode("summarize: " + input_text, return_tensors="pt")
    outputs = model.generate(
        input_ids, max_length=150, min_length=50, num_beams=4, early_stopping=True
    )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

st.title("Text Summarizer")
st.write("Enter a text below to summarize the text:")
user_input = st.text_area('Text to analyze', placeholder="Type something...")
if st.button('Summarize Text'):
    if len(user_input) > 0:
        summary = get_summary(model, user_input)
        st.write(summary)
    else:
        st.warning("Please enter some text to analyze.")

# Footer
st.markdown("Created with ❤️ using Streamlit and Hugging Face Transformers.")

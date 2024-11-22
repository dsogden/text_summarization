import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('t5-large')
model = T5ForConditionalGeneration.from_pretrained('t5-large')

def get_keywords(
        model: TT5ForConditionalGeneration, input_text: str
    ) -> str:
    """
    Returns the number of top words to be summarized from the text.
    """
    input_text = "Your book description here"
    input_ids = tokenizer.encode("summarize: " + input_text, return_tensors="pt")
    outputs = model.generate(
        input_ids, max_length=150, min_length=50, num_beams=4, early_stopping=True
    )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

st.title("Text Summarizer")
st.write("Enter a text below to summarize the text:")
user_input = st.text_area('Text to analyze', placeholder="Type something...")


# ML-Based Document Accessibility Tool (PyTorch Version)
# All-in-one script using Streamlit for UI, NLP for text processing, and TTS

import streamlit as st
import pyttsx3
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Load summarization pipeline using PyTorch-backed model
model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, framework="pt")

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize TTS engine
engine = pyttsx3.init()


def simplify_text(text):
    """Basic simplification using sentence splitting."""
    doc = nlp(text)
    return " ".join([sent.text for sent in doc.sents])


def summarize_text(text):
    """Summarize using Hugging Face Transformers (PyTorch)."""
    if len(text.split()) < 50:
        return text
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']


def text_to_speech(text):
    """Convert text to speech using pyttsx3."""
    engine.say(text)
    engine.runAndWait()


def personalize_content(text, disability):
    """Personalize content based on disability."""
    if disability == "Visual Impairment":
        st.info("Playing audio version...")
        text_to_speech(text)
        return "Audio output played."

    elif disability == "Hearing Impairment":
        st.success("Transcript displayed.")
        return text

    elif disability == "Cognitive Disability":
        st.info("Simplifying and summarizing text...")
        simplified = simplify_text(text)
        summarized = summarize_text(simplified)
        return summarized

    else:
        return text


# Streamlit UI
st.set_page_config(page_title="Document Accessibility Tool", layout="centered")
st.title("📄 ML-Based Document Accessibility Tool (PyTorch)")

uploaded_file = st.file_uploader("Upload a text file (.txt)", type=["txt"])
disability = st.selectbox("Select Disability Type", ["Visual Impairment", "Hearing Impairment", "Cognitive Disability"])

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
    st.subheader("Original Text")
    st.text_area("Input Text", text, height=200)

    if st.button("Personalize Content"):
        result = personalize_content(text, disability)
        if disability != "Visual Impairment":
            st.subheader("Personalized Output")
            st.text_area("Output", result, height=200)

st.markdown("---")
st.caption("Built with 💡 using PyTorch, Streamlit, and NLP")

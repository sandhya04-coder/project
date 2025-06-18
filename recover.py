# app.py
# app.py
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pyttsx3
import os

# -------------------- Streamlit UI Config --------------------
st.set_page_config(page_title="ML-Based Document Accessibility Tool", layout="wide")

# ----------------------- Setup -----------------------
@st.cache_resource

def load_models():
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    return summarizer, tokenizer, model

summarizer, tokenizer, simplifier_model = load_models()

# --------------------- Functions ----------------------

from transformers import MarianMTModel, MarianTokenizer

@st.cache_resource
def load_translation_model():
    model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"  # Covers en→fr/es/it/ro/pt
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

translation_tokenizer, translation_model = load_translation_model()

def translate_text(text, target_lang_code="fr"):
    src_text = f">>{target_lang_code}<< {text}"
    input_ids = translation_tokenizer(src_text, return_tensors="pt", padding=True, truncation=True).input_ids
    outputs = translation_model.generate(input_ids, max_length=512)
    return translation_tokenizer.decode(outputs[0], skip_special_tokens=True)

def simplify_text(text):
    input_ids = tokenizer("simplify: " + text, return_tensors="pt", max_length=512, truncation=True).input_ids
    outputs = simplifier_model.generate(input_ids, max_length=256, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def format_text_for_dyslexia(text):
    # Add extra spacing between words and lines
    return "  ".join(text.split()).replace(".", ".\n\n")

# -------------------- Streamlit UI --------------------
st.title("📄 ML-Based Document Accessibility Tool")

st.markdown("""
This tool **personalizes e-learning content** for users with various accessibility needs using **NLP and PyTorch**.
Upload a `.txt` file and select a preferred mode below:
""")

# Upload file
uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

disability_type = st.selectbox(
    "Select a target accessibility mode:",
    ["General (Summarize)", "Visual Impairment (Speech Output)", "Cognitive Disability (Simplify Text)", "Learning Disability (Enhanced Spacing)","Translate to French", "Translate to Spanish", "Translate to Italian", "Translate to Romanian", "Translate to Portuguese"]
)

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")

    with st.expander("📄 Original Text", expanded=False):
        st.write(text)

    st.markdown("---")
    st.subheader("🛠️ Processed Output")

    if disability_type == "General (Summarize)":
        with st.spinner("Summarizing text..."):
            summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            st.success("Summary Generated!")
            st.write(summary)

    elif disability_type == "Visual Impairment (Speech Output)":
        st.success("Reading the text aloud...")
        st.write(text[:1000])
        speak_text(text[:1000])

    elif disability_type == "Cognitive Disability (Simplify Text)":
        with st.spinner("Simplifying content..."):
            simplified = simplify_text(text[:512])
            st.success("Simplified Version")
            st.write(simplified)

    elif disability_type == "Learning Disability (Enhanced Spacing)":
        formatted = format_text_for_dyslexia(text)
        st.success("Formatted for better readability")
        st.text(formatted)

    elif disability_type == "Translate to French":
       with st.spinner("Translating..."):
          translated = translate_text(text[:512], target_lang_code="fr")
          st.success("Translated Version (French):")
          st.write(translated)

    elif disability_type == "Translate to Spanish":
        with st.spinner("Translating..."):
            translated = translate_text(text[:512], target_lang_code="es")
            st.success("Translated Version (Spanish):")
            st.write(translated)

    elif disability_type == "Translate to Italian":
        with st.spinner("Translating..."):
            translated = translate_text(text[:512], target_lang_code="it")
            st.success("Translated Version (Italian):")
            st.write(translated)

    elif disability_type == "Translate to Romanian":
        with st.spinner("Translating..."):
            translated = translate_text(text[:512], target_lang_code="ro")
            st.success("Translated Version (Romanian):")
            st.write(translated)

    elif disability_type == "Translate to Portuguese":
        with st.spinner("Translating..."):
            translated = translate_text(text[:512], target_lang_code="pt")
            st.success("Translated Version (Portuguese):")
            st.write(translated)

    else:
        st.error("Please select a valid option.")

    st.download_button("⬇️ Download Processed Text", data=text.encode('utf-8'), file_name="processed_output.txt")

st.markdown("---")
st.caption("Built with ❤️ using PyTorch, HuggingFace Transformers, and Streamlit")

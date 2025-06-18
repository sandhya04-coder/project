import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel, MarianTokenizer
import pyttsx3
import torch

# -------------------- Streamlit UI Config --------------------
st.set_page_config(page_title="ML-Based Document Accessibility Tool", layout="wide")

# ----------------------- Load Models -----------------------
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    summarizer_tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    simplifier_tokenizer = AutoTokenizer.from_pretrained("t5-small")
    simplifier_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    return summarizer, summarizer_tokenizer, simplifier_tokenizer, simplifier_model

summarizer, summarizer_tokenizer, simplifier_tokenizer, simplifier_model = load_models()

@st.cache_resource
def load_translation_model():
    model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

translation_tokenizer, translation_model = load_translation_model()

# --------------------- Utility Functions ---------------------

def translate_text(text, target_lang_code="fr"):
    src_text = f">>{target_lang_code}<< {text}"
    input_ids = translation_tokenizer(src_text, return_tensors="pt", padding=True, truncation=True).input_ids
    outputs = translation_model.generate(input_ids, max_length=512)
    return translation_tokenizer.decode(outputs[0], skip_special_tokens=True)

def truncate_text(text, tokenizer, max_tokens=512):
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens, skip_special_tokens=True)

def simplify_text(text):
    input_text = truncate_text("simplify: " + text, simplifier_tokenizer, max_tokens=512)
    input_ids = simplifier_tokenizer(input_text, return_tensors="pt").input_ids
    outputs = simplifier_model.generate(input_ids, max_length=256, num_beams=4, early_stopping=True)
    return simplifier_tokenizer.decode(outputs[0], skip_special_tokens=True)

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def format_text_for_dyslexia(text):
    return "  ".join(text.split()).replace(".", ".\n\n")

# -------------------- Chunking and Safe Summarization --------------------

def chunk_text(text, tokenizer, max_tokens=1024):
    tokens = tokenizer.encode(text, truncation=False)
    chunks = []

    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        chunks.append(chunk_text)

    return chunks

def summarize_text_safe(text):
    try:
        if len(summarizer_tokenizer.encode(text)) <= 1024:
            return summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        else:
            chunks = chunk_text(text, summarizer_tokenizer)
            summaries = []
            for chunk in chunks:
                try:
                    summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
                    summaries.append(summary)
                except Exception as e:
                    summaries.append("[Error summarizing chunk]")
            return " ".join(summaries)
    except Exception as e:
        return f"[Summarization Error] {str(e)}"

# -------------------- Streamlit UI --------------------

st.title("📄 ML-Based Document Accessibility Tool")

st.markdown("""
This tool **personalizes e-learning content** for users with various accessibility needs using **NLP and PyTorch**.
Upload a `.txt` file and select a preferred mode below:
""")

uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

disability_type = st.selectbox(
    "Select a target accessibility mode:",
    [
        "General (Summarize)",
        "Visual Impairment (Speech Output)",
        "Cognitive Disability (Simplify Text)",
        "Learning Disability (Enhanced Spacing)",
        "Translate to French", "Translate to Spanish", "Translate to Italian",
        "Translate to Romanian", "Translate to Portuguese"
    ]
)

processed_output = None

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")

    with st.expander("📄 Original Text", expanded=False):
        st.write(text)

    st.markdown("---")
    st.subheader("🛠️ Processed Output")

    if disability_type == "General (Summarize)":
        with st.spinner("Summarizing text..."):
            processed_output = summarize_text_safe(text)
            st.success("Summary Generated!")
            st.write(processed_output)

    elif disability_type == "Visual Impairment (Speech Output)":
        st.success("Reading the text aloud...")
        short_text = truncate_text(text, summarizer_tokenizer, max_tokens=512)
        st.write(short_text)
        speak_text(short_text)
        processed_output = short_text

    elif disability_type == "Cognitive Disability (Simplify Text)":
        with st.spinner("Simplifying content..."):
            processed_output = simplify_text(text)
            st.success("Simplified Version")
            st.write(processed_output)

    elif disability_type == "Learning Disability (Enhanced Spacing)":
        processed_output = format_text_for_dyslexia(text)
        st.success("Formatted for better readability")
        st.text(processed_output)

    elif "Translate to" in disability_type:
        lang_code = {
            "Translate to French": "fr",
            "Translate to Spanish": "es",
            "Translate to Italian": "it",
            "Translate to Romanian": "ro",
            "Translate to Portuguese": "pt"
        }[disability_type]

        with st.spinner("Translating..."):
            input_text = truncate_text(text, translation_tokenizer, max_tokens=512)
            processed_output = translate_text(input_text, target_lang_code=lang_code)
            st.success(f"Translated Version ({lang_code.upper()}):")
            st.write(processed_output)

    else:
        st.error("Please select a valid option.")

    if processed_output:
        st.download_button("⬇️ Download Processed Text", data=processed_output.encode('utf-8'), file_name="processed_output.txt")

st.markdown("---")
st.caption("Built with ❤️ using PyTorch, HuggingFace Transformers, and Streamlit")

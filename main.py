import streamlit as st
import pandas as pd
from utils import generate_wordcloud, extract_keywords, analyze_sentiment

st.title("📊 NLP Data Visualization Assistant")

# File uploader
uploaded_file = st.file_uploader("Upload text data (CSV/TXT)", type=["csv", "txt"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        text_data = " ".join(df.iloc[:, 0].astype(str))
    else:
        text_data = uploaded_file.read().decode("utf-8")

    # Word Cloud
    st.subheader("📌 Word Cloud Visualization")
    st.pyplot(generate_wordcloud(text_data))

    # Keyword Extraction
    st.subheader("📌 Keyword Extraction (TF-IDF)")
    keywords = extract_keywords(text_data.split("\n"))
    st.write(", ".join(keywords))

    # Sentiment Analysis
    st.subheader("📌 Sentiment Analysis")
    sentiment_score = analyze_sentiment(text_data)
    st.write(f"Overall Sentiment Score: {sentiment_score:.2f}")

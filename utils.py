import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download NLTK data
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

def generate_wordcloud(text):
    """ Generate a word cloud from input text """
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    return plt

def extract_keywords(text_list, top_n=10):
    """ Extract top N keywords using TF-IDF """
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(text_list)
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1
    keywords = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)[:top_n]
    return [kw[0] for kw in keywords]

def analyze_sentiment(text):
    """ Compute sentiment score (-1 to 1) """
    return sia.polarity_scores(text)["compound"]

import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
import nltk

@st.cache_resource
def load_stopwords():
    nltk.download("stopwords")
    return stopwords.words("english")

@st.cache_resource
def load_model_and_vectorizer():
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    return model, vectorizer

def predict_sentiment(text, model, vectorizer, stop_words):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()

    text = [word for word in text if word not in stop_words]
    text = " ".join(text)

    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)[0]

    return "Negative" if prediction == 0 else "Positive"

def main():
    st.title("AI Powered X Sentiment Analyzer")

    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()

    text_input = st.text_area("Enter text to analyze sentiment")

    if st.button("Analyze"):
        if text_input.strip() == "":
            st.warning("Please enter some text")
        else:
            sentiment = predict_sentiment(
                text_input,
                model,
                vectorizer,
                stop_words
            )

            if sentiment == "Positive":
                st.success(f"Sentiment: {sentiment}")
            else:
                st.error(f"Sentiment: {sentiment}")

if __name__ == "__main__":
    main()

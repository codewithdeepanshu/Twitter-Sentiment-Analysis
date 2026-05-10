import os
import streamlit as st
import tweepy
import pickle
import re
import nltk
from nltk.corpus import stopwords

# ---------------- API KEYS ----------------
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

# ---------------- NLTK ----------------
nltk.download('stopwords')
stop_words = stopwords.words('english')

# ---------------- LOAD MODEL ----------------
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# ---------------- TWEEPY CLIENT ----------------
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# ---------------- SENTIMENT FUNCTION ----------------
def predict_sentiment(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()

    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)

    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)[0]

    return "Negative" if prediction == 0 else "Positive"

# ---------------- STREAMLIT UI ----------------
st.title("AI Powered X Sentiment Analyzer")

username = st.text_input("Enter X Username")

if st.button("Analyze User Tweets"):

    try:
        user = client.get_user(username=username)

        user_id = user.data.id

        tweets = client.get_users_tweets(
            id=user_id,
            max_results=5
        )

        if tweets.data:

            for tweet in tweets.data:

                sentiment = predict_sentiment(tweet.text)

                if sentiment == "Positive":
                    st.success(f"{tweet.text}\n\nSentiment: {sentiment}")

                else:
                    st.error(f"{tweet.text}\n\nSentiment: {sentiment}")

        else:
            st.warning("No tweets found.")

    except Exception as e:
        st.error(f"Error: {e}")

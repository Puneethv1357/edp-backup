import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Load model and tokenizer
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/best_emotion_model.keras")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer/tokenizer.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
tokenizer = load_tokenizer()

emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
maxlen = 200

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Tweet Emotion Classifier", layout="centered")
st.title("💬 Tweet Emotion Classifier")
st.write("Enter a tweet below to detect its emotion:")

tweet = st.text_area("📝 Your Tweet", height=100)

if st.button("Predict Emotion"):
    if not tweet.strip():
        st.warning("Please enter a tweet!")
    else:
        seq = tokenizer.texts_to_sequences([tweet])
        padded = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
        prediction = model.predict(padded)
        predicted_label = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        st.success(f"Predicted Emotion: **{predicted_label}**")
        st.write(f"Confidence: `{confidence:.2f}`")

        st.bar_chart(prediction.flatten())

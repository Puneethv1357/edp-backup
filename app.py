import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
import altair as alt
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Background style
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to right, #black, #c3cfe2);
        }
        .centered-textarea textarea {
            margin: 0 auto;
            display: block;
            border-radius: 10px;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Load model and tokenizer
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
emoji_map = {
    'joy': '😊', 'sadness': '😢', 'anger': '😠',
    'love': '❤️', 'fear': '😨', 'surprise': '😲'
}
maxlen = 50

# Title
st.markdown("""
    <div style='text-align: center; padding-top: 10px;'>
        <h1 style='color:#4e79a7;'>Tweet Emotion Classifier 💬</h1>
        <p style='color:#5d63b9;'>Type a tweet or review to find its emotion.</p>
    </div>
""", unsafe_allow_html=True)

# Input
st.markdown("<div class='centered-textarea'>", unsafe_allow_html=True)
tweet = st.text_area(
    label="Your Tweet or Review:",
    placeholder="e.g. I'm feeling great today!",
    height=140
)
st.markdown("</div>", unsafe_allow_html=True)

# Predict
if st.button("Predict Emotion"):
    if not tweet.strip():
        st.warning("Please enter something.")
    else:
        seq = tokenizer.texts_to_sequences([tweet])
        padded = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
        prediction = model.predict(padded)
        predicted_label = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        st.markdown(f"""
            <div style='text-align: center; padding-top: 20px;'>
                <h2>{predicted_label} {emoji_map[predicted_label]}</h2>
                <p>Confidence: <code>{confidence:.2f}</code></p>
            </div>
        """, unsafe_allow_html=True)

        probs_df = pd.DataFrame({
            "Emotion": emotion_labels,
            "Confidence": prediction.flatten()
        })

        chart = alt.Chart(probs_df).mark_bar().encode(
            x=alt.X('Emotion', sort=None),
            y='Confidence',
            color=alt.value("#4e79a7")
        ).properties(width=500)

        st.altair_chart(chart, use_container_width=True)



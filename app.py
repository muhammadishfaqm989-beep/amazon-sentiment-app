import streamlit as st
import pickle
import os

st.title('Amazon Review Sentiment Analysis')
st.write('Type your review below and check if it is Positive or Negative')

# ---- Load model & vectorizer ----
MODEL_PATH = 'sentiment_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'

# Check if files exist
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
else:
    st.error("Model or vectorizer files not found! Please upload them in the same folder as app.py")
    st.stop()

# ---- User Input ----
user_input = st.text_area('Enter your review here:')

# ---- Prediction ----
if st.button('Predict Sentiment'):
    if user_input.strip() != '':
        review = user_input.lower()
        review_vector = vectorizer.transform([review])
        prediction = model.predict(review_vector)[0]
        
        if prediction == 'Positive':
            st.success(f'The sentiment is: {prediction} 😊')
        else:
            st.error(f'The sentiment is: {prediction} 😞')
    else:
        st.error('Please enter a review first')

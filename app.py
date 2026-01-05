import streamlit as st
import pickle

# Load model & vectorizer
with open('/content/sentiment_model.pkl','rb') as f:
    model = pickle.load(f)
with open('/content/tfidf_vectorizer.pkl','rb') as f:
    vectorizer = pickle.load(f)

st.title('Amazon Review Sentiment Analysis')
st.write('Type your review below and check if it is Positive or Negative')

user_input = st.text_area('Enter your review here:')

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

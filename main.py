import streamlit as st
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re


estimator = pickle.load(open('estimator', 'rb'))
# text_preprocessor = pickle.load(open('text_preprocessor', 'rb'))    not working for some reason
vectorizer = pickle.load(open('vectorizer', 'rb'))


def preprocess_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    temp = []
    pattern = r'^[a-zA-Z]+$'
    for word in words:
        if word not in stopwords.words('english') and re.search(pattern, word):
            temp.append(word)
    stemmer = PorterStemmer()
    temp = [stemmer.stem(word) for word in temp]

    processed_text = ' '.join(temp)
    return processed_text


st.title('Email Spam Classifier')

text = st.text_area(
    "Type email to analyze"
)

if st.button("Analyze"):
    text_preprocessed = preprocess_text(text)
    text_vector = vectorizer.transform([text_preprocessed])    # note that vectorizer needs an array of strings
    prediction = estimator.predict(text_vector)
    if prediction == 0:
        st.subheader("Analysis: No spam Good good 😏")
    else:
        st.write('Yajibu alayka an tastaghfirullah!! 🤨')


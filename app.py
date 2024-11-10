import streamlit as st
import pickle
import traceback
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    ps = PorterStemmer()

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]  # cloning instead of direct assigning
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

try:
    tfidf = pickle.load(open(r'D:\ML_2024\ML_2024\Spam Classifier\vectorizer.pkl','rb'))
    model = pickle.load(open(r'D:\ML_2024\ML_2024\Spam Classifier\model.pkl','rb'))

    st.title("SMS/EMAIL SPAM CLASSIFIER")

    input_sms = st.text_area("Enter the SMS/EMAIL")

    if st.button('Predict'):
        #1. Pre-process
        transformed_sms = transform_text(input_sms)
        #2. vectorize
        vector_input = tfidf.transform([transformed_sms])
        #3. Predict
        result = model.predict(vector_input)[0]
        #4. Display
        if result == 1:
            st.write("THIS IS A SPAM")
        else:
            st.write("THIS IS NOT A SPAM")

except:
    print(traceback.format_exc())
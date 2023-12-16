import streamlit as st
import pickle 
import string 
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords') 
from nltk.stem import PorterStemmer


tfidf = pickle.load(open("vectorizer.pkl","rb"))
model = pickle.load(open("model.pkl","rb"))

stemmer = PorterStemmer()

def preprocess_data(text):

  # lowercase
  text = text.lower()
  # word tokenization
  text = nltk.word_tokenize(text)

  y = []
  for i in text:
    if i.isalnum():
      y.append(i)
  
  text = y.copy()
  y.clear()

  for i in text:
    if i not in stopwords.words("english") and i not in string.punctuation:
      y.append(i)
    
  text = y.copy()
  y.clear()

  for i in text:
    y.append(stemmer.stem(i))

  return " ".join(y)

st.title("SMS Spam Classifier")

sms = st.text_area("Enter The Message:")
predict_button = st.button("Predict")


if predict_button:
    
    transform_sms = preprocess_data(sms)

    vector_input = tfidf.transform([transform_sms])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam") 
    else:
        st.header("Not Spam")
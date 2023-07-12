from flask import Flask, render_template,request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix
import seaborn as sns
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet
from nltk import pos_tag
from scipy.sparse import csr_matrix

app = Flask(__name__)

count_vector = pickle.load(open("Models/count-Vectorizer.pkl","rb"))
model = pickle.load(open("Models/Movies_Review_Classification.pkl","rb"))

def predict(text):
    sen = csr_matrix(count_vector.transform([text])).toarray()
    res = model.predict(sen)[0]
    return res

@app.route("/",methods=["GET","POST"])
def home():
    status =0
    if request.method=='POST':
        text = request.form.get("message")        
        res = predict(text)
        if res==1:
            status=1
        else:
            status=2

        return render_template("index.html",status=status)


    return render_template("index.html",status=status)

if __name__ == "__main__":
    app.debug=True
    app.run(port=8000)

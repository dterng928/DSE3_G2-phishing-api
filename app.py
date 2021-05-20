from flask import Flask, request
import pandas as pd
import numpy as np

import re
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
import pickle

tokenizer = RegexpTokenizer(r'[A-Za-z]+')
def splitUrlIntoToken(url):
    # print(url)
    return tokenizer.tokenize(url)

def removeTextFromList(input, rmTxt):
    input = [x for x in input if (x not in rmTxt)]
    return input

snowball = SnowballStemmer("english")
def textNormByStemType(inputTxt, stemtype):
    inputTxt =[stemtype.stem(word) for word in inputTxt]
    return inputTxt

tfidVec = pickle.load(open('tfidVec_model.mod', 'rb'))
loaded_model = pickle.load(open('svc_model.mod', 'rb'))

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello Flask-Heroku"

@app.route('/api', methods=['GET'])
def get_api():
    phishingUrl = request.args['url']

    splitText = splitUrlIntoToken(phishingUrl)
    stemText = textNormByStemType(splitText, snowball)

    data = ' '.join(stemText)
    df = pd.DataFrame([data], columns = ['data'])

    # print(df)
    vectorText = tfidVec.transform(df['data'])
    # print(vectorText.shape)

    result = loaded_model.predict(vectorText)
    # print(result)
    return result[0]

if __name__ == "__main__":
    app.run(threaded=True)
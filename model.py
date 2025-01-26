import pandas as pd
import nltk
import re
import json

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

data = pd.read_json('newdata.json')

print(data.head())

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    
    # Regex
    text = re.sub(r'[^a-z\s]', '', text)
    
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    return " ".join(words)

# Preprocess
data['processed_review'] = data['reviewText'].apply(preprocess_text)
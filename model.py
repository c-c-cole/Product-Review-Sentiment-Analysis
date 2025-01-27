import pandas as pd
import nltk
import re
import json

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef

import util

data = pd.read_json('newdata.json')

# I'm good to delete this line
# print(data.head())

# Keep commented ig
# nltk.download('stopwords')

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

vectorizer = TfidfVectorizer()
# vectorizer = CountVectorizer()

X = vectorizer.fit_transform(data['processed_review'])

# I'm good to delete this
# print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, data['overall'], test_size=0.2, random_state=42)


model = LogisticRegression(max_iter=1000)
# The following line penalizes the model for making mistakes on minority classes
# This can be useful because we have very imbalanced data (I think)
# The model focuses more on minority classes, 
#       which reduces accuracy (as it's more focused on the less likely cases)
# but does improve the mcc, which cares about the classes in a more equal way 
# So, accuracy goes from .69 -> .57 but mcc goes from .22 -> .27
# Overarching question is: "Which metrics do we care about? Which ones mean our model is better?"
#       As a side thing, we could try seeing what the metrics for something pre-built is (BERT, HuggingFace etc), and use their 
#       performance in these stats to get a general idea of what we're aiming for 
#       (eg, if BERT has low accuracy but high mcc, maybe that means the model is better, if we can justify this)
# model = LogisticRegression(class_weight='balanced', max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

comparison = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
print(comparison.sample(10))

util.test()

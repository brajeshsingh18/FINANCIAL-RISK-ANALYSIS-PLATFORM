import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from bs4 import BeautifulSoup
import warnings
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for resource in ('stopwords', 'punkt', 'wordnet', 'omw-1.4'):
    nltk.download(resource, quiet=True)
STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

CONTRACTIONS = {
    "ain't": "am not", "aren't": "are not", "can't": "can not", "can't've": "can not have",
    "could've": "could have", "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
    "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
    "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did",
    "how'll": "how will", "how's": "how is", "i'd": "i would", "i'll": "i will",
    "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
    "it'll": "it will", "it's": "it is", "let's": "let us", "ma'am": "madam",
    "might've": "might have", "mightn't": "might not", "must've": "must have",
    "mustn't": "must not", "needn't": "need not", "shan't": "shall not",
    "she'd": "she would", "she'll": "she will", "she's": "she is", "should've": "should have",
    "shouldn't": "should not", "that's": "that is", "there'd": "there would",
    "there's": "there is", "they'd": "they would", "they'll": "they will",
    "they're": "they are", "they've": "they have", "wasn't": "was not",
    "we'd": "we would", "we'll": "we will", "we're": "we are", "we've": "we have",
    "weren't": "were not", "what'll": "what will", "what're": "what are",
    "what's": "what is", "what've": "what have", "where's": "where is",
    "who'll": "who will", "who's": "who is", "won't": "will not", "would've": "would have",
    "wouldn't": "would not", "you'd": "you would", "you'll": "you will",
    "you're": "you are", "you've": "you have",
}

HTML_RE = re.compile(r'<.*?>')
URL_RE = re.compile(r'https?://\S+|www\.\S+')
NON_ALPHA_RE = re.compile(r'[^a-z\s]')
WS_RE = re.compile(r'\s+')
CONTRACTIONS_RE = re.compile(r'\b(' + '|'.join(re.escape(k) for k in CONTRACTIONS.keys()) + r')\b')

def preprocess(q, remove_stopwords: bool = False, lemmatize: bool = True) -> str:
    if not isinstance(q, str):
        return ""
    q = str(q).lower().strip()
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    q = q.replace('[math]', '')
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)
    if not isinstance(q, str):
        return ''
    q = BeautifulSoup(q, 'html.parser').get_text()
    q = URL_RE.sub(' ', q)
    q = CONTRACTIONS_RE.sub(lambda m: CONTRACTIONS[m.group(0)], q)
    q = NON_ALPHA_RE.sub(' ', q)
    q = WS_RE.sub(' ', q).strip()
    # if remove_stopwords:
    #     q = ' '.join(w for w in q.split() if w not in STOPWORDS)
    # if lemmatize:
    #     q = ' '.join(lemmatizer.lemmatize(w) for w in q.split())
    return q

def load_news_data()-> pd.DataFrame:
    df=pd.read_csv("data/processed/news/cleaned_news_data.csv")
    return df

def cleaning_news(df:pd.DataFrame)->pd.DataFrame:
    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")
    df["processed_text"] = (df["title"].str.strip()+ " "+ df["description"].str.strip())
    df["processed_text"]=df["processed_text"].apply(preprocess)
    return df

df=load_news_data()
df=cleaning_news(df)

from pathlib import Path

Path("data/feature_engineered/news").mkdir(
    parents=True,
    exist_ok=True
)

df.to_parquet(
    "data/feature_engineered/news/news_preprocessed.parquet",
    index=False
)

print(df.head()["processed_text"])



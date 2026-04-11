import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import pandas as pd
import numpy as np
import ast

# this could be a feature engineering function used when 
# the dataset is still raw. What I like to call pre feature
# engineering
def count_capital_chars(corpus: str):
    capital_chars = map(str.isupper, corpus)

    n_capital_chars = sum(capital_chars)

    return n_capital_chars

def count_capital_words(corpus: str):
    # will return a list of booleans
    # note we can directly sum a generator object produced by map
    capital_words = map(str.isupper, corpus.split())
    
    # recall that summing a list of booleans will return the total
    # number of true values since true values are directly 1 when casted
    # to an integer
    n_capital_words = sum(capital_words)

    return n_capital_words

def count_punctuations(corpus: str):
    punctuations = r"!\"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~"
    d = dict()
    for i in punctuations:
        key = f"{i}_punct_count"
        d[key] = corpus.count(i)
    return d

# def count_words_in_quotes(corpus: str):
#     x = re.findall(r'(\'.\'|\".\")', corpus)
#     count = 0
#     if len(x) == 0:
#         return 0
#     else:
#         for i in x:
#             t = i[1:-1]
#             count += count_words(t)
#         return count
    
def count_sent(corpus: str):
    sentences = sent_tokenize(corpus)
    n_sents = len(sentences)
    return n_sents

def count_htags(corpus: str):
    """
    counts the hashtag"s in a corpus e.g. "programming is 
    fun #helloworld, #programming" in this case would 
    match #helloworld and #programming as hashtags and 
    count these 
    """

    htags = re.findall(r'(#\w+)', corpus)
    n_htags = len(htags)
    return n_htags

def count_mentions(corpus: str):
    """
    counts the hashtag"s in a corpus e.g. "programming is 
    fun #helloworld, #programming" in this case would 
    match #helloworld and #programming as hashtags and 
    count these 
    """

    mentions = re.findall(r'(@\w+)', corpus)
    n_mentions = len(mentions)
    return n_mentions

def count_stopwords(corpus: str):
    """
    returns the number of stopwords in a row or corpus
    """

    stop_words = set(stopwords.words('english'))  
    word_tokens = corpus.split()
    sw_in_corpus = [word_token for word_token in word_tokens if word_token in stop_words]
    n_sw_in_corpus = len(sw_in_corpus)
    return n_sw_in_corpus

# this could be a feature engineering function used when 
# the dataset has been cleaned and preproessed. One I like 
# to call post feature engineering
def count_chars(corpus: str):
    n_chars = len(corpus)
    return n_chars

def count_words(corpus: str):
    n_words = len(corpus.split())
    return n_words

def count_unique_words(corpus: str):
    unique = set(corpus.split())
    n_unique = len(unique)
    return n_unique


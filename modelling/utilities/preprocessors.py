import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import pandas as pd
import numpy as np
import ast

def encode_features(X):
    """
    encodes the categorical features of a dataset into numerical values
    given the desired feature to encode and the input X to transform

    if shape of input is a one dimensional array and not a typical
    matrix reshape it to an m x 1 matrix instead by expanding its 
    dimensions. Usually this will be a the target column of 1 
    dimension. Otherwise use the ordinal encoder which is suitable for
    matrices like the set of independent variables of an X input

    used during training, validation, and testing/deployment (since
    encoder is saved and later used)
    """

    enc = LabelEncoder() if len(X.shape) < 2 else OrdinalEncoder(dtype=np.int64)
    enc_feats = enc.fit_transform(X)
    return enc_feats, enc

def normalize_train_cross(X_trains, X_cross, scaler='min_max'):
    """
    normalizes training and cross validation datasets using either
    a standard z-distribution or min max scaler

    args:
        X_trains - 
        X_cross - 
        scaler - scaler to use which can either be 'min_max' or 'standard'

    used during training, validation, and testing/deployment (since
    scaler is saved and later used)
    """

    temp = MinMaxScaler() if scaler is 'min_max' else StandardScaler()
    X_trains_normed = temp.fit_transform(X_trains)
    X_cross_normed = temp.transform(X_cross)

    return X_trains_normed, X_cross_normed, temp

def lower_words(corpus: str):
    """
    lowers all chars in corpus

    used during training, validation, and testing/deployment
    """
    # print(corpus)

    return corpus.lower()

def remove_contractions(text_string: str):
    """
    removes contractions and replace them e.g. don't becomes
    do not and so on

    used during training, validation, and testing/deployment
    """

    text_string = re.sub(r"don't", "do not ", text_string)
    text_string = re.sub(r"didn't", "did not ", text_string)
    text_string = re.sub(r"aren't", "are not ", text_string)
    text_string = re.sub(r"weren't", "were not", text_string)
    text_string = re.sub(r"isn't", "is not ", text_string)
    text_string = re.sub(r"can't", "cannot ", text_string)
    text_string = re.sub(r"doesn't", "does not ", text_string)
    text_string = re.sub(r"shouldn't", "should not ", text_string)
    text_string = re.sub(r"couldn't", "could not ", text_string)
    text_string = re.sub(r"mustn't", "must not ", text_string)
    text_string = re.sub(r"wouldn't", "would not ", text_string)

    text_string = re.sub(r"what's", "what is ", text_string)
    text_string = re.sub(r"that's", "that is ", text_string)
    text_string = re.sub(r"he's", "he is ", text_string)
    text_string = re.sub(r"she's", "she is ", text_string)
    text_string = re.sub(r"it's", "it is ", text_string)
    text_string = re.sub(r"that's", "that is ", text_string)

    text_string = re.sub(r"could've", "could have ", text_string)
    text_string = re.sub(r"would've", "would have ", text_string)
    text_string = re.sub(r"should've", "should have ", text_string)
    text_string = re.sub(r"must've", "must have ", text_string)
    text_string = re.sub(r"i've", "i have ", text_string)
    text_string = re.sub(r"we've", "we have ", text_string)

    text_string = re.sub(r"you're", "you are ", text_string)
    text_string = re.sub(r"they're", "they are ", text_string)
    text_string = re.sub(r"we're", "we are ", text_string)

    text_string = re.sub(r"you'd", "you would ", text_string)
    text_string = re.sub(r"they'd", "they would ", text_string)
    text_string = re.sub(r"she'd", "she would ", text_string)
    text_string = re.sub(r"he'd", "he would ", text_string)
    text_string = re.sub(r"it'd", "it would ", text_string)
    text_string = re.sub(r"we'd", "we would ", text_string)

    text_string = re.sub(r"you'll", "you will ", text_string)
    text_string = re.sub(r"they'll", "they will ", text_string)
    text_string = re.sub(r"she'll", "she will ", text_string)
    text_string = re.sub(r"he'll", "he will ", text_string)
    text_string = re.sub(r"it'll", "it will ", text_string)
    text_string = re.sub(r"we'll", "we will ", text_string)

    text_string = re.sub(r"\n't", " not ", text_string) #
    text_string = re.sub(r"\'s", " ", text_string) 
    text_string = re.sub(r"\'ve", " have ", text_string) #
    text_string = re.sub(r"\'re", " are ", text_string) #
    text_string = re.sub(r"\'d", " would ", text_string) #
    text_string = re.sub(r"\'ll", " will ", text_string) # 
    
    text_string = re.sub(r"i'm", "i am ", text_string)
    text_string = re.sub(r"%", " percent ", text_string)
    print(text_string)

    return text_string

def rem_non_alpha_num(corpus: str):
    """
    removes all non-alphanumeric values in the given corpus

    used during training, validation, and testing/deployment
    """
    # print(corpus)
    return re.sub(r"[^0-9a-zA-ZñÑ.\"]+", ' ', corpus)

def rem_numeric(corpus: str):
    # print(corpus)
    return re.sub(r"[0-9]+", ' ', corpus)

def capitalize(corpus: str):
    """
    capitalizes all individual words in the corpus

    used during training, validation, and testing/deployment
    """
    # print(corpus)
    return corpus.title()

def filter_valid(corpus: str, to_exclude: list=
        ['Crsp', 'Rpm', 'Mapsy', 'Cssgb', 'Chra', 
        'Mba', 'Es', 'Csswb', 'Cphr', 'Clssyb', 
        'Cssyb', 'Mdrt', 'Ceqp', 'Icyb']):
    
    """
    a function that filters only valid names and
    joins only the words that is valid in the profile
    name e.g. 'Christian Cachola Chrp Crsp'
    results only in 'Christian Cachola'

    used during training, validation, and testing/deployment
    """

    # filter and remove the words in the sequence
    # included in list of words that are invalid
    sequence = corpus.split()
    filt_sequence = list(filter(lambda word: word not in to_exclude, sequence))
    
    # join the filtered words
    temp = " ".join(filt_sequence)

    return temp

def partition_corpus(corpus: str):
    """
    splits a corpus like name, phrase, sentence, 
    paragraph, or corpus into individual strings

    used during training, validation, and testing/deployment
    """
    # print(corpus)

    return corpus.split()

def rem_stop_words(corpus: str, other_exclusions: list=["#ff", "ff", "rt", "amp"]):
    """
    removes stop words of a given corpus

    used during training, validation, and testing/deployment
    """

    # get individual words of corpus
    words = corpus.split()

    # extract stop words and if provided with other exclusions
    # extend this to the list of stop words
    stop_words = stopwords.words('english')
    stop_words.extend(other_exclusions)

    # include only the words not in the list of stop words
    words = [word for word in words if not word in stop_words]

    # rejoin the individual words of the now removed stop words
    corpus = " ".join(words)
    # print(corpus)

    return corpus

def stem_corpus_words(corpus: str):
    """
    stems individual words of a given corpus

    used during training, validation, and testing/deployment
    """
    # get individual words of corpus
    words = corpus.split()

    # stem each individual word in corpus
    snowball = SnowballStemmer("english", ignore_stopwords=True)
    words = [snowball.stem(word) for word in words]

    # rejoin the now lemmatized individual words
    corpus = " ".join(words)
    # print(corpus)

    return corpus

def lemmatize_corpus_words(corpus: str):
    """
    lemmatizes individual words of a given corpus

    used during training, validation, and testing/deployment
    """

    # get individual words of corpus
    words = corpus.split()

    # lemmatize each individual word in corpus
    wordnet = WordNetLemmatizer()
    words = [wordnet.lemmatize(word) for word in words]

    # rejoin the now lemmatized individual words
    corpus = " ".join(words)
    # print(corpus)

    return corpus

def clean_tweets(corpus):
    """
    Accepts a text string or corpus and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned

    used during training, validation, and testing/deployment
    """

    space_pattern = '\s+'

    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
    '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    
    mention_regex = '@[\w\-]+'

    text_string = re.sub(space_pattern, ' ', corpus)
    text_string = re.sub(giant_url_regex, '', text_string)
    text_string = re.sub(mention_regex, '', text_string)
    # print(text_string)
    
    return text_string

def strip_final_corpus(corpus):
    """
    ideally used in final phase of preprocessing corpus/text string
    which strips the final corpus of all its white spaces

    used during training, validation, and testing/deployment
    """
    
    return corpus.strip()

def join_word_list(word_list):
    return " ".join(word_list)

def string_list_to_list(column: pd.Series):
    """
    saving df to csv with a column that is of a list data type
    does not preserve its type and is converted instead to an
    str so convert first str to list or series "["a", "b", 
    "hello"]" to ["a", "b", "hello"]

    used only during training and validation
    """
    column = column.apply(lambda comment: ast.literal_eval(comment))

    return column

def flatten_series_of_lists(column: pd.Series):
    """this converts the series or column of a df
    of lists to a flattened version of it

    used only during training and validation
    """

    return pd.Series([item for sublist in column for item in sublist])

def decode_one_hot(Y_preds):
    """
    whether for image, sentiment, or general classification
    this function takes in an (m x 1) or (m x n_y) matrix of
    the predicted values of a classifier

    e.g. if binary the input Y_preds would be 
    [[0 1]
    [1 0]
    [1 0]
    [0 1]
    [1 0]
    [1 0]]

    if multi-class the Y_preds for instance would be...

    [[0 0 0 1]
    [1 0 0 0
    [0 0 1 0]
    ...
    [0 1 0 0]]

    what this function does is it takes the argmax along the
    1st dimension/axis, and once decoded would be just two
    binary categorial values e.g. 0 or 1 or if multi-class
    0, 1, 2, or 3

    main args:
        Y_preds - 

    used during training, validation, and testing/deployment
    """

    # check if Y_preds is multi-class by checking if shape
    # of matrix is (m, n_y), (m, m, n_y), or just m
    if len(Y_preds.shape) >= 2:
        # take the argmax if Y_preds are multi labeled
        sparse_categories = np.argmax(Y_preds, axis=1)

    return sparse_categories

def re_encode_sparse_labels(sparse_labels, new_labels: list=['DER', 'APR', 'NDG']):
    """
    sometimes a dataset will only have its target values 
    be sparse values such as 0, 1, 2 right at the start
    so this function re encodes these sparse values/labels
    to a more understandable representation

    upon reencoding this can be used by other encoders
    such as encode_features() which allows us to save
    the encoder to be used later on in model training

    used only during training and validation
    """

    # return use as index the sparse_labels to the new labels
    v_func = np.vectorize(lambda sparse_label: new_labels[sparse_label])
    re_encoded_labels = v_func(sparse_labels)

    return re_encoded_labels

def translate_labels(labels, translations: dict={'DER': 'Derogatory', 
                                                 'NDG': 'Non-Derogatory', 
                                                 'HOM': 'Homonym', 
                                                 'APR': 'Appropriative'}):
    """
    transforms an array of shortened versions of the
    labels e.g. array(['DER', 'NDG', 'DER', 'HOM', 'APR', 
    'DER', 'NDG', 'HOM', 'HOM', 'HOM', 'DER', 'DER', 'NDG', 
    'DER', 'HOM', 'DER', 'APR', 'APR', 'DER'] to a more lengthened
    and understandable version to potentially send back to client
    e.g. array(['DEROGATORY', NON-DEROGATORY, 'DEROGATORY', 'HOMONYM',
    'APPROPRIATIVE', ...])

    used during training, validation, and testing/deployment
    """

    v_func = np.vectorize(lambda label: translations[label])
    translated_labels = v_func(labels)
    return translated_labels

def vectorize_sent(X_trains, X_tests):
    """
    vectorizes a set of sentences either using term frequency
    inverse document frequency or by count/frequency of a word
    in the sentence/s

    returns a vectorizer object for later saving
    """

    vectorizer = TfidfVectorizer()
    vectorizer.fit(X_trains)
    X_trains_vec = vectorizer.transform(X_trains).toarray()
    X_tests_vec = vectorizer.transform(X_tests).toarray()
    
    return X_trains_vec, X_tests_vec, vectorizer
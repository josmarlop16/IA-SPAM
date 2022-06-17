import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import pickle as cPickle

# Creating DataFrames
dataframe1 = pd.read_csv("legit.csv")
dataframe2 = pd.read_csv("spam.csv")
dataframe = pd.concat([dataframe1, dataframe2])

# remove duplicate entries
dataframe.drop_duplicates(inplace=True)

# remove null values
dataframe.dropna(inplace=True)

# Function that obtain a bag of tokenized words
def bag_of_words_tokenizer(email):
    # Remove punctuation
    no_punctuation = [char for char in email if char not in string.punctuation]
    no_punctuation = "".join(no_punctuation)
    # Remove stopwords
    clean_words = [
        word
        for word in no_punctuation.split()
        if word.lower() not in stopwords.words("english") and word.isalpha()
    ]
    stemmed_words = stem_words(clean_words)
    lemmatized_words = lemmatize_words(stemmed_words)
    return lemmatized_words


# import lemmatizer
lemmanator = WordNetLemmatizer()

# function that lemmatizes words using WordNetLemmatizer from nltk
def lemmatize_words(words):
    lemmatized_words = []
    for word in words:
        lemmatized_words.append(lemmanator.lemmatize(word))
    return lemmatized_words


# import stemmer
Stemmerator = PorterStemmer()

# function that stems words using PorterStemmer from nltk
def stem_words(words):
    stemmed_words = []
    for word in words:
        stemmed_words.append(Stemmerator.stem(word))
    return stemmed_words


# training the vectorizer
def train_countvectorizer(dataframe):
    # Create a bag of words vectorizer
    count_vectorizer = CountVectorizer(analyzer=bag_of_words_tokenizer)
    # Fit the bag of words vectorizer to the training set
    bag_of_words = count_vectorizer.fit_transform(dataframe)
    return bag_of_words


# Create a bag of words vectorizer and fit it to the dataframe
cv = CountVectorizer(analyzer=bag_of_words_tokenizer)
bag_of_words = cv.fit_transform(dataframe["Email"])

# Serialization with pickle
with open("vectorizer.pickle", "wb") as f:
    cPickle.dump(cv, f, protocol=-1)

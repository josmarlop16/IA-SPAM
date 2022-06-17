import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
import pickle as cPickle

print(" \n .·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·. kNN Classifier .·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.")

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


Stemmerator = PorterStemmer()

# function that stems words using PorterStemmer from nltk
def stem_words(words):
    stemmed_words = []
    for word in words:
        stemmed_words.append(Stemmerator.stem(word))
    return stemmed_words


# Create a TF-IDF Vectorizer Object
tfidf_vectorizer = TfidfVectorizer(analyzer=bag_of_words_tokenizer)

# Fit the TF-IDF vectorizer to the training data
features = tfidf_vectorizer.fit_transform(dataframe["Email"])
# # Convert to dataframe
# features = pd.DataFrame(
#     features.todense(), columns=tfidf_vectorizer.get_feature_names()
# )
# features.to_csv("features.csv", index=False)
# features = pd.read_csv("features.csv")
# Split the data into training and testing sets (60% training and 40% testing)
X_train, X_test, y_train, y_test = train_test_split(
    features, dataframe["isSpam"], test_size=0.4, random_state=42
)

# KNN Classifier
Knn_Classifier_n10 = KNeighborsClassifier(n_neighbors=10).fit(X_train, y_train)
Knn_Classifier_n15 = KNeighborsClassifier(n_neighbors=15).fit(X_train, y_train)
Knn_Classifier_n20 = KNeighborsClassifier(n_neighbors=20).fit(X_train, y_train)
# Let's see Classification report
print(
    "Classification report (n=10): \n",
    classification_report(y_test, Knn_Classifier_n10.predict(X_test)),
)
print(
    "Classification report (n=15): \n",
    classification_report(y_test, Knn_Classifier_n15.predict(X_test)),
)
print(
    "Classification report (n=20): \n",
    classification_report(y_test, Knn_Classifier_n20.predict(X_test)),
)
# Let's see Confusion Matrix
print(
    "Confusion Matrix (n=10): \n ",
    confusion_matrix(y_test, Knn_Classifier_n10.predict(X_test)),
)
print(
    "Confusion Matrix (n=15): \n ",
    confusion_matrix(y_test, Knn_Classifier_n15.predict(X_test)),
)
print(
    "Confusion Matrix (n=20): \n ",
    confusion_matrix(y_test, Knn_Classifier_n20.predict(X_test)),
)
# Let's see Accuracy score
print(
    "Accuracy score (n=10): \n",
    accuracy_score(y_test, Knn_Classifier_n10.predict(X_test)),
)
print(
    "Accuracy score (n=15): \n",
    accuracy_score(y_test, Knn_Classifier_n15.predict(X_test)),
)
print(
    "Accuracy score (n=20): \n",
    accuracy_score(y_test, Knn_Classifier_n20.predict(X_test)),
)
# Let's see Precision score
print(
    "Precision score (n=10): \n",
    precision_score(y_test, Knn_Classifier_n10.predict(X_test)),
)
print(
    "Precision score (n=15): \n",
    precision_score(y_test, Knn_Classifier_n15.predict(X_test)),
)
print(
    "Precision score (n=20): \n",
    precision_score(y_test, Knn_Classifier_n20.predict(X_test)),
)
# Let's see Recall score
print(
    "Recall score (n=10): \n", recall_score(y_test, Knn_Classifier_n10.predict(X_test))
)
print(
    "Recall score (n=15): \n", recall_score(y_test, Knn_Classifier_n15.predict(X_test))
)
print(
    "Recall score (n=20): \n", recall_score(y_test, Knn_Classifier_n20.predict(X_test))
)
# Let's see F1 score
print("F1 score (n=10): \n", f1_score(y_test, Knn_Classifier_n10.predict(X_test)))
print("F1 score (n=15): \n", f1_score(y_test, Knn_Classifier_n15.predict(X_test)))
print("F1 score (n=20): \n", f1_score(y_test, Knn_Classifier_n20.predict(X_test)))

print(
    ".·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·..·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·."
)
print(
    "¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·"
)
print("\n")

# Serialization with Pickle
with open("knn.pickle", "wb") as f:
    cPickle.dump(Knn_Classifier_n15, f, protocol=-1)

with open("knnvectorizer.pickle", "wb") as f:
    cPickle.dump(tfidf_vectorizer, f, protocol=-1)

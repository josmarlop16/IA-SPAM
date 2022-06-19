import pandas as pd
from vectorizer import train_countvectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle as cPickle

# nltk.download("stopwords")
# nltk.download("words")
# nltk.download("wordnet")
# nltk.download("omw-1.4")

## Los metodos y lineas de codigo comentadas son para la version de prueba de la practica ##

print(
    ".·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·|| Naive Bayes Multinomial Classifier ||·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·."
)

# Creating DataFrames
dataframe1 = pd.read_csv("legit.csv")
dataframe2 = pd.read_csv("spam.csv")
dataframe = pd.concat([dataframe1, dataframe2])

# remove duplicate entries
dataframe.drop_duplicates(inplace=True)

# remove null values
dataframe.dropna(inplace=True)

# Function that obtain a bag of tokenized words
bag_of_words = train_countvectorizer(dataframe["Email"])

# Split the data into training and testing sets (60% training and 40% testing)
# X stands for feature training dataset, y stands for target training dataset
X_train, X_test, y_train, y_test = train_test_split(
    bag_of_words, dataframe["isSpam"], test_size=0.4, random_state=42, shuffle=True
)

# Create a Polynomial Naive Bayes classifier and train it with the datasets
# Softened hyperparameter is alpha
NB_classifier_alpha_1 = MultinomialNB().fit(X_train, y_train)
NB_classifier_alpha_5 = MultinomialNB(alpha=5).fit(X_train, y_train)
NB_classifier_alpha_10 = MultinomialNB(alpha=10).fit(X_train, y_train)

# Let's see the accuracy of the model
print(
    "Classification report (alpha=1): \n",
    classification_report(y_test, NB_classifier_alpha_1.predict(X_test)),
)
print(
    "Classification report (alpha=5): \n",
    classification_report(y_test, NB_classifier_alpha_5.predict(X_test)),
)
print(
    "Classification report (alpha=10): \n",
    classification_report(y_test, NB_classifier_alpha_10.predict(X_test)),
)

# Let's see Confusion Matrix
print(
    "Confusion Matrix (alpha=1): \n",
    confusion_matrix(y_test, NB_classifier_alpha_1.predict(X_test)),
)
print(
    "Confusion Matrix (alpha=5): \n",
    confusion_matrix(y_test, NB_classifier_alpha_5.predict(X_test)),
)
print(
    "Confusion Matrix (alpha=10): \n",
    confusion_matrix(y_test, NB_classifier_alpha_10.predict(X_test)),
)

# Let's see Accuracy score
print(
    "Accuracy score (alpha=1): \n",
    accuracy_score(y_test, NB_classifier_alpha_1.predict(X_test)),
)
print(
    "Accuracy score (alpha=5): \n",
    accuracy_score(y_test, NB_classifier_alpha_5.predict(X_test)),
)
print(
    "Accuracy score (alpha=10): \n",
    accuracy_score(y_test, NB_classifier_alpha_10.predict(X_test)),
)

# Let's see Precision score
print(
    "Precision score (alpha=1): \n",
    precision_score(y_test, NB_classifier_alpha_1.predict(X_test)),
)
print(
    "Precision score (alpha=5): \n",
    precision_score(y_test, NB_classifier_alpha_5.predict(X_test)),
)
print(
    "Precision score (alpha=10): \n",
    precision_score(y_test, NB_classifier_alpha_10.predict(X_test)),
)

# Let's see Recall score
print(
    "Recall score (alpha=1): \n",
    recall_score(y_test, NB_classifier_alpha_1.predict(X_test)),
)
print(
    "Recall score (alpha=5): \n",
    recall_score(y_test, NB_classifier_alpha_5.predict(X_test)),
)
print(
    "Recall score (alpha=10): \n",
    recall_score(y_test, NB_classifier_alpha_10.predict(X_test)),
)

# Let's see F1 score
print("F1 score (alpha=1): \n", f1_score(y_test, NB_classifier_alpha_1.predict(X_test)))
print("F1 score (alpha=5): \n", f1_score(y_test, NB_classifier_alpha_5.predict(X_test)))
print(
    "F1 score (alpha=10): \n", f1_score(y_test, NB_classifier_alpha_10.predict(X_test))
)

print(
    ".·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·."
)
print(
    "¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨·.·¨"
)
print("\n")

# # Serialization with Pickle
# with open("nbm.pickle", "wb") as f:
#     cPickle.dump(NB_classifier_alpha_10, f, protocol=-1)
